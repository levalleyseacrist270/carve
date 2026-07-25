"""Objective quality gates for generated video edits.

Five per-clip checks guard structure and motion outside the edited region:
masked DINO structural similarity, motion-histogram distance (MHD),
flow-based displacement of non-edit regions, flow-jump rate (FJR), and a
capture-quality match against a lightweight probe. All measurements run at a
fixed square analysis resolution.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import yaml

from carve.core import AccidentSpec, Branch, EnvironmentSpec, GateResult

_EPS = 1e-8


@dataclasses.dataclass(frozen=True)
class ObjectiveGateConfig:
    """Thresholds and measurement settings for the objective gates."""

    dino_similarity_min: float = 0.92
    dino_model: str = "facebook/dinov2-base"
    dino_frame_samples: int = 8
    mhd_max: float = 0.25
    orientation_bins: int = 8
    magnitude_bins: int = 8
    magnitude_clip: float = 20.0
    displacement_px_max: float = 4.0
    displacement_frame_samples: int = 8
    fjr_max: float = 0.12
    gamma_f: float | None = None
    gamma_f_percentile: float = 95.0
    analysis_resolution: int = 224

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ObjectiveGateConfig":
        """Load the ``gates.objective`` block plus the analysis resolution."""
        with open(path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        block = dict(raw.get("gates", {}).get("objective", {}))
        block["analysis_resolution"] = raw.get("analysis", {}).get(
            "resolution", cls.analysis_resolution
        )
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in block.items() if k in fields})


@dataclasses.dataclass
class EditCandidate:
    """A generated edit awaiting quality gating.

    Attributes:
        source_path: Path of the real source clip.
        edited_path: Path of the generated edit.
        branch: Quartet branch of the edit.
        edit_mask: Boolean edit-region mask, either static ``(H, W)`` or
            per-frame ``(T, H, W)``. ``None`` means the edit is a global
            appearance change and the whole frame is treated as the
            preservation region.
        environment: Requested environment of the edit; for accident-only
            edits this is the source environment, which must be preserved.
        accident: Requested accident script, or ``None`` for accident-free
            branches.
    """

    source_path: str
    edited_path: str
    branch: Branch
    edit_mask: np.ndarray | None
    environment: EnvironmentSpec
    accident: AccidentSpec | None = None


def load_frames(path: str | Path, resolution: int, max_frames: int | None = None) -> np.ndarray:
    """Decode a clip into a ``(T, H, W, 3)`` uint8 RGB array at ``resolution``."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise IOError(f"cannot open video: {path}")
    frames: list[np.ndarray] = []
    while max_frames is None or len(frames) < max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        frame = cv2.resize(frame, (resolution, resolution), interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise IOError(f"no decodable frames in: {path}")
    return np.stack(frames)


def _non_edit_masks(
    edit_mask: np.ndarray | None, num_frames: int, resolution: int
) -> np.ndarray:
    """Return per-frame boolean masks of the region outside the edit."""
    if edit_mask is None:
        return np.ones((num_frames, resolution, resolution), dtype=bool)
    mask = np.asarray(edit_mask)
    if mask.ndim == 2:
        mask = np.broadcast_to(mask[None], (num_frames,) + mask.shape)
    if mask.shape[0] != num_frames:
        raise ValueError(
            f"per-frame mask has {mask.shape[0]} frames, clip has {num_frames}"
        )
    resized = np.stack(
        [
            cv2.resize(
                frame.astype(np.uint8), (resolution, resolution), interpolation=cv2.INTER_NEAREST
            )
            for frame in mask
        ]
    )
    return ~resized.astype(bool)


def _gray(frames: np.ndarray) -> np.ndarray:
    return np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames])


def _flow(prev: np.ndarray, nxt: np.ndarray) -> np.ndarray:
    """Farneback dense optical flow between two grayscale frames."""
    return cv2.calcOpticalFlowFarneback(
        prev, nxt, None, pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )


def _sample_indices(num_frames: int, count: int) -> np.ndarray:
    return np.unique(np.linspace(0, num_frames - 1, num=min(count, num_frames)).astype(int))


class DinoEmbedder:
    """Frame embedder built on a self-supervised vision backbone.

    The model is loaded lazily on first use so that pipelines which never
    reach the structural gate do not pay for the download.
    """

    def __init__(self, model_name: str = "facebook/dinov2-base", device: str | None = None):
        self._model_name = model_name
        self._device = device
        self._model = None
        self._processor = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as error:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "the structural similarity gate requires torch and transformers"
            ) from error
        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._processor = AutoImageProcessor.from_pretrained(self._model_name)
        self._model = AutoModel.from_pretrained(self._model_name).to(device).eval()
        self._device = device

    def embed(self, frames: np.ndarray) -> np.ndarray:
        """Embed ``(N, H, W, 3)`` uint8 RGB frames into L2-normalized vectors."""
        self._ensure_model()
        import torch

        with torch.no_grad():
            inputs = self._processor(images=list(frames), return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            pooled = self._model(**inputs).last_hidden_state.mean(dim=1)
            pooled = torch.nn.functional.normalize(pooled, dim=-1)
        return pooled.cpu().numpy()


def masked_dino_similarity(
    source: np.ndarray,
    edited: np.ndarray,
    non_edit: np.ndarray,
    config: ObjectiveGateConfig,
    embedder: DinoEmbedder | None = None,
) -> float:
    """Mean cosine similarity of masked frame embeddings, edited vs. source.

    The edit region is zeroed in both clips before embedding, so the score
    reflects structural agreement of the preservation region only.
    """
    embedder = embedder or DinoEmbedder(config.dino_model)
    count = min(len(source), len(edited), len(non_edit))
    indices = _sample_indices(count, config.dino_frame_samples)
    keep = non_edit[indices, ..., None].astype(np.uint8)
    src = embedder.embed(source[indices] * keep)
    edt = embedder.embed(edited[indices] * keep)
    return float(np.mean(np.sum(src * edt, axis=-1)))


def _flow_histogram(flow: np.ndarray, region: np.ndarray, config: ObjectiveGateConfig) -> np.ndarray:
    """Normalized magnitude-orientation histogram of flow inside ``region``."""
    fx, fy = flow[..., 0][region], flow[..., 1][region]
    magnitude = np.clip(np.hypot(fx, fy), 0.0, config.magnitude_clip)
    orientation = np.mod(np.arctan2(fy, fx), 2.0 * np.pi)
    hist, _, _ = np.histogram2d(
        orientation,
        magnitude,
        bins=[config.orientation_bins, config.magnitude_bins],
        range=[[0.0, 2.0 * np.pi], [0.0, config.magnitude_clip]],
    )
    return hist / max(hist.sum(), _EPS)


def _chi_square(a: np.ndarray, b: np.ndarray) -> float:
    return float(0.5 * np.sum((a - b) ** 2 / (a + b + _EPS)))


def motion_histogram_distance(
    source: np.ndarray,
    edited: np.ndarray,
    non_edit: np.ndarray,
    config: ObjectiveGateConfig,
) -> float:
    """Per-clip MHD: mean chi-squared distance between flow histograms.

    Histograms are computed outside the edit region for each adjacent-frame
    transition of the edited clip and its source, then compared per frame and
    averaged.
    """
    src_gray, edt_gray = _gray(source), _gray(edited)
    count = min(len(src_gray), len(edt_gray), len(non_edit))
    distances = []
    for t in range(count - 1):
        region = non_edit[t] & non_edit[t + 1]
        if not region.any():
            continue
        hist_src = _flow_histogram(_flow(src_gray[t], src_gray[t + 1]), region, config)
        hist_edt = _flow_histogram(_flow(edt_gray[t], edt_gray[t + 1]), region, config)
        distances.append(_chi_square(hist_src, hist_edt))
    return float(np.mean(distances)) if distances else 0.0


def displacement_outside_edit(
    source: np.ndarray,
    edited: np.ndarray,
    non_edit: np.ndarray,
    config: ObjectiveGateConfig,
) -> float:
    """Median flow displacement (px) of non-edit regions between matched frames.

    Dense flow is computed from each sampled source frame to the temporally
    matched edited frame; the median magnitude over non-edit pixels measures
    how far preserved content moved under the edit.
    """
    src_gray, edt_gray = _gray(source), _gray(edited)
    count = min(len(src_gray), len(edt_gray), len(non_edit))
    magnitudes = []
    for t in _sample_indices(count, config.displacement_frame_samples):
        region = non_edit[t]
        if not region.any():
            continue
        flow = _flow(src_gray[t], edt_gray[t])
        magnitudes.append(np.hypot(flow[..., 0], flow[..., 1])[region])
    if not magnitudes:
        return 0.0
    return float(np.median(np.concatenate(magnitudes)))


def _mean_flow_changes(gray: np.ndarray, non_edit: np.ndarray) -> np.ndarray:
    """Mean absolute flow-magnitude change outside the edit, per transition."""
    count = min(len(gray), len(non_edit))
    magnitudes = [
        np.hypot(f[..., 0], f[..., 1])
        for f in (_flow(gray[t], gray[t + 1]) for t in range(count - 1))
    ]
    changes = []
    for t in range(len(magnitudes) - 1):
        region = non_edit[t] & non_edit[t + 1] & non_edit[min(t + 2, count - 1)]
        if not region.any():
            continue
        changes.append(float(np.mean(np.abs(magnitudes[t + 1][region] - magnitudes[t][region]))))
    return np.asarray(changes)


def flow_jump_rate(
    edited: np.ndarray,
    non_edit: np.ndarray,
    gamma_f: float,
) -> float:
    """FJR: fraction of adjacent-frame transitions with an abrupt flow change.

    A transition counts as a jump when the mean absolute change of flow
    magnitude outside the edit between consecutive flow fields exceeds
    ``gamma_f``.
    """
    changes = _mean_flow_changes(_gray(edited), non_edit)
    if changes.size == 0:
        return 0.0
    return float(np.mean(changes > gamma_f))


def calibrate_gamma_f(
    real_clips: Iterable[str | Path],
    percentile: float = 95.0,
    resolution: int = 224,
    max_frames: int | None = 64,
) -> float:
    """Calibrate the flow-jump threshold on real clips held out from generation.

    Collects the mean absolute flow-magnitude change of every adjacent-frame
    transition over full frames of the given real clips and returns the
    requested percentile.
    """
    values: list[np.ndarray] = []
    for path in real_clips:
        frames = load_frames(path, resolution, max_frames)
        full = np.ones((len(frames),) + frames.shape[1:3], dtype=bool)
        values.append(_mean_flow_changes(_gray(frames), full))
    pooled = np.concatenate([v for v in values if v.size]) if values else np.array([])
    if pooled.size == 0:
        raise ValueError("no flow transitions found in the calibration clips")
    return float(np.percentile(pooled, percentile))


def run_objective_gates(
    candidate: EditCandidate,
    config: ObjectiveGateConfig,
    embedder: DinoEmbedder | None = None,
    probe=None,
    gamma_f: float | None = None,
) -> GateResult:
    """Run the five objective gates on a candidate edit.

    Args:
        candidate: Edit to evaluate.
        config: Gate thresholds and measurement settings.
        embedder: Optional shared frame embedder for the structural gate.
        probe: Optional trained capture-quality probe; when omitted, the
            capture gate is recorded as not evaluated instead of passing
            silently.
        gamma_f: Calibrated flow-jump threshold; overrides ``config.gamma_f``.

    Returns:
        A stage ``"objective"`` result whose details carry the value,
        threshold, and verdict of every gate.
    """
    resolved_gamma = gamma_f if gamma_f is not None else config.gamma_f
    if resolved_gamma is None:
        raise ValueError(
            "gamma_f is not set; run calibrate_gamma_f on held-out real clips first"
        )
    resolution = config.analysis_resolution
    source = load_frames(candidate.source_path, resolution)
    edited = load_frames(candidate.edited_path, resolution)
    count = min(len(source), len(edited))
    source, edited = source[:count], edited[:count]
    non_edit = _non_edit_masks(candidate.edit_mask, count, resolution)

    dino = masked_dino_similarity(source, edited, non_edit, config, embedder)
    mhd = motion_histogram_distance(source, edited, non_edit, config)
    displacement = displacement_outside_edit(source, edited, non_edit, config)
    fjr = flow_jump_rate(edited, non_edit, resolved_gamma)

    details: dict = {
        "branch": candidate.branch.value,
        "dino_similarity": {
            "value": dino, "threshold": config.dino_similarity_min,
            "passed": dino >= config.dino_similarity_min,
        },
        "mhd": {"value": mhd, "threshold": config.mhd_max, "passed": mhd <= config.mhd_max},
        "displacement_px": {
            "value": displacement, "threshold": config.displacement_px_max,
            "passed": displacement <= config.displacement_px_max,
        },
        "fjr": {
            "value": fjr, "threshold": config.fjr_max, "gamma_f": resolved_gamma,
            "passed": fjr <= config.fjr_max,
        },
    }
    if probe is not None:
        from carve.gates.probe import predict

        predicted = predict(probe, edited)
        details["capture_quality"] = {
            "evaluated": True,
            "predicted": predicted,
            "requested": candidate.environment.capture_quality,
            "passed": predicted == candidate.environment.capture_quality,
        }
    else:
        details["capture_quality"] = {"evaluated": False, "passed": None}

    evaluated = [gate["passed"] for gate in details.values()
                 if isinstance(gate, dict) and gate.get("passed") is not None]
    return GateResult(passed=all(evaluated), stage="objective", details=details)
