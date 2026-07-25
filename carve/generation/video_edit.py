"""Phase two: reference-guided video editing and full quartet assembly.

Each quartet needs three generated branches. The environment branch edits
nuisance appearance under a strict minimal-change instruction guided by the
phase-one reference image; the accident branch inserts the scripted collision
under the source environment and omits the reference; the joint branch
realizes both, sharing the same reference image as the environment branch.
Every branch must pass the objective gates and the validity panel; the
quartet is dropped on the first failure and the failing stage is recorded.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Sequence

import cv2
import numpy as np

from carve.core import (
    AccidentSpec,
    Branch,
    EnvironmentSpec,
    GateResult,
    QuartetRecord,
)
from carve.gates.objective import (
    DinoEmbedder,
    EditCandidate,
    ObjectiveGateConfig,
    load_frames,
    run_objective_gates,
)
from carve.gates.panel import EvidencePacket, PanelRubric, run_panel_gate
from carve.generation.layout_cues import (
    Box,
    accident_region_mask,
    render_trajectory_sketch,
)
from carve.generation.providers import (
    ImageEditBackend,
    VideoEditBackend,
    VideoEditRequest,
    VlmJudge,
)
from carve.generation.reference_image import ReferenceResult, generate_reference
from carve.generation.scripts_builder import (
    accident_script,
    branch_instruction,
    environment_script,
    onset_window,
)

if TYPE_CHECKING:  # avoids importing torch when the probe is unused
    from carve.gates.probe import CaptureQualityProbe

_GENERATED_BRANCHES = (Branch.VE, Branch.VA, Branch.VAE)


@dataclasses.dataclass
class SourceUnit:
    """A real source clip eligible for quartet generation.

    Attributes:
        source_id: Stable identifier of the source unit.
        dataset: Origin dataset name.
        split: Partition of the source unit.
        path: Clip file path.
        environment: Environment of the real footage; the accident branch
            must preserve it, including its capture quality.
        flags: Source-level feasibility measurements from the manifest, for
            example ``participant_scale`` or ``visible_road_fraction``.
    """

    source_id: str
    dataset: str
    split: str
    path: str
    environment: EnvironmentSpec
    flags: dict = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class Backends:
    """Generation backends plus optional perception callables."""

    image: ImageEditBackend
    video: VideoEditBackend
    lane_segmenter: Callable[[np.ndarray], np.ndarray] | None = None
    vehicle_detector: Callable[[np.ndarray], Sequence[Box]] | None = None


@dataclasses.dataclass
class GateSuite:
    """Everything needed to run the three quality-control layers."""

    objective: ObjectiveGateConfig
    gamma_f: float
    rubric: PanelRubric
    panel_judges: Sequence[VlmJudge]
    reference_judge: VlmJudge
    probe: "CaptureQualityProbe | None" = None
    embedder: DinoEmbedder | None = None


def _timestamp(index: int, fps: int) -> str:
    return f"f{index:03d} t={index / fps:.2f}s"


def _annotate(frame: np.ndarray, label: str) -> np.ndarray:
    out = frame.copy()
    cv2.putText(out, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3)
    cv2.putText(out, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    return out


def _save(image: np.ndarray, path: Path) -> str:
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return str(path)


def _key_indices(count: int, num: int = 8) -> np.ndarray:
    return np.unique(np.linspace(0, count - 1, num=min(num, count)).astype(int))


def build_evidence_packet(
    candidate: EditCandidate,
    script_text: str,
    target_label: int,
    out_dir: str | Path,
    fps: int = 16,
    reference_image: str | None = None,
    onset: tuple[int, int] | None = None,
    participant_boxes: Sequence[Box] = (),
    impact_box: Box | None = None,
    resolution: int = 224,
) -> EvidencePacket:
    """Render the standardized evidence packet for one candidate.

    The packet contains a timestamped key-frame grid of the edit, matched
    source/edited thumbnail rows, an edit-region overlay with participant and
    impact boxes when the edit is localized, and a frame-difference motion
    visualization. For positive branches, grid sampling includes frames
    around the scripted onset window.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    source = load_frames(candidate.source_path, resolution)
    edited = load_frames(candidate.edited_path, resolution)
    count = min(len(source), len(edited))
    indices = _key_indices(count)
    if onset is not None:
        anchors = np.clip(np.arange(onset[0] - 2, onset[1] + 3, 2), 0, count - 1)
        indices = np.unique(np.concatenate([indices, anchors]))

    grid_rows = [_annotate(edited[i], _timestamp(int(i), fps)) for i in indices]
    grid = np.concatenate(grid_rows, axis=1)
    pair = np.concatenate(
        [
            np.concatenate([_annotate(source[i], "src") for i in indices], axis=1),
            np.concatenate([_annotate(edited[i], "edit") for i in indices], axis=1),
        ],
        axis=0,
    )

    mask_path: str | None = None
    if candidate.edit_mask is not None:
        mask = candidate.edit_mask if candidate.edit_mask.ndim == 2 else candidate.edit_mask[0]
        scale = resolution / max(1, mask.shape[1])
        mask = cv2.resize(mask.astype(np.uint8), (resolution, resolution)) > 0
        overlay = edited[count // 2].copy()
        overlay[mask] = (0.55 * overlay[mask] + np.array([114.75, 0.0, 0.0])).astype(np.uint8)
        for box in [*participant_boxes, *([impact_box] if impact_box else [])]:
            x1, y1, x2, y2 = (int(round(v * scale)) for v in box)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 0), 1)
        mask_path = _save(overlay, out / "mask_overlay.png")

    diff = np.mean(
        [np.abs(edited[i + 1].astype(np.int16) - edited[i].astype(np.int16)).mean(axis=-1)
         for i in range(count - 1)],
        axis=0,
    )
    heat = cv2.applyColorMap(
        cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_JET
    )
    motion_path = str(out / "motion_evidence.png")
    cv2.imwrite(motion_path, heat)

    return EvidencePacket(
        branch=candidate.branch,
        script_text=script_text,
        target_label=target_label,
        key_frame_grid=_save(grid, out / "key_frame_grid.png"),
        pair_strip=_save(pair, out / "pair_strip.png"),
        mask_overlay=mask_path,
        motion_evidence=motion_path,
        reference_image=reference_image,
        onset_window=onset,
    )


def _participant_index(participant: object) -> int | None:
    """Detection index of a participant id such as ``2`` or ``"veh_2"``."""
    if isinstance(participant, int):
        return participant
    digits = re.search(r"(\d+)$", str(participant))
    return int(digits.group(1)) if digits else None


def _participant_geometry(
    reference: ReferenceResult,
    accident: AccidentSpec,
    backends: Backends,
    frame_shape: tuple[int, int],
) -> tuple[list[Box], Box | None]:
    """Resolve participant boxes and the impact box for the edit region.

    Participant ids index into the vehicle detections of the first key frame
    when a detector is available. A normalized impact region is scaled to
    pixel coordinates of the key frame.
    """
    boxes: list[Box] = []
    if backends.vehicle_detector is not None and accident.participants:
        frame = cv2.cvtColor(cv2.imread(reference.key_frames[0]), cv2.COLOR_BGR2RGB)
        detections = list(backends.vehicle_detector(frame))
        for participant in accident.participants:
            index = _participant_index(participant)
            if index is not None and 0 <= index < len(detections):
                boxes.append(tuple(int(v) for v in detections[index]))
    impact = accident.impact_region
    if not _is_box(impact):
        return boxes, None
    height, width = frame_shape
    if all(0.0 <= v <= 1.0 for v in impact):
        impact = (impact[0] * width, impact[1] * height, impact[2] * width, impact[3] * height)
    return boxes, tuple(int(round(v)) for v in impact)


def _is_box(value) -> bool:
    return (
        isinstance(value, (tuple, list))
        and len(value) == 4
        and all(isinstance(v, (int, float)) for v in value)
    )


def _preservation_mask_path(edit_mask: np.ndarray | None, shape: tuple[int, int], out: Path) -> str:
    keep = np.ones(shape, dtype=np.uint8) if edit_mask is None else (~edit_mask).astype(np.uint8)
    path = out / "preservation_mask.png"
    cv2.imwrite(str(path), keep * 255)
    return str(path)


def _log(rejections: list | None, quartet_id: str, branch: str | None, result: GateResult) -> None:
    if rejections is not None:
        rejections.append(
            {"quartet_id": quartet_id, "stage": result.stage, "branch": branch,
             "details": result.details}
        )


def generate_quartet(
    source: SourceUnit,
    env_spec: EnvironmentSpec,
    accident_spec: AccidentSpec,
    backends: Backends,
    gates: GateSuite,
    out_dir: str | Path = "generated",
    quartet_id: str | None = None,
    held_out: bool = False,
    script_note: str | None = None,
    fps: int = 16,
    key_frame_count: int = 4,
    rejection_log: list | None = None,
) -> QuartetRecord | None:
    """Generate and fully gate one candidate quartet.

    Runs phase one, then generates and gates the three edited branches in
    order. Processing stops at the first failed gate.

    Args:
        source: Source clip and its metadata.
        env_spec: Target environment for the environment and joint branches.
        accident_spec: Accident script for the accident and joint branches.
        backends: Generation backends and optional perception callables.
        gates: Configured quality-control layers.
        out_dir: Root directory for generated media and evidence.
        quartet_id: Identifier; derived from the source and scripts when
            omitted.
        held_out: Whether the edit tuple belongs to the held-out composition
            set.
        script_note: Feasibility note required by the factor combination.
        fps: Decode rate used for timestamps and onset anchors.
        key_frame_count: Key frames extracted for phase one.
        rejection_log: Optional list receiving one entry per failed gate with
            the failing stage and branch.

    Returns:
        The accepted quartet record, or ``None`` when any gate fails.
    """
    quartet_id = quartet_id or (
        f"{source.source_id}-{accident_spec.accident_type}-{accident_spec.onset}"
        f"-{env_spec.illumination}-{env_spec.weather}"
    )
    work = Path(out_dir) / quartet_id
    work.mkdir(parents=True, exist_ok=True)

    reference = generate_reference(
        source.path, env_spec, backends.image, gates.reference_judge, work / "phase1",
        key_frame_count=key_frame_count, script_note=script_note,
        lane_segmenter=backends.lane_segmenter, vehicle_detector=backends.vehicle_detector,
    )
    if not reference.accepted:
        _log(rejection_log, quartet_id, None, reference.gate)
        return None

    frame_shape = cv2.imread(reference.key_frames[0]).shape[:2]
    participant_boxes, impact_box = _participant_geometry(
        reference, accident_spec, backends, frame_shape
    )
    onset = onset_window(accident_spec.onset)
    paths: dict[str, str] = {Branch.V0.value: source.path,
                             "reference": reference.image_path}

    for branch in _GENERATED_BRANCHES:
        branch_dir = work / branch.value.lower()
        branch_dir.mkdir(parents=True, exist_ok=True)
        positive = branch in (Branch.VA, Branch.VAE)
        environment = source.environment if branch is Branch.VA else env_spec
        edit_mask = (
            accident_region_mask(frame_shape, participant_boxes, impact_box)
            if positive and (participant_boxes or impact_box)
            else None
        )
        sketch = None
        if positive:
            frame = cv2.cvtColor(cv2.imread(reference.key_frames[0]), cv2.COLOR_BGR2RGB)
            sketch = render_trajectory_sketch(
                frame, participant_boxes, impact_box, branch_dir / "trajectory_sketch.png"
            )
        request = VideoEditRequest(
            source_path=source.path,
            instruction=branch_instruction(
                branch, env_spec, accident_spec if positive else None, note=script_note
            ),
            out_path=str(branch_dir / f"{branch.value}.mp4"),
            reference_image=None if branch is Branch.VA else reference.image_path,
            preservation_mask=_preservation_mask_path(edit_mask, frame_shape, branch_dir),
            lane_mask=reference.cue_paths.get("lane_mask"),
            trajectory_sketch=sketch,
        )
        edited_path = backends.video.edit(request)
        candidate = EditCandidate(
            source_path=source.path, edited_path=edited_path, branch=branch,
            edit_mask=edit_mask, environment=environment,
            accident=accident_spec if positive else None,
        )
        objective = run_objective_gates(
            candidate, gates.objective, embedder=gates.embedder,
            probe=gates.probe, gamma_f=gates.gamma_f,
        )
        if not objective.passed:
            _log(rejection_log, quartet_id, branch.value, objective)
            return None
        script_text = (
            accident_script(accident_spec) if positive
            else environment_script(env_spec, script_note)
        )
        packet = build_evidence_packet(
            candidate, script_text, int(positive), branch_dir / "evidence", fps=fps,
            reference_image=None if branch is Branch.VA else reference.image_path,
            onset=onset if positive else None,
            participant_boxes=participant_boxes, impact_box=impact_box,
            resolution=gates.objective.analysis_resolution,
        )
        panel = run_panel_gate(packet, gates.panel_judges, gates.rubric)
        if not panel.passed:
            _log(rejection_log, quartet_id, branch.value, panel)
            return None
        paths[branch.value] = edited_path

    return QuartetRecord(
        quartet_id=quartet_id,
        source_id=source.source_id,
        dataset=source.dataset,
        split=source.split,
        env=env_spec,
        accident=accident_spec,
        held_out_composition=held_out,
        paths=paths,
        scores={},
    )
