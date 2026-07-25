"""Phase one: environment reference-image generation and gating.

For each quartet with an environment edit, the image backend turns source key
frames, layout cues, and the environment script into one reference image that
fixes the requested nuisance appearance under the source layout. The image is
checked by the reference gate before any video generation; one accepted
reference is shared by the environment and joint branches.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np

from carve.core import EnvironmentSpec, GateResult
from carve.gates.objective import load_frames
from carve.gates.reference import run_reference_gate
from carve.generation.layout_cues import Box, extract_layout_cues, save_layout_cues
from carve.generation.providers import ImageEditBackend, VlmJudge
from carve.generation.scripts_builder import reference_image_instruction


@dataclasses.dataclass
class ReferenceResult:
    """Outcome of phase one for a single quartet.

    Attributes:
        image_path: Path of the generated reference image.
        gate: Reference-gate outcome; only a passing reference may condition
            video generation.
        key_frames: Paths of the extracted source key frames.
        cue_paths: Paths of saved layout-cue images.
    """

    image_path: str
    gate: GateResult
    key_frames: list[str]
    cue_paths: dict[str, str]

    @property
    def accepted(self) -> bool:
        return self.gate.passed


def extract_key_frames(
    clip_path: str | Path,
    out_dir: str | Path,
    count: int = 4,
    resolution: int = 512,
) -> tuple[list[str], list[np.ndarray]]:
    """Save uniformly spaced key frames of a clip and return paths and arrays.

    Key frames are written at a higher resolution than the 224-pixel analysis
    grid because they condition image generation rather than measurement.
    """
    frames = load_frames(clip_path, resolution)
    indices = np.linspace(0, len(frames) - 1, num=min(count, len(frames))).astype(int)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    arrays: list[np.ndarray] = []
    for rank, index in enumerate(np.unique(indices)):
        path = out / f"key_{rank:02d}_f{index:03d}.jpg"
        cv2.imwrite(str(path), cv2.cvtColor(frames[index], cv2.COLOR_RGB2BGR))
        paths.append(str(path))
        arrays.append(frames[index])
    return paths, arrays


def generate_reference(
    source_clip: str | Path,
    environment: EnvironmentSpec,
    backend: ImageEditBackend,
    judge: VlmJudge,
    out_dir: str | Path,
    key_frame_count: int = 4,
    script_note: str | None = None,
    lane_segmenter: Callable[[np.ndarray], np.ndarray] | None = None,
    vehicle_detector: Callable[[np.ndarray], Sequence[Box]] | None = None,
) -> ReferenceResult:
    """Run phase one for one quartet.

    Args:
        source_clip: Real source clip providing key frames and layout.
        environment: Target environment for the reference image.
        backend: Image-editing backend.
        judge: VLM used by the reference gate.
        out_dir: Directory for key frames, cue images, and the reference.
        key_frame_count: Number of key frames to extract.
        script_note: Feasibility note propagated into the instruction, when
            the factor combination requires a stated cause.
        lane_segmenter: Optional lane-mask callable.
        vehicle_detector: Optional vehicle-box callable.

    Returns:
        The reference image together with its gate outcome. Callers must
        discard results whose gate failed before generating any video.
    """
    out = Path(out_dir)
    key_paths, key_arrays = extract_key_frames(source_clip, out, key_frame_count)
    cues = extract_layout_cues(
        key_arrays[0], lane_segmenter=lane_segmenter, vehicle_detector=vehicle_detector
    )
    cue_paths = save_layout_cues(cues, out, "layout")
    instruction = reference_image_instruction(environment, script_note)
    image_path = backend.generate(
        key_frames=key_paths,
        layout_overlays=list(cue_paths.values()),
        instruction=instruction,
        out_path=str(out / "reference.png"),
    )
    gate = run_reference_gate(image_path, key_paths, environment, judge)
    return ReferenceResult(
        image_path=image_path, gate=gate, key_frames=key_paths, cue_paths=cue_paths
    )
