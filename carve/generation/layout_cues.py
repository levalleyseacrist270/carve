"""Layout cues extracted from source key frames.

Cues constrain generation to the source scene geometry: lane masks, a road-
boundary overlay, boxes around vehicles outside the scripted collision, and a
coarse perspective estimate. Heavy models stay optional; lane segmentation
and vehicle detection are injected as callables, while the boundary overlay
and the vanishing-point estimate have lightweight edge-based fallbacks.
"""

from __future__ import annotations

import dataclasses
import itertools
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np

Box = tuple[int, int, int, int]  # x1, y1, x2, y2


@dataclasses.dataclass
class LayoutCues:
    """Layout constraints for one source scene.

    Attributes:
        lane_mask: Binary lane mask, when a segmenter is available.
        boundary_overlay: Key frame with detected road boundaries drawn in.
        nonparticipant_boxes: Boxes around vehicles outside the scripted
            collision.
        vanishing_point: Coarse ``(x, y)`` perspective estimate, when line
            structure supports one.
    """

    lane_mask: np.ndarray | None
    boundary_overlay: np.ndarray
    nonparticipant_boxes: list[Box]
    vanishing_point: tuple[float, float] | None


def _detect_lines(gray: np.ndarray) -> np.ndarray:
    edges = cv2.Canny(gray, 60, 160)
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180, threshold=60, minLineLength=gray.shape[1] // 6,
        maxLineGap=12,
    )
    return lines[:, 0] if lines is not None else np.empty((0, 4), dtype=int)


def road_boundary_overlay(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Draw edge-based road boundary lines onto a copy of the frame.

    Returns:
        The overlay image and the detected line segments as ``(N, 4)`` arrays
        of ``x1, y1, x2, y2``.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    lines = _detect_lines(gray)
    overlay = frame.copy()
    for x1, y1, x2, y2 in lines:
        cv2.line(overlay, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    return overlay, lines


def estimate_vanishing_point(lines: np.ndarray, min_angle_deg: float = 10.0) -> tuple[float, float] | None:
    """Coarse vanishing point from clustered pairwise line intersections.

    Near-parallel pairs are skipped; the median of the remaining
    intersections is robust to a moderate fraction of off-road lines.
    """
    candidates: list[tuple[float, float]] = []
    for (x1, y1, x2, y2), (x3, y3, x4, y4) in itertools.combinations(map(tuple, lines), 2):
        d1, d2 = (x2 - x1, y2 - y1), (x4 - x3, y4 - y3)
        angle = abs(
            np.degrees(np.arctan2(d1[1], d1[0]) - np.arctan2(d2[1], d2[0]))
        ) % 180.0
        if min(angle, 180.0 - angle) < min_angle_deg:
            continue
        denominator = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(denominator) < 1e-6:
            continue
        t = ((x3 - x1) * d2[1] - (y3 - y1) * d2[0]) / denominator
        candidates.append((x1 + t * d1[0], y1 + t * d1[1]))
    if len(candidates) < 3:
        return None
    points = np.asarray(candidates)
    return float(np.median(points[:, 0])), float(np.median(points[:, 1]))


def _overlap_ratio(box: Box, other: Box) -> float:
    x1 = max(box[0], other[0])
    y1 = max(box[1], other[1])
    x2 = min(box[2], other[2])
    y2 = min(box[3], other[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area = max(1, (box[2] - box[0]) * (box[3] - box[1]))
    return inter / area


def extract_layout_cues(
    key_frame: np.ndarray,
    lane_segmenter: Callable[[np.ndarray], np.ndarray] | None = None,
    vehicle_detector: Callable[[np.ndarray], Sequence[Box]] | None = None,
    participant_boxes: Sequence[Box] = (),
    overlap_max: float = 0.5,
) -> LayoutCues:
    """Extract all layout cues from one key frame.

    Args:
        key_frame: RGB key frame of the source clip.
        lane_segmenter: Optional callable returning a binary lane mask.
        vehicle_detector: Optional callable returning vehicle boxes.
        participant_boxes: Boxes of scripted participants; detected vehicles
            overlapping them beyond ``overlap_max`` are excluded from the
            non-participant set.
        overlap_max: Overlap ratio above which a detection counts as a
            participant.
    """
    overlay, lines = road_boundary_overlay(key_frame)
    boxes: list[Box] = []
    if vehicle_detector is not None:
        for box in vehicle_detector(key_frame):
            if all(_overlap_ratio(box, p) <= overlap_max for p in participant_boxes):
                boxes.append(tuple(int(v) for v in box))
    return LayoutCues(
        lane_mask=None if lane_segmenter is None else lane_segmenter(key_frame),
        boundary_overlay=overlay,
        nonparticipant_boxes=boxes,
        vanishing_point=estimate_vanishing_point(lines),
    )


def accident_region_mask(
    frame_shape: tuple[int, int],
    participant_boxes: Sequence[Box],
    impact_box: Box | None = None,
    dilation_px: int = 12,
) -> np.ndarray:
    """Static edit-region mask covering participants and the impact region.

    The union of boxes is dilated so boundary pixels of the scripted motion
    are attributed to the edit rather than to the preservation region.
    """
    mask = np.zeros(frame_shape, dtype=np.uint8)
    boxes = [*participant_boxes] + ([impact_box] if impact_box else [])
    for box in boxes:
        x1, y1, x2, y2 = (int(v) for v in box)
        mask[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)] = 1
    if dilation_px > 0:
        kernel = np.ones((dilation_px, dilation_px), np.uint8)
        mask = cv2.dilate(mask, kernel)
    return mask.astype(bool)


def render_trajectory_sketch(
    frame: np.ndarray,
    participant_boxes: Sequence[Box],
    impact_box: Box | None,
    out_path: str | Path,
) -> str:
    """Draw a coarse participant-trajectory sketch and save it.

    Arrows run from each participant box center toward the impact-region
    center, giving the editor a spatial script without prescribing exact
    kinematics.
    """
    sketch = frame.copy()
    target = None
    if impact_box is not None:
        x1, y1, x2, y2 = impact_box
        target = ((x1 + x2) // 2, (y1 + y2) // 2)
        cv2.rectangle(sketch, (x1, y1), (x2, y2), (255, 0, 0), 2)
    for box in participant_boxes:
        x1, y1, x2, y2 = box
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        cv2.rectangle(sketch, (x1, y1), (x2, y2), (255, 255, 0), 2)
        if target is not None and target != center:
            cv2.arrowedLine(sketch, center, target, (255, 0, 0), 2, tipLength=0.2)
    cv2.imwrite(str(out_path), cv2.cvtColor(sketch, cv2.COLOR_RGB2BGR))
    return str(out_path)


def save_layout_cues(cues: LayoutCues, out_dir: str | Path, stem: str) -> dict[str, str]:
    """Persist cue images for use as backend conditioning inputs."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    boundary = out / f"{stem}_boundaries.png"
    cv2.imwrite(str(boundary), cv2.cvtColor(cues.boundary_overlay, cv2.COLOR_RGB2BGR))
    paths["boundary_overlay"] = str(boundary)
    if cues.lane_mask is not None:
        lane = out / f"{stem}_lanes.png"
        cv2.imwrite(str(lane), cues.lane_mask.astype(np.uint8) * 255)
        paths["lane_mask"] = str(lane)
    return paths
