"""Unified 64-frame clip protocol shared by all datasets.

Videos are decoded at 16 fps and represented by 64-frame clips analyzed at
224 x 224. Positive clips place the first visible impact frame inside the
24-40 onset band: at 16 fps this guarantees at least 1.5 s of pre-impact
context and at least 1.5 s of post-impact evidence within the clip, whereas
an onset outside the band would truncate one side at the clip boundary and
leave the label ambiguous. The onset levels partition the band, not the
whole clip: early = 24-28, middle = 29-34, late = 35-40.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

FRAME_RATE = 16
CLIP_FRAMES = 64
ANALYSIS_RESOLUTION = (224, 224)

#: Inclusive band of admissible clip-relative onset frames for positives.
ONSET_RANGE = (24, 40)
#: Inclusive clip-relative frame windows for the three onset levels.
ONSET_WINDOWS: dict[str, tuple[int, int]] = {
    "early": (24, 28),
    "middle": (29, 34),
    "late": (35, 40),
}
#: Minimum context preserved on each side of the onset band, in seconds.
MIN_CONTEXT_SECONDS = 1.5

#: Negative windows are sampled from the middle fraction of a video to avoid
#: title cards and fade artifacts.
NEGATIVE_CENTER_FRACTION = 0.8


@dataclass(frozen=True)
class ClipWindow:
    """A 64-frame window in source-video frame coordinates.

    ``start`` is inclusive, ``end`` exclusive, and ``onset`` (absolute frame
    of first visible impact) is ``None`` for negative windows.
    """

    start: int
    end: int
    label: int
    onset: Optional[int] = None

    @property
    def onset_in_clip(self) -> Optional[int]:
        return None if self.onset is None else self.onset - self.start


def onset_window(level: str) -> tuple[int, int]:
    """Clip-relative frame window of an onset level."""
    try:
        return ONSET_WINDOWS[level]
    except KeyError as exc:
        raise ValueError(f"unknown onset level {level!r}") from exc


def onset_level(frame_in_clip: int) -> str:
    """Onset level of a clip-relative impact frame inside the 24-40 band."""
    for level, (lo, hi) in ONSET_WINDOWS.items():
        if lo <= frame_in_clip <= hi:
            return level
    raise ValueError(
        f"onset frame {frame_in_clip} lies outside the admissible band {ONSET_RANGE}"
    )


def positive_window(
    impact_frame: int,
    total_frames: int,
    rng: Optional[np.random.Generator] = None,
    onset_range: Optional[tuple[int, int]] = None,
) -> Optional[ClipWindow]:
    """Positive clip centered on the first visible impact frame.

    The window start places the onset at the center of the admissible band
    (frame 32 under the default 24-40 band), jittered uniformly over the
    band when ``rng`` is given, then clamped to the video and to the band.
    ``onset_range`` overrides the band for protocol variants; when omitted
    the module default applies. Returns ``None`` when no valid 64-frame
    window exists.
    """
    band = onset_range if onset_range is not None else ONSET_RANGE
    if not 0 <= band[0] <= band[1] < CLIP_FRAMES:
        raise ValueError(
            f"onset band {band} must satisfy 0 <= low <= high < {CLIP_FRAMES}"
        )
    lo = max(impact_frame - band[1], 0)
    hi = min(impact_frame - band[0], total_frames - CLIP_FRAMES)
    if hi < lo:
        return None
    if rng is not None:
        target_onset = int(rng.integers(band[0], band[1] + 1))
    else:
        target_onset = (band[0] + band[1]) // 2
    start = int(np.clip(impact_frame - target_onset, lo, hi))
    return ClipWindow(start=start, end=start + CLIP_FRAMES, label=1, onset=impact_frame)


def negative_window(
    total_frames: int, rng: np.random.Generator
) -> Optional[ClipWindow]:
    """Negative clip sampled from the middle 80% of an accident-free video."""
    margin = int(round(total_frames * (1.0 - NEGATIVE_CENTER_FRACTION) / 2.0))
    lo = margin
    hi = total_frames - margin - CLIP_FRAMES
    if hi < lo:
        return None
    start = int(rng.integers(lo, hi + 1))
    return ClipWindow(start=start, end=start + CLIP_FRAMES, label=0)


def cadp_paired_windows(
    onset_frame: int,
    total_frames: int,
    rng: Optional[np.random.Generator] = None,
    onset_range: Optional[tuple[int, int]] = None,
) -> Optional[tuple[ClipWindow, ClipWindow]]:
    """Paired positive and negative windows for one CADP event.

    The positive clip is centered on the annotated impact; the negative clip
    is the verified clean 4-second window ending before the annotated onset.
    Both clips derive from the same event and must be assigned to the same
    split (the split unit is the event). ``onset_range`` is forwarded to the
    positive-window placement. Returns ``None`` when the event cannot
    provide both windows.
    """
    positive = positive_window(onset_frame, total_frames, rng, onset_range)
    if positive is None or onset_frame < CLIP_FRAMES:
        return None
    negative = ClipWindow(
        start=onset_frame - CLIP_FRAMES, end=onset_frame, label=0
    )
    return positive, negative


def frames_to_seconds(frames: int) -> float:
    """Duration of a frame count under the 16 fps decode rate."""
    return frames / FRAME_RATE


__all__ = [
    "ANALYSIS_RESOLUTION",
    "CLIP_FRAMES",
    "ClipWindow",
    "FRAME_RATE",
    "MIN_CONTEXT_SECONDS",
    "NEGATIVE_CENTER_FRACTION",
    "ONSET_RANGE",
    "ONSET_WINDOWS",
    "cadp_paired_windows",
    "frames_to_seconds",
    "negative_window",
    "onset_level",
    "onset_window",
    "positive_window",
]
