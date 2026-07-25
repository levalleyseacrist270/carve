"""Data protocol: clip windows, source-disjoint splits, and datasets.

Torch-backed datasets and manifest I/O live in :mod:`carve.data.quartets`
and are exposed lazily so the clip protocol and split logic stay importable
with numpy alone.
"""

from __future__ import annotations

import importlib
from typing import Any

from .clips import (
    ANALYSIS_RESOLUTION,
    CLIP_FRAMES,
    ClipWindow,
    FRAME_RATE,
    NEGATIVE_CENTER_FRACTION,
    ONSET_RANGE,
    ONSET_WINDOWS,
    cadp_paired_windows,
    negative_window,
    onset_level,
    onset_window,
    positive_window,
)
from .splits import assign_splits, group_by_hash, hamming_distance

_LAZY = {
    "ClipDataset": "carve.data.quartets",
    "ClipLoader": "carve.data.quartets",
    "LabeledClip": "carve.data.quartets",
    "QuartetDataset": "carve.data.quartets",
    "load_labeled_clips": "carve.data.quartets",
    "load_manifest": "carve.data.quartets",
    "record_from_dict": "carve.data.quartets",
    "record_to_dict": "carve.data.quartets",
    "save_manifest": "carve.data.quartets",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ANALYSIS_RESOLUTION",
    "CLIP_FRAMES",
    "ClipDataset",
    "ClipLoader",
    "ClipWindow",
    "FRAME_RATE",
    "LabeledClip",
    "NEGATIVE_CENTER_FRACTION",
    "ONSET_RANGE",
    "ONSET_WINDOWS",
    "QuartetDataset",
    "assign_splits",
    "cadp_paired_windows",
    "group_by_hash",
    "hamming_distance",
    "load_labeled_clips",
    "load_manifest",
    "negative_window",
    "onset_level",
    "onset_window",
    "positive_window",
    "record_from_dict",
    "record_to_dict",
    "save_manifest",
]
