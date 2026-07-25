"""Detector interfaces and backbones (torch-dependent)."""

from .detector import (
    AccidentDetector,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHECKPOINT,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_NUM_FRAMES,
    DEFAULT_WEIGHT_DECAY,
    VideoMAEDetector,
    configure_training,
)

__all__ = [
    "AccidentDetector",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_CHECKPOINT",
    "DEFAULT_EPOCHS",
    "DEFAULT_LEARNING_RATE",
    "DEFAULT_NUM_FRAMES",
    "DEFAULT_WEIGHT_DECAY",
    "VideoMAEDetector",
    "configure_training",
]
