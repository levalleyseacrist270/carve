"""Hardening: PMR-FSS sampling, quartet losses, and the CGAA-IC loop.

The sampler and budget arithmetic are numpy-only and imported eagerly; the
torch-dependent losses and loop are exposed lazily.
"""

from __future__ import annotations

import importlib
from typing import Any

from .sampler import (
    DEFAULT_ALPHA,
    DEFAULT_BUDGET_FRACTION,
    DEFAULT_OVERSAMPLE,
    EDITS_PER_QUARTET,
    RoundBudget,
    WeakFactorSampler,
    sampling_probabilities,
)

_LAZY = {
    "LossBreakdown": "carve.training.losses",
    "LossWeights": "carve.training.losses",
    "binary_supervision_loss": "carve.training.losses",
    "environment_purity_loss": "carve.training.losses",
    "faithfulness_margin_loss": "carve.training.losses",
    "hardening_objective": "carve.training.losses",
    "intervention_consistency_loss": "carve.training.losses",
    "HardeningConfig": "carve.training.cgaa_ic",
    "HardeningResult": "carve.training.cgaa_ic",
    "QualityGate": "carve.training.cgaa_ic",
    "QuartetGenerator": "carve.training.cgaa_ic",
    "ReferenceGenerator": "carve.training.cgaa_ic",
    "RoundReport": "carve.training.cgaa_ic",
    "SourceClip": "carve.training.cgaa_ic",
    "calibrate_operating_point": "carve.training.cgaa_ic",
    "generate_round": "carve.training.cgaa_ic",
    "run_hardening": "carve.training.cgaa_ic",
    "score_clips": "carve.training.cgaa_ic",
    "score_quartet_records": "carve.training.cgaa_ic",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_BUDGET_FRACTION",
    "DEFAULT_OVERSAMPLE",
    "EDITS_PER_QUARTET",
    "HardeningConfig",
    "HardeningResult",
    "LossBreakdown",
    "LossWeights",
    "QualityGate",
    "QuartetGenerator",
    "ReferenceGenerator",
    "RoundBudget",
    "RoundReport",
    "SourceClip",
    "WeakFactorSampler",
    "binary_supervision_loss",
    "calibrate_operating_point",
    "environment_purity_loss",
    "faithfulness_margin_loss",
    "generate_round",
    "hardening_objective",
    "intervention_consistency_loss",
    "run_hardening",
    "sampling_probabilities",
    "score_clips",
    "score_quartet_records",
]
