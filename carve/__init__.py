"""CARVE: counterfactual audit and hardening for CCTV accident detectors.

CARVE (Counterfactual Accident Robustness via Video Editing) builds matched
video quartets (V0, VE, VA, VAE) from real roadside CCTV clips: the source
clip, a nuisance-only environment edit, an accident insertion under the
source environment, and the joint edit. On this structure it provides

* an audit suite (CPS, NESI, ESI, CFI, CGS, plus PMR and FSS) that
  decomposes detector robustness by failure mode and factor level;
* CGAA-IC, a closed hardening loop that samples weak factors (PMR-FSS),
  generates and gates fresh quartets, and trains with environment-purity,
  intervention-consistency, and faithfulness-margin losses while leaving the
  inference graph unchanged.

The audit stack (``carve.core``, ``carve.factors``, ``carve.metrics``, the
sampler, clip protocol, and splits) needs only numpy; torch-backed modules
(detector, losses, datasets, hardening loop) are exposed lazily.
"""

from __future__ import annotations

import importlib
from typing import Any

from .core import AccidentSpec, Branch, EnvironmentSpec, GateResult, QuartetRecord
from .factors import (
    ACCIDENT_LEVELS,
    ALL_LEVELS,
    ENVIRONMENT_LEVELS,
    EditContext,
    FactorLevel,
    check_feasibility,
    composition_key,
    feasible_compositions,
    held_out_keys,
    sample_held_out_compositions,
)
from .metrics import (
    AuditSummary,
    CGSResult,
    LevelDiagnostics,
    audit_quartets,
    factor_stress_scores,
    per_level_diagnostics,
    select_threshold,
)
from .training.sampler import RoundBudget, WeakFactorSampler, sampling_probabilities

__version__ = "1.0.0"

_LAZY = {
    "AccidentDetector": "carve.models.detector",
    "VideoMAEDetector": "carve.models.detector",
    "configure_training": "carve.models.detector",
    "ClipDataset": "carve.data.quartets",
    "LabeledClip": "carve.data.quartets",
    "QuartetDataset": "carve.data.quartets",
    "load_labeled_clips": "carve.data.quartets",
    "load_manifest": "carve.data.quartets",
    "save_manifest": "carve.data.quartets",
    "LossWeights": "carve.training.losses",
    "hardening_objective": "carve.training.losses",
    "HardeningConfig": "carve.training.cgaa_ic",
    "HardeningResult": "carve.training.cgaa_ic",
    "QualityGate": "carve.training.cgaa_ic",
    "QuartetGenerator": "carve.training.cgaa_ic",
    "ReferenceGenerator": "carve.training.cgaa_ic",
    "SourceClip": "carve.training.cgaa_ic",
    "run_hardening": "carve.training.cgaa_ic",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ACCIDENT_LEVELS",
    "ALL_LEVELS",
    "AccidentDetector",
    "AccidentSpec",
    "AuditSummary",
    "Branch",
    "CGSResult",
    "ClipDataset",
    "EditContext",
    "ENVIRONMENT_LEVELS",
    "EnvironmentSpec",
    "FactorLevel",
    "GateResult",
    "HardeningConfig",
    "HardeningResult",
    "LabeledClip",
    "LevelDiagnostics",
    "LossWeights",
    "QualityGate",
    "QuartetDataset",
    "QuartetGenerator",
    "QuartetRecord",
    "ReferenceGenerator",
    "RoundBudget",
    "SourceClip",
    "VideoMAEDetector",
    "WeakFactorSampler",
    "audit_quartets",
    "check_feasibility",
    "composition_key",
    "configure_training",
    "factor_stress_scores",
    "feasible_compositions",
    "hardening_objective",
    "held_out_keys",
    "load_labeled_clips",
    "load_manifest",
    "per_level_diagnostics",
    "run_hardening",
    "sample_held_out_compositions",
    "sampling_probabilities",
    "save_manifest",
    "select_threshold",
    "__version__",
]
