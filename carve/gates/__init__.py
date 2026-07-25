"""Three-layer quality control for generated counterfactual edits.

Layer one checks the environment reference image before any video is
generated; layer two applies objective structure and motion gates outside the
edited region; layer three queries a three-judge validity panel with a fixed
rubric. A quartet is retained only when all of its generated branches pass
every applicable layer.
"""

from carve.gates.objective import (
    DinoEmbedder,
    EditCandidate,
    ObjectiveGateConfig,
    calibrate_gamma_f,
    run_objective_gates,
)
from carve.gates.panel import (
    EvidencePacket,
    PanelRubric,
    panel_semantic_score,
    run_panel_gate,
)
from carve.gates.probe import (
    CaptureQualityProbe,
    ProbeConfig,
    load_probe,
    predict,
    save_probe,
    train_probe,
)
from carve.gates.reference import run_reference_gate

__all__ = [
    "CaptureQualityProbe",
    "DinoEmbedder",
    "EditCandidate",
    "EvidencePacket",
    "ObjectiveGateConfig",
    "PanelRubric",
    "ProbeConfig",
    "calibrate_gamma_f",
    "load_probe",
    "panel_semantic_score",
    "predict",
    "run_objective_gates",
    "run_panel_gate",
    "run_reference_gate",
    "save_probe",
    "train_probe",
]
