"""Shared data contracts for the CARVE audit-and-hardening framework.

Every stage of the pipeline exchanges the same small set of types: the four
quartet branches, the environment and accident factor assignments of an edit,
the per-quartet bookkeeping record, and the outcome of a quality-gate stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Branch(str, Enum):
    """The four branches of a matched counterfactual quartet.

    ``V0`` is the real source clip without an accident, ``VE`` edits only
    environment nuisance factors, ``VA`` inserts accident evidence under the
    source environment, and ``VAE`` combines the accident with the edited
    environment. Branch values double as keys into ``QuartetRecord.paths``
    and ``QuartetRecord.scores``.
    """

    V0 = "V0"
    VE = "VE"
    VA = "VA"
    VAE = "VAE"


@dataclass(frozen=True)
class EnvironmentSpec:
    """Environment-side factor assignment of an edit (11 levels over 4 factors)."""

    illumination: str        # "day" | "dusk" | "night"
    weather: str             # "clear" | "rain" | "fog"
    road_surface: str        # "dry" | "wet"
    capture_quality: str     # "clean" | "compressed" | "motion_blur"


@dataclass(frozen=True)
class AccidentSpec:
    """Accident-side factor assignment of an edit (12 levels over 3 factors).

    ``participants`` and ``impact_region`` localize the scripted collision on a
    concrete source clip; they are not part of the factor-composition identity.
    """

    accident_type: str       # "rear_end" | "side_impact" | "t_bone" | "pedestrian_strike" | "motorcycle_collision" | "rollover"
    severity: str            # "minor" | "moderate" | "severe"
    onset: str               # "early" | "middle" | "late"
    participants: tuple[str, ...] = ()
    impact_region: Optional[tuple[float, float, float, float]] = None  # normalized (x1, y1, x2, y2)


@dataclass
class QuartetRecord:
    """Bookkeeping for one accepted quartet derived from a single source clip.

    ``paths`` maps branch names ("V0", "VE", "VA", "VAE") to media locations;
    generation stages may add auxiliary keys such as "reference". ``scores``
    caches detector probabilities under the same branch keys and is what the
    audit metrics read. ``held_out_composition`` marks quartets whose factor
    composition belongs to the held-out set used for CGS.
    """

    quartet_id: str
    source_id: str
    dataset: str                       # "tad" | "cadp" | "surveillancecrash"
    split: str                         # "train" | "val" | "test"
    env: EnvironmentSpec
    accident: AccidentSpec
    held_out_composition: bool = False
    paths: dict[str, str] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)


@dataclass
class GateResult:
    """Outcome of one quality-gate stage for a candidate reference or clip."""

    passed: bool
    stage: str                         # "reference" | "objective" | "panel"
    details: dict = field(default_factory=dict)


__all__ = [
    "AccidentSpec",
    "Branch",
    "EnvironmentSpec",
    "GateResult",
    "QuartetRecord",
]
