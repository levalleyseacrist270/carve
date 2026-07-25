"""Editable factor space, feasibility rules, and held-out composition sampling.

The factor space has 23 levels: 11 environment levels (illumination 3,
weather 3, road surface 2, capture quality 3) and 12 accident levels
(accident type 6, severity 3, onset 3). A composition is one full assignment
of all seven factors; the feasibility mask ``check_feasibility`` removes
physically implausible or visually underspecified compositions before any
edit is scripted.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable, Collection, Iterator, Optional

import numpy as np

from .core import AccidentSpec, EnvironmentSpec

ILLUMINATION_LEVELS = ("day", "dusk", "night")
WEATHER_LEVELS = ("clear", "rain", "fog")
ROAD_SURFACE_LEVELS = ("dry", "wet")
CAPTURE_QUALITY_LEVELS = ("clean", "compressed", "motion_blur")
ACCIDENT_TYPE_LEVELS = (
    "rear_end",
    "side_impact",
    "t_bone",
    "pedestrian_strike",
    "motorcycle_collision",
    "rollover",
)
SEVERITY_LEVELS = ("minor", "moderate", "severe")
ONSET_LEVELS = ("early", "middle", "late")

ENVIRONMENT_FACTORS: dict[str, tuple[str, ...]] = {
    "illumination": ILLUMINATION_LEVELS,
    "weather": WEATHER_LEVELS,
    "road_surface": ROAD_SURFACE_LEVELS,
    "capture_quality": CAPTURE_QUALITY_LEVELS,
}
ACCIDENT_FACTORS: dict[str, tuple[str, ...]] = {
    "accident_type": ACCIDENT_TYPE_LEVELS,
    "severity": SEVERITY_LEVELS,
    "onset": ONSET_LEVELS,
}

#: Number of held-out feasible compositions reserved for CGS.
HELD_OUT_COUNT = 18
#: Candidate edit tuples per source clip during benchmark construction.
PER_SOURCE_CANDIDATE_CAP = 6
#: Held-out tuples drawn per test source, replacing regular candidates.
HELD_OUT_PER_TEST_SOURCE = (1, 2)

#: Minimum normalized participant height for a plausible rollover script.
MIN_ROLLOVER_PARTICIPANT_SCALE = 0.05
#: Minimum visible-road area fraction for a plausible rollover script.
MIN_ROLLOVER_ROAD_FRACTION = 0.25

#: Script vocabulary for plausible wet-pavement causes under clear weather.
WET_PAVEMENT_CAUSES = ("recent_rain", "standing_water", "road_cleaning")


@dataclass(frozen=True)
class FactorLevel:
    """One level of one editable factor, e.g. illumination=night."""

    group: str   # "environment" | "accident"
    factor: str
    level: str

    @property
    def key(self) -> str:
        """Canonical identifier used by diagnostics and the sampler."""
        return f"{self.factor}={self.level}"


def _build_levels(group: str, factors: dict[str, tuple[str, ...]]) -> tuple[FactorLevel, ...]:
    return tuple(
        FactorLevel(group, factor, level)
        for factor, levels in factors.items()
        for level in levels
    )


ENVIRONMENT_LEVELS: tuple[FactorLevel, ...] = _build_levels("environment", ENVIRONMENT_FACTORS)
ACCIDENT_LEVELS: tuple[FactorLevel, ...] = _build_levels("accident", ACCIDENT_FACTORS)
ALL_LEVELS: tuple[FactorLevel, ...] = ENVIRONMENT_LEVELS + ACCIDENT_LEVELS
LEVEL_INDEX: dict[str, FactorLevel] = {lvl.key: lvl for lvl in ALL_LEVELS}


def levels_of_environment(env: EnvironmentSpec) -> tuple[FactorLevel, ...]:
    """Return the four environment levels active in ``env``."""
    return (
        LEVEL_INDEX[f"illumination={env.illumination}"],
        LEVEL_INDEX[f"weather={env.weather}"],
        LEVEL_INDEX[f"road_surface={env.road_surface}"],
        LEVEL_INDEX[f"capture_quality={env.capture_quality}"],
    )


def levels_of_accident(accident: AccidentSpec) -> tuple[FactorLevel, ...]:
    """Return the three accident levels active in ``accident``."""
    return (
        LEVEL_INDEX[f"accident_type={accident.accident_type}"],
        LEVEL_INDEX[f"severity={accident.severity}"],
        LEVEL_INDEX[f"onset={accident.onset}"],
    )


def composition_key(env: EnvironmentSpec, accident: AccidentSpec) -> str:
    """Canonical identity of a factor composition (localization fields excluded)."""
    return "|".join(
        (
            env.illumination,
            env.weather,
            env.road_surface,
            env.capture_quality,
            accident.accident_type,
            accident.severity,
            accident.onset,
        )
    )


def iter_compositions() -> Iterator[tuple[EnvironmentSpec, AccidentSpec]]:
    """Iterate the full 54 x 54 grid of environment-accident compositions."""
    env_grid = itertools.product(
        ILLUMINATION_LEVELS, WEATHER_LEVELS, ROAD_SURFACE_LEVELS, CAPTURE_QUALITY_LEVELS
    )
    for illumination, weather, surface, quality in env_grid:
        env = EnvironmentSpec(illumination, weather, surface, quality)
        acc_grid = itertools.product(ACCIDENT_TYPE_LEVELS, SEVERITY_LEVELS, ONSET_LEVELS)
        for accident_type, severity, onset in acc_grid:
            yield env, AccidentSpec(accident_type, severity, onset)


@dataclass(frozen=True)
class EditContext:
    """Auxiliary attributes a candidate edit script may carry.

    ``wet_pavement_cause`` and ``light_rain_onset`` are script declarations:
    they default to absent, which makes the corresponding compositions
    infeasible until the script states them explicitly. ``participant_scale``
    (normalized bounding-box height of the largest scripted participant) and
    ``visible_road_fraction`` are source measurements: ``None`` means not yet
    measured, and the source-dependent rules defer until a value is attached.
    """

    wet_pavement_cause: Optional[str] = None
    light_rain_onset: bool = False
    participant_scale: Optional[float] = None
    visible_road_fraction: Optional[float] = None


FeasibilityRule = Callable[
    [EnvironmentSpec, AccidentSpec, EditContext], tuple[bool, str]
]


def wet_surface_requires_cause(
    env: EnvironmentSpec, accident: AccidentSpec, context: EditContext
) -> tuple[bool, str]:
    """Clear weather with a wet surface needs an explicit wet-pavement cause."""
    if env.weather == "clear" and env.road_surface == "wet":
        if not context.wet_pavement_cause:
            return False, "clear+wet needs an explicit wet-pavement cause in the script"
    return True, "ok"


def rain_over_dry_requires_transition(
    env: EnvironmentSpec, accident: AccidentSpec, context: EditContext
) -> tuple[bool, str]:
    """Rain over a dry surface is only plausible as a light-rain transitional onset."""
    if env.weather == "rain" and env.road_surface == "dry":
        if not context.light_rain_onset:
            return False, "rain+dry needs a light-rain transitional onset without pooling"
    return True, "ok"


def rollover_requires_scale_and_space(
    env: EnvironmentSpec, accident: AccidentSpec, context: EditContext
) -> tuple[bool, str]:
    """Rollover scripts need sufficient participant scale and visible road space.

    Both checks read source measurements; an unmeasured value defers the rule
    until the composition is instantiated on a concrete source clip.
    """
    if accident.accident_type != "rollover":
        return True, "ok"
    scale = context.participant_scale
    if scale is not None and scale < MIN_ROLLOVER_PARTICIPANT_SCALE:
        return False, "rollover needs a participant of sufficient scale in frame"
    road = context.visible_road_fraction
    if road is not None and road < MIN_ROLLOVER_ROAD_FRACTION:
        return False, "rollover needs sufficient visible road space"
    return True, "ok"


FEASIBILITY_RULES: tuple[FeasibilityRule, ...] = (
    wet_surface_requires_cause,
    rain_over_dry_requires_transition,
    rollover_requires_scale_and_space,
)


def check_feasibility(
    env: EnvironmentSpec,
    accident: AccidentSpec,
    context: Optional[EditContext] = None,
) -> tuple[bool, str]:
    """Evaluate the feasibility mask for one composition.

    Returns ``(True, "feasible")`` when every rule passes, otherwise the first
    failing rule's ``(False, reason)``.
    """
    ctx = context if context is not None else EditContext()
    for rule in FEASIBILITY_RULES:
        ok, reason = rule(env, accident, ctx)
        if not ok:
            return False, reason
    return True, "feasible"


def feasible_compositions(
    context: Optional[EditContext] = None,
    blocked: Collection[str] = (),
) -> list[tuple[EnvironmentSpec, AccidentSpec]]:
    """Enumerate feasible compositions, excluding ``blocked`` composition keys.

    Held-out composition keys are passed as ``blocked`` when sampling edit
    tuples for training and validation sources; test sources instead draw one
    or two of their at most six candidate tuples from the held-out set.
    """
    blocked_set = set(blocked)
    out: list[tuple[EnvironmentSpec, AccidentSpec]] = []
    for env, accident in iter_compositions():
        if composition_key(env, accident) in blocked_set:
            continue
        if check_feasibility(env, accident, context)[0]:
            out.append((env, accident))
    return out


def sample_held_out_compositions(
    rng: np.random.Generator,
    count: int = HELD_OUT_COUNT,
    context: Optional[EditContext] = None,
) -> list[tuple[EnvironmentSpec, AccidentSpec]]:
    """Draw the held-out compositions reserved for CGS.

    Stratified random selection: a greedy pass first guarantees that every
    accident type and every environment factor level appears in at least one
    held-out composition, then the remaining slots are filled uniformly at
    random from the feasible set. Deterministic given ``rng``.
    """
    pool = feasible_compositions(context)
    required = {lvl.key for lvl in ENVIRONMENT_LEVELS}
    required.update(f"accident_type={t}" for t in ACCIDENT_TYPE_LEVELS)

    def coverage(item: tuple[EnvironmentSpec, AccidentSpec]) -> set[str]:
        env, accident = item
        keys = {lvl.key for lvl in levels_of_environment(env)}
        keys.add(f"accident_type={accident.accident_type}")
        return keys

    order = [int(i) for i in rng.permutation(len(pool))]
    selected: list[int] = []
    uncovered = set(required)
    while uncovered:
        if len(selected) >= count:
            raise ValueError(
                f"cannot cover every accident type and environment level with {count} compositions"
            )
        best_idx, best_gain = -1, 0
        for idx in order:
            if idx in selected:
                continue
            gain = len(coverage(pool[idx]) & uncovered)
            if gain > best_gain:
                best_idx, best_gain = idx, gain
        selected.append(best_idx)
        uncovered -= coverage(pool[best_idx])

    remaining = [idx for idx in order if idx not in selected]
    fill = count - len(selected)
    if fill > 0:
        chosen = rng.choice(len(remaining), size=fill, replace=False)
        selected.extend(remaining[int(i)] for i in chosen)
    return [pool[idx] for idx in selected]


def held_out_keys(
    compositions: Collection[tuple[EnvironmentSpec, AccidentSpec]],
) -> frozenset[str]:
    """Composition keys of a held-out set, for blocking during sampling."""
    return frozenset(composition_key(env, acc) for env, acc in compositions)


__all__ = [
    "ACCIDENT_FACTORS",
    "ACCIDENT_LEVELS",
    "ACCIDENT_TYPE_LEVELS",
    "ALL_LEVELS",
    "CAPTURE_QUALITY_LEVELS",
    "ENVIRONMENT_FACTORS",
    "ENVIRONMENT_LEVELS",
    "EditContext",
    "FEASIBILITY_RULES",
    "FactorLevel",
    "HELD_OUT_COUNT",
    "HELD_OUT_PER_TEST_SOURCE",
    "ILLUMINATION_LEVELS",
    "LEVEL_INDEX",
    "MIN_ROLLOVER_PARTICIPANT_SCALE",
    "MIN_ROLLOVER_ROAD_FRACTION",
    "ONSET_LEVELS",
    "PER_SOURCE_CANDIDATE_CAP",
    "ROAD_SURFACE_LEVELS",
    "SEVERITY_LEVELS",
    "WEATHER_LEVELS",
    "WET_PAVEMENT_CAUSES",
    "check_feasibility",
    "composition_key",
    "feasible_compositions",
    "held_out_keys",
    "iter_compositions",
    "levels_of_accident",
    "levels_of_environment",
    "sample_held_out_compositions",
]
