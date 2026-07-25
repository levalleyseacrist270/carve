"""PMR-FSS weak-factor sampling and per-round generation budgets.

Generation probability follows P(c) = FSS(c)^alpha / sum FSS^alpha with
alpha = 1.5, which concentrates budget on the strongest failures while
keeping multiple factors active. A uniform variant over the feasible
composition set serves as the random-quartet baseline under the same budget.

The per-round budget is counted in generated edit entries: each accepted
quartet contributes VE, VA, and VAE (three entries), while V0 is the paired
real source and is not counted. The target is 25% of the real training-clip
count; because roughly a quarter of candidates fail the gates, the generator
oversamples candidate tuples by about one third and stops once the accepted
target is reached.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Collection, Mapping, Optional

import numpy as np

from ..core import AccidentSpec, EnvironmentSpec
from ..factors import (
    EditContext,
    feasible_compositions,
    levels_of_accident,
    levels_of_environment,
)

DEFAULT_ALPHA = 1.5
EDITS_PER_QUARTET = 3
DEFAULT_BUDGET_FRACTION = 0.25
DEFAULT_OVERSAMPLE = 1.0 / 3.0


def sampling_probabilities(
    stress: Mapping[str, float], alpha: float = DEFAULT_ALPHA
) -> dict[str, float]:
    """P(c) = FSS(c)^alpha / sum FSS^alpha; uniform when all stress is zero."""
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    keys = sorted(stress)
    if not keys:
        raise ValueError("stress scores are empty")
    values = np.asarray([stress[k] for k in keys], dtype=float)
    if (values < 0).any():
        raise ValueError("stress scores must be non-negative")
    powered = values**alpha
    total = powered.sum()
    if total <= 0:
        powered = np.ones_like(powered)
        total = powered.sum()
    return {k: float(p / total) for k, p in zip(keys, powered)}


@dataclass(frozen=True)
class RoundBudget:
    """Accepted-edit budget for one hardening round.

    ``real_clip_count`` is the number of real training clips of the source
    dataset; the accepted target is ``ceil(fraction * real_clip_count)``
    generated edit entries, reached in whole quartets of three entries.
    """

    real_clip_count: int
    fraction: float = DEFAULT_BUDGET_FRACTION
    oversample: float = DEFAULT_OVERSAMPLE

    @property
    def edit_target(self) -> int:
        """Generated edit entries to accept this round."""
        return math.ceil(self.fraction * self.real_clip_count)

    @property
    def quartet_target(self) -> int:
        """Accepted quartets needed to reach the edit target."""
        return math.ceil(self.edit_target / EDITS_PER_QUARTET)

    @property
    def candidate_target(self) -> int:
        """Candidate tuples to draw, oversampled for expected gate rejections."""
        return math.ceil(self.quartet_target * (1.0 + self.oversample))

    def reached(self, accepted_quartets: int) -> bool:
        """Whether the accepted quartets cover the edit target."""
        return accepted_quartets * EDITS_PER_QUARTET >= self.edit_target


class WeakFactorSampler:
    """Sample target factor compositions for the next generation round.

    A target level is drawn from the PMR-FSS distribution, then a full
    composition is drawn uniformly among feasible compositions containing
    that level. Held-out composition keys are passed as ``blocked`` when
    generating for training and validation sources. With ``uniform=True``
    compositions are drawn uniformly from the feasible set instead (the
    random-quartet baseline). The per-source cap of six candidate tuples
    applies to benchmark construction only; hardening rounds may revisit a
    source beyond it.
    """

    def __init__(
        self,
        stress: Mapping[str, float],
        alpha: float = DEFAULT_ALPHA,
        uniform: bool = False,
        context: Optional[EditContext] = None,
        blocked: Collection[str] = (),
    ) -> None:
        self.uniform = uniform
        self.pool: tuple[tuple[EnvironmentSpec, AccidentSpec], ...] = tuple(
            feasible_compositions(context, blocked)
        )
        if not self.pool:
            raise ValueError("no feasible compositions remain after blocking")
        members: dict[str, list[int]] = {}
        for idx, (env, accident) in enumerate(self.pool):
            for lvl in levels_of_environment(env) + levels_of_accident(accident):
                members.setdefault(lvl.key, []).append(idx)
        supported = {k: v for k, v in stress.items() if k in members}
        if not supported:
            raise ValueError("no stress level maps into the feasible composition pool")
        self._probabilities = sampling_probabilities(supported, alpha)
        self._level_keys = list(self._probabilities)
        self._level_probs = np.asarray(
            [self._probabilities[k] for k in self._level_keys], dtype=float
        )
        self._members = {k: np.asarray(members[k]) for k in self._level_keys}

    @property
    def probabilities(self) -> dict[str, float]:
        """Level-sampling distribution P(c) actually in use."""
        return dict(self._probabilities)

    def sample_level(self, rng: np.random.Generator) -> str:
        """Draw one target factor level from P(c)."""
        idx = rng.choice(len(self._level_keys), p=self._level_probs)
        return self._level_keys[int(idx)]

    def sample_composition(
        self, rng: np.random.Generator
    ) -> tuple[EnvironmentSpec, AccidentSpec]:
        """Draw one feasible target composition."""
        if self.uniform:
            return self.pool[int(rng.integers(len(self.pool)))]
        members = self._members[self.sample_level(rng)]
        return self.pool[int(members[int(rng.integers(members.size))])]

    def sample(
        self, rng: np.random.Generator, count: int
    ) -> list[tuple[EnvironmentSpec, AccidentSpec]]:
        """Draw ``count`` compositions (with replacement across draws)."""
        return [self.sample_composition(rng) for _ in range(count)]


__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_BUDGET_FRACTION",
    "DEFAULT_OVERSAMPLE",
    "EDITS_PER_QUARTET",
    "RoundBudget",
    "WeakFactorSampler",
    "sampling_probabilities",
]
