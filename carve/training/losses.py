"""Quartet-aware training losses.

The objective couples plain supervision with three constraints aligned to the
audit metrics: environment purity (CPS/NESI) keeps both negative branches
low, intervention consistency (ESI) stabilizes accident confidence across
matched positive branches, and the faithfulness margin (CFI) separates each
accident branch from its matched non-accident counterpart.

Total objective: L = L_sup + 0.5 * L_ep + 1.0 * L_ic + 0.25 * L_fm.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

_EPS = 1e-6


@dataclass(frozen=True)
class LossWeights:
    """Loss coefficients and the faithfulness margin."""

    environment_purity: float = 0.5
    intervention_consistency: float = 1.0
    faithfulness_margin: float = 0.25
    margin: float = 0.4


def _bce(probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Numerically safe BCE on probabilities."""
    return F.binary_cross_entropy(probs.clamp(_EPS, 1.0 - _EPS), targets)


def binary_supervision_loss(probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """BCE over labeled clips; quartet branches carry y(V0)=y(VE)=0, y(VA)=y(VAE)=1."""
    return _bce(probs, labels.to(probs.dtype))


def environment_purity_loss(p_source: torch.Tensor, p_env: torch.Tensor) -> torch.Tensor:
    """L_ep = mean over quartets of BCE(p(V0), 0) + BCE(p(VE), 0)."""
    zeros = torch.zeros_like(p_source)
    return _bce(p_source, zeros) + _bce(p_env, torch.zeros_like(p_env))


def intervention_consistency_loss(
    p_accident: torch.Tensor, p_joint: torch.Tensor
) -> torch.Tensor:
    """L_ic = mean |p(VA) - p(VAE)|."""
    return (p_accident - p_joint).abs().mean()


def faithfulness_margin_loss(
    p_source: torch.Tensor,
    p_env: torch.Tensor,
    p_accident: torch.Tensor,
    p_joint: torch.Tensor,
    margin: float = 0.4,
) -> torch.Tensor:
    """L_fm: hinge separating each accident branch from its matched negative.

    L_fm = mean of max(0, margin - p(VA) + p(V0)) + max(0, margin - p(VAE) + p(VE)).
    """
    source_side = torch.relu(margin - p_accident + p_source)
    edited_side = torch.relu(margin - p_joint + p_env)
    return (source_side + edited_side).mean()


@dataclass
class LossBreakdown:
    """Total objective plus its components, kept as tensors for backward."""

    total: torch.Tensor
    supervised: torch.Tensor
    environment_purity: torch.Tensor
    intervention_consistency: torch.Tensor
    faithfulness_margin: torch.Tensor

    def as_floats(self) -> dict[str, float]:
        return {
            "total": float(self.total.detach()),
            "supervised": float(self.supervised.detach()),
            "environment_purity": float(self.environment_purity.detach()),
            "intervention_consistency": float(self.intervention_consistency.detach()),
            "faithfulness_margin": float(self.faithfulness_margin.detach()),
        }


def hardening_objective(
    real_probs: Optional[torch.Tensor],
    real_labels: Optional[torch.Tensor],
    quartet_probs: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    weights: LossWeights = LossWeights(),
) -> LossBreakdown:
    """Combine supervision over all labeled clips with the quartet losses.

    ``quartet_probs`` is ``(p(V0), p(VE), p(VA), p(VAE))`` for a batch of
    matched quartets; either the real batch or the quartet batch may be
    absent (pretraining passes no quartets), but not both.
    """
    sup_probs: list[torch.Tensor] = []
    sup_labels: list[torch.Tensor] = []
    if real_probs is not None and real_probs.numel() > 0:
        if real_labels is None:
            raise ValueError("real_probs given without real_labels")
        sup_probs.append(real_probs)
        sup_labels.append(real_labels.to(real_probs.dtype))
    if quartet_probs is not None:
        p_source, p_env, p_accident, p_joint = quartet_probs
        sup_probs.extend([p_source, p_env, p_accident, p_joint])
        sup_labels.extend(
            [
                torch.zeros_like(p_source),
                torch.zeros_like(p_env),
                torch.ones_like(p_accident),
                torch.ones_like(p_joint),
            ]
        )
    if not sup_probs:
        raise ValueError("objective needs a real batch, a quartet batch, or both")
    supervised = _bce(torch.cat(sup_probs), torch.cat(sup_labels))

    if quartet_probs is not None:
        ep = environment_purity_loss(p_source, p_env)
        ic = intervention_consistency_loss(p_accident, p_joint)
        fm = faithfulness_margin_loss(p_source, p_env, p_accident, p_joint, weights.margin)
    else:
        zero = torch.zeros((), dtype=supervised.dtype, device=supervised.device)
        ep = ic = fm = zero
    total = (
        supervised
        + weights.environment_purity * ep
        + weights.intervention_consistency * ic
        + weights.faithfulness_margin * fm
    )
    return LossBreakdown(
        total=total,
        supervised=supervised,
        environment_purity=ep,
        intervention_consistency=ic,
        faithfulness_margin=fm,
    )


__all__ = [
    "LossBreakdown",
    "LossWeights",
    "binary_supervision_loss",
    "environment_purity_loss",
    "faithfulness_margin_loss",
    "hardening_objective",
    "intervention_consistency_loss",
]
