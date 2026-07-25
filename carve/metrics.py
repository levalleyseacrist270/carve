"""Quartet audit metrics, factor-level diagnostics, and threshold selection.

The audit reads a pool of matched quartets (V0, VE, VA, VAE) scored by a
detector and decomposes robustness into complementary quantities:

* CPS: fraction of environment-only edits kept below the operating threshold.
* V0 FPR: false-positive rate on the unedited source clips, the corpus-shift
  control read against 1 - CPS.
* NESI / ESI: mean absolute score drift on negative and positive pairs.
* CFI: mean score increase produced by accident insertion under both the
  source and the edited environment.
* CGS: unseen-over-seen AUC ratio and absolute gap on held-out compositions.
* PMR: fraction of positive branches at or below the threshold.
* FSS: weighted aggregate of normalized per-factor brittleness that feeds the
  PMR-FSS weak-factor sampler.

All functions are pure numpy; detector scores are read from
``QuartetRecord.scores`` keyed by branch name.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

from .core import Branch, QuartetRecord
from .factors import (
    LEVEL_INDEX,
    FactorLevel,
    composition_key,
    levels_of_accident,
    levels_of_environment,
)

#: FSS term weights for (1-CPS, NESI, ESI, PMR, CGS gap), in that order.
FSS_WEIGHTS: tuple[float, float, float, float, float] = (0.25, 0.15, 0.20, 0.25, 0.15)

#: Empirical-Bayes shrinkage strength for per-level validation slices.
DEFAULT_SHRINKAGE = 8.0


def _as_array(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=float)
    if arr.size == 0:
        raise ValueError("metric computed on an empty score array")
    return arr


def auc_score(negatives: Iterable[float], positives: Iterable[float]) -> Optional[float]:
    """Rank-based AUC with average-tie handling; ``None`` if a side is empty."""
    neg = np.asarray(list(negatives), dtype=float)
    pos = np.asarray(list(positives), dtype=float)
    if neg.size == 0 or pos.size == 0:
        return None
    combined = np.concatenate([neg, pos])
    uniq, inverse, counts = np.unique(combined, return_inverse=True, return_counts=True)
    starts = np.concatenate(([0.0], np.cumsum(counts)[:-1]))
    ranks = (starts + (counts + 1) / 2.0)[inverse]
    pos_ranks = ranks[neg.size:]
    u = pos_ranks.sum() - pos.size * (pos.size + 1) / 2.0
    return float(u / (pos.size * neg.size))


def cps(env_scores: Iterable[float], tau: float) -> float:
    """Causal Purity Score: fraction of environment-only edits at or below tau."""
    p = _as_array(env_scores)
    return float(1.0 - np.mean(p > tau))


def v0_false_positive_rate(source_scores: Iterable[float], tau: float) -> float:
    """False-positive rate on unedited source clips at the operating threshold."""
    p = _as_array(source_scores)
    return float(np.mean(p > tau))


def nesi(source_scores: Iterable[float], env_scores: Iterable[float]) -> float:
    """Negative environment sensitivity: mean |p(VE) - p(V0)|."""
    p0, pe = _as_array(source_scores), _as_array(env_scores)
    return float(np.mean(np.abs(pe - p0)))


def esi(accident_scores: Iterable[float], joint_scores: Iterable[float]) -> float:
    """Environment sensitivity on positives: mean |p(VAE) - p(VA)|."""
    pa, pae = _as_array(accident_scores), _as_array(joint_scores)
    return float(np.mean(np.abs(pae - pa)))


def cfi(
    source_scores: Iterable[float],
    env_scores: Iterable[float],
    accident_scores: Iterable[float],
    joint_scores: Iterable[float],
) -> float:
    """Counterfactual Faithfulness Index: mean accident-induced score increase."""
    p0, pe = _as_array(source_scores), _as_array(env_scores)
    pa, pae = _as_array(accident_scores), _as_array(joint_scores)
    return float(np.mean((pa - p0 + pae - pe) / 2.0))


def pmr(accident_scores: Iterable[float], joint_scores: Iterable[float], tau: float) -> float:
    """Positive miss rate: fraction of VA and VAE branches at or below tau."""
    pa, pae = _as_array(accident_scores), _as_array(joint_scores)
    return float((np.mean(pa <= tau) + np.mean(pae <= tau)) / 2.0)


@dataclass(frozen=True)
class CGSResult:
    """Compositional generalization: seen and unseen pool AUCs.

    The ratio is read jointly with the absolute AUCs; a ratio near one is
    uninformative when both pools sit near chance level.
    """

    auc_seen: Optional[float]
    auc_unseen: Optional[float]

    @property
    def ratio(self) -> Optional[float]:
        if self.auc_seen is None or self.auc_unseen is None or self.auc_seen <= 0.0:
            return None
        return self.auc_unseen / self.auc_seen

    @property
    def ratio_percent(self) -> Optional[float]:
        """100 x ratio, the form used in summaries."""
        return None if self.ratio is None else 100.0 * self.ratio

    @property
    def gap(self) -> Optional[float]:
        if self.auc_seen is None or self.auc_unseen is None:
            return None
        return self.auc_seen - self.auc_unseen


def branch_scores(records: Sequence[QuartetRecord], branch: Branch) -> np.ndarray:
    """Collect one branch's cached detector scores across records."""
    out = np.empty(len(records), dtype=float)
    for i, record in enumerate(records):
        try:
            out[i] = record.scores[branch.value]
        except KeyError as exc:
            raise KeyError(
                f"quartet {record.quartet_id!r} has no score for branch {branch.value!r}"
            ) from exc
    return out


def _pool_scores(records: Sequence[QuartetRecord]) -> tuple[np.ndarray, np.ndarray]:
    negatives = np.concatenate(
        [branch_scores(records, Branch.V0), branch_scores(records, Branch.VE)]
    )
    positives = np.concatenate(
        [branch_scores(records, Branch.VA), branch_scores(records, Branch.VAE)]
    )
    return negatives, positives


def compute_cgs(records: Sequence[QuartetRecord]) -> CGSResult:
    """Seen/unseen pool AUCs; each pool uses {V0, VE} negatives, {VA, VAE} positives."""
    seen = [r for r in records if not r.held_out_composition]
    unseen = [r for r in records if r.held_out_composition]
    auc_seen = auc_score(*_pool_scores(seen)) if seen else None
    auc_unseen = auc_score(*_pool_scores(unseen)) if unseen else None
    return CGSResult(auc_seen=auc_seen, auc_unseen=auc_unseen)


@dataclass(frozen=True)
class AuditSummary:
    """Pool-level audit of one detector at one operating threshold."""

    quartets: int
    tau: float
    cps: float
    v0_fpr: float
    nesi: float
    esi: float
    cfi: float
    pmr: float
    cgs: CGSResult

    def as_dict(self) -> dict[str, Optional[float]]:
        """Flat summary; CGS reported as 100 x ratio alongside absolute AUCs."""
        return {
            "quartets": self.quartets,
            "tau": self.tau,
            "cps": self.cps,
            "v0_fpr": self.v0_fpr,
            "nesi": self.nesi,
            "esi": self.esi,
            "cfi": self.cfi,
            "pmr": self.pmr,
            "cgs": self.cgs.ratio_percent,
            "cgs_gap": self.cgs.gap,
            "auc_seen": self.cgs.auc_seen,
            "auc_unseen": self.cgs.auc_unseen,
        }


def audit_quartets(records: Sequence[QuartetRecord], tau: float) -> AuditSummary:
    """Run the full audit over a scored quartet pool."""
    records = list(records)
    if not records:
        raise ValueError("audit requires at least one quartet")
    p0 = branch_scores(records, Branch.V0)
    pe = branch_scores(records, Branch.VE)
    pa = branch_scores(records, Branch.VA)
    pae = branch_scores(records, Branch.VAE)
    return AuditSummary(
        quartets=len(records),
        tau=float(tau),
        cps=cps(pe, tau),
        v0_fpr=v0_false_positive_rate(p0, tau),
        nesi=nesi(p0, pe),
        esi=esi(pa, pae),
        cfi=cfi(p0, pe, pa, pae),
        pmr=pmr(pa, pae, tau),
        cgs=compute_cgs(records),
    )


def empirical_bayes_shrink(
    value: float, n: int, pooled: float, strength: float = DEFAULT_SHRINKAGE
) -> float:
    """Shrink a slice statistic toward the pooled mean with weight n / (n + strength)."""
    if strength < 0:
        raise ValueError("shrinkage strength must be non-negative")
    if n <= 0:
        return float(pooled)
    w = n / (n + strength)
    return float(w * value + (1.0 - w) * pooled)


@dataclass(frozen=True)
class LevelDiagnostics:
    """Per-factor-level slice statistics after empirical-Bayes shrinkage.

    ``cps`` and ``nesi`` are defined for environment levels, ``pmr`` for
    accident levels, ``esi`` for all levels, and ``cgs_gap`` only where the
    audit pool contains held-out compositions with that level.
    """

    level: FactorLevel
    n: int
    cps: Optional[float] = None
    nesi: Optional[float] = None
    esi: Optional[float] = None
    pmr: Optional[float] = None
    cgs_gap: Optional[float] = None


def _composition_gaps(
    records: Sequence[QuartetRecord],
) -> tuple[dict[str, tuple[float, int, frozenset[str]]], Optional[float]]:
    """Per-held-out-composition seen-unseen AUC gaps and their pooled mean."""
    seen = [r for r in records if not r.held_out_composition]
    unseen = [r for r in records if r.held_out_composition]
    if not seen or not unseen:
        return {}, None
    auc_seen = auc_score(*_pool_scores(seen))
    if auc_seen is None:
        return {}, None
    groups: dict[str, list[QuartetRecord]] = {}
    for record in unseen:
        groups.setdefault(composition_key(record.env, record.accident), []).append(record)
    gaps: dict[str, tuple[float, int, frozenset[str]]] = {}
    for key, group in groups.items():
        auc_unseen = auc_score(*_pool_scores(group))
        if auc_unseen is None:
            continue
        level_keys = frozenset(
            lvl.key
            for lvl in levels_of_environment(group[0].env) + levels_of_accident(group[0].accident)
        )
        gaps[key] = (auc_seen - auc_unseen, len(group), level_keys)
    pooled = float(np.mean([g for g, _, _ in gaps.values()])) if gaps else None
    return gaps, pooled


def per_level_diagnostics(
    records: Sequence[QuartetRecord],
    tau: float,
    shrinkage_strength: float = DEFAULT_SHRINKAGE,
) -> dict[str, LevelDiagnostics]:
    """Slice the audit by factor level, with empirical-Bayes stabilization.

    Each slice statistic is shrunk toward its pooled value with weight
    n_c / (n_c + strength) before any normalization, which stabilizes small
    validation slices. Levels absent from the pool are omitted.
    """
    records = list(records)
    if not records:
        raise ValueError("diagnostics require at least one quartet")
    p0 = branch_scores(records, Branch.V0)
    pe = branch_scores(records, Branch.VE)
    pa = branch_scores(records, Branch.VA)
    pae = branch_scores(records, Branch.VAE)

    pooled_fa = float(np.mean(pe > tau))
    pooled_nesi = nesi(p0, pe)
    pooled_esi = esi(pa, pae)
    pooled_pmr = pmr(pa, pae, tau)
    gaps, pooled_gap = _composition_gaps(records)

    env_members: dict[str, list[int]] = {}
    acc_members: dict[str, list[int]] = {}
    for i, record in enumerate(records):
        for lvl in levels_of_environment(record.env):
            env_members.setdefault(lvl.key, []).append(i)
        for lvl in levels_of_accident(record.accident):
            acc_members.setdefault(lvl.key, []).append(i)

    def shrink(value: float, n: int, pooled: float) -> float:
        return empirical_bayes_shrink(value, n, pooled, shrinkage_strength)

    out: dict[str, LevelDiagnostics] = {}
    for key, idx in env_members.items():
        sel = np.asarray(idx)
        n = sel.size
        fa_c = shrink(float(np.mean(pe[sel] > tau)), n, pooled_fa)
        out[key] = LevelDiagnostics(
            level=LEVEL_INDEX[key],
            n=n,
            cps=1.0 - fa_c,
            nesi=shrink(float(np.mean(np.abs(pe[sel] - p0[sel]))), n, pooled_nesi),
            esi=shrink(float(np.mean(np.abs(pae[sel] - pa[sel]))), n, pooled_esi),
        )
    for key, idx in acc_members.items():
        sel = np.asarray(idx)
        n = sel.size
        miss = (np.mean(pa[sel] <= tau) + np.mean(pae[sel] <= tau)) / 2.0
        out[key] = LevelDiagnostics(
            level=LEVEL_INDEX[key],
            n=n,
            esi=shrink(float(np.mean(np.abs(pae[sel] - pa[sel]))), n, pooled_esi),
            pmr=shrink(float(miss), n, pooled_pmr),
        )

    if gaps and pooled_gap is not None:
        for key, diag in list(out.items()):
            touching = [(g, n_u) for g, n_u, lvls in gaps.values() if key in lvls]
            if not touching:
                continue
            raw = float(np.mean([g for g, _ in touching]))
            n_u = int(sum(n for _, n in touching))
            out[key] = LevelDiagnostics(
                level=diag.level,
                n=diag.n,
                cps=diag.cps,
                nesi=diag.nesi,
                esi=diag.esi,
                pmr=diag.pmr,
                cgs_gap=shrink(raw, n_u, pooled_gap),
            )
    return out


def _minmax(values: dict[str, float]) -> dict[str, float]:
    """Min-max normalize within a term's defined levels; constant terms map to 0."""
    if not values:
        return {}
    arr = np.asarray(list(values.values()), dtype=float)
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-12:
        return {key: 0.0 for key in values}
    return {key: (val - lo) / (hi - lo) for key, val in values.items()}


def factor_stress_scores(
    diagnostics: Mapping[str, LevelDiagnostics],
    weights: tuple[float, float, float, float, float] = FSS_WEIGHTS,
) -> dict[str, float]:
    """Factor stress score FSS(c) on a common [0, 1] scale.

    Terms are min-max normalized within the levels where they are defined:
    ESI across all levels, CPS and NESI across the environment levels, PMR
    across the accident levels, and the CGS gap across the levels touched by
    held-out compositions. When a term is undefined for a factor group or
    unavailable in the audit pool, the remaining weights are renormalized to
    sum to one.
    """
    false_alarm = _minmax(
        {k: 1.0 - d.cps for k, d in diagnostics.items() if d.cps is not None}
    )
    neg_drift = _minmax({k: d.nesi for k, d in diagnostics.items() if d.nesi is not None})
    pos_drift = _minmax({k: d.esi for k, d in diagnostics.items() if d.esi is not None})
    miss = _minmax({k: d.pmr for k, d in diagnostics.items() if d.pmr is not None})
    gap = _minmax({k: d.cgs_gap for k, d in diagnostics.items() if d.cgs_gap is not None})

    stress: dict[str, float] = {}
    for key in diagnostics:
        terms = (
            (weights[0], false_alarm.get(key)),
            (weights[1], neg_drift.get(key)),
            (weights[2], pos_drift.get(key)),
            (weights[3], miss.get(key)),
            (weights[4], gap.get(key)),
        )
        available = [(w, t) for w, t in terms if t is not None]
        if not available:
            stress[key] = 0.0
            continue
        total_weight = sum(w for w, _ in available)
        stress[key] = float(sum(w * t for w, t in available) / total_weight)
    return stress


def select_threshold(scores: Iterable[float], labels: Iterable[int]) -> float:
    """Operating threshold maximizing F1 on a validation split.

    Predictions use the strict rule ``score > tau``, matching the thresholded
    audit metrics. Selection runs on the source-dataset validation split only;
    target test clips and test quartets never enter this sweep. On F1 ties the
    higher threshold wins.
    """
    s = np.asarray(list(scores), dtype=float)
    y = np.asarray(list(labels), dtype=int)
    if s.size == 0 or s.size != y.size:
        raise ValueError("scores and labels must be non-empty and aligned")
    if not np.isfinite(s).all():
        raise ValueError("scores must be finite")
    if y.min() == y.max():
        raise ValueError("threshold calibration needs both classes")
    order = np.argsort(-s, kind="stable")
    s_sorted, y_sorted = s[order], y[order]
    tp = np.cumsum(y_sorted)
    total_pos = int(tp[-1])
    boundaries = np.flatnonzero(
        np.concatenate([s_sorted[:-1] > s_sorted[1:], [True]])
    )
    predicted = boundaries + 1
    f1 = 2.0 * tp[boundaries] / (predicted + total_pos)
    best = int(boundaries[np.argmax(f1)])
    if best < s.size - 1:
        return float((s_sorted[best] + s_sorted[best + 1]) / 2.0)
    return float(s_sorted[-1] - 1e-6)


def threshold_metrics(
    scores: Iterable[float], labels: Iterable[int], tau: float
) -> dict[str, float]:
    """Precision, recall, and F1 of ``score > tau`` predictions."""
    s = np.asarray(list(scores), dtype=float)
    y = np.asarray(list(labels), dtype=int)
    pred = s > tau
    tp = float(np.sum(pred & (y == 1)))
    fp = float(np.sum(pred & (y == 0)))
    fn = float(np.sum(~pred & (y == 1)))
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {"tau": float(tau), "precision": precision, "recall": recall, "f1": f1}


__all__ = [
    "AuditSummary",
    "CGSResult",
    "DEFAULT_SHRINKAGE",
    "FSS_WEIGHTS",
    "LevelDiagnostics",
    "audit_quartets",
    "auc_score",
    "branch_scores",
    "cfi",
    "compute_cgs",
    "cps",
    "empirical_bayes_shrink",
    "esi",
    "factor_stress_scores",
    "nesi",
    "per_level_diagnostics",
    "pmr",
    "select_threshold",
    "threshold_metrics",
    "v0_false_positive_rate",
]
