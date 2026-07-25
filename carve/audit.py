"""Multi-detector benchmark audit.

Scores every quartet branch with each detector, calibrates the operating
point on validation quartets when none is supplied, and assembles the audit
table: clean AUC on the unedited branch pair, the false-positive rate on real
source clips, and the quartet robustness metrics. Per-factor diagnostics are
exported alongside the table for weak-factor analysis.
"""

from __future__ import annotations

import csv
import dataclasses
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from carve.core import Branch, QuartetRecord
from carve.metrics import audit_quartets, per_level_diagnostics, select_threshold

Detector = Callable[[str], float]

_NEGATIVE_BRANCHES = (Branch.V0, Branch.VE)
_POSITIVE_BRANCHES = (Branch.VA, Branch.VAE)


def score_records(records: Sequence[QuartetRecord], detector: Detector) -> None:
    """Fill ``record.scores`` with branch probabilities from one detector.

    Existing scores are overwritten, so records can be reused across the
    detectors of one audit run.
    """
    for record in records:
        record.scores.clear()
        for branch in Branch:
            path = record.paths.get(branch.value)
            if path is not None:
                record.scores[branch.value] = float(detector(path))


def _branch_scores(records: Sequence[QuartetRecord], branches: Sequence[Branch]) -> np.ndarray:
    values = [
        record.scores[branch.value]
        for record in records
        for branch in branches
        if branch.value in record.scores
    ]
    return np.asarray(values, dtype=float)


def rank_auc(negatives: np.ndarray, positives: np.ndarray) -> float:
    """Rank-based AUC with tie correction."""
    if negatives.size == 0 or positives.size == 0:
        return float("nan")
    pooled = np.concatenate([negatives, positives])
    ranks = np.empty_like(pooled)
    order = np.argsort(pooled, kind="mergesort")
    ranks[order] = np.arange(1, pooled.size + 1, dtype=float)
    for value in np.unique(pooled):
        tied = pooled == value
        if tied.sum() > 1:
            ranks[tied] = ranks[tied].mean()
    rank_sum = ranks[negatives.size :].sum()
    n_pos, n_neg = positives.size, negatives.size
    return float((rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def clean_branch_auc(records: Sequence[QuartetRecord]) -> float:
    """AUC on the clean branch pair: real sources negative, accident edits positive."""
    return rank_auc(
        _branch_scores(records, (Branch.V0,)), _branch_scores(records, (Branch.VA,))
    )


def calibration_pairs(records: Sequence[QuartetRecord]) -> tuple[np.ndarray, np.ndarray]:
    """Scores and binary labels over all branches, for threshold calibration."""
    scores = np.concatenate(
        [_branch_scores(records, _NEGATIVE_BRANCHES), _branch_scores(records, _POSITIVE_BRANCHES)]
    )
    labels = np.concatenate(
        [
            np.zeros(_branch_scores(records, _NEGATIVE_BRANCHES).size),
            np.ones(_branch_scores(records, _POSITIVE_BRANCHES).size),
        ]
    )
    return scores, labels


def _report_row(report, records: Sequence[QuartetRecord]) -> dict[str, float | None]:
    """Flatten one audit summary into a table row, prepending the clean AUC.

    The summary already reports CGS as 100 x the unseen/seen AUC ratio; pools
    without held-out compositions yield ``None`` for the CGS entries.
    """
    return {"clean_auc": clean_branch_auc(records), **report.as_dict()}


def audit_detectors(
    detectors: Mapping[str, Detector],
    records: Sequence[QuartetRecord],
    tau: float | None = None,
    calibration_records: Sequence[QuartetRecord] | None = None,
) -> tuple[dict[str, dict[str, float | None]], dict[str, dict]]:
    """Audit every detector on the given quartet records.

    Args:
        detectors: Mapping of detector name to a callable that scores a clip
            path with an accident probability.
        records: Audit pool, typically the test split.
        tau: Fixed operating point. When ``None``, a per-detector threshold
            is calibrated on ``calibration_records``.
        calibration_records: Validation quartets used for calibration; must
            be disjoint from ``records`` so the operating point never sees
            the audit pool.

    Returns:
        The audit table rows keyed by detector name, and the per-factor
        diagnostics keyed by detector name.
    """
    if tau is None and calibration_records is None:
        raise ValueError("either a fixed tau or calibration_records is required")
    table: dict[str, dict[str, float | None]] = {}
    factors: dict[str, dict] = {}
    for name, detector in detectors.items():
        if tau is None:
            score_records(calibration_records, detector)
            scores, labels = calibration_pairs(calibration_records)
            detector_tau = float(select_threshold(scores, labels))
        else:
            detector_tau = float(tau)
        score_records(records, detector)
        report = audit_quartets(records, detector_tau)
        table[name] = _report_row(report, records)
        factors[name] = _to_plain(per_level_diagnostics(records, detector_tau))
    return table, factors


def _to_plain(value):
    """Recursively convert diagnostics into JSON-serializable structures."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _to_plain(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def export_audit(
    table: Mapping[str, Mapping[str, float | None]],
    factors: Mapping[str, Mapping],
    out_dir: str | Path,
) -> None:
    """Write the audit table and per-factor diagnostics as CSV and JSON."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "audit_table.json", "w", encoding="utf-8") as handle:
        json.dump(_to_plain(dict(table)), handle, indent=2)
    columns = ["detector", *next(iter(table.values())).keys()] if table else ["detector"]
    with open(out / "audit_table.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for name, row in table.items():
            writer.writerow(
                [name, *("" if row[c] is None else f"{row[c]:.4f}" for c in columns[1:])]
            )
    with open(out / "per_factor_diagnostics.json", "w", encoding="utf-8") as handle:
        json.dump(_to_plain(dict(factors)), handle, indent=2)
    for name, diagnostics in factors.items():
        _write_factor_csv(diagnostics, out / f"per_factor_{name}.csv")


def _write_factor_csv(diagnostics: Mapping, path: Path) -> None:
    """Flatten one detector's per-factor diagnostics into a CSV."""
    plain = _to_plain(diagnostics)
    rows: list[dict] = []
    for level, values in plain.items():
        row = {"factor_level": level}
        row.update(values if isinstance(values, Mapping) else {"value": values})
        rows.append(row)
    if not rows:
        return
    columns: list[str] = []
    for row in rows:
        columns += [c for c in row if c not in columns]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
