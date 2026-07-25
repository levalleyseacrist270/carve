#!/usr/bin/env python
"""Select the operating threshold tau on a validation split by F1 sweep.

Input is a scored validation file: CSV with ``score`` and ``label`` columns,
JSON lines with ``{"score": ..., "label": ...}`` objects, or an ``.npz``
archive with ``scores`` and ``labels`` arrays. Threshold selection runs on
the source-dataset validation split only and is repeated at the start of
every hardening round; target test clips and test quartets stay out of the
sweep.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
from pathlib import Path


def _ensure_package() -> None:
    try:
        importlib.import_module("carve")
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_scores(
    path: Path, score_key: str = "score", label_key: str = "label"
) -> tuple[list[float], list[int]]:
    """Read (scores, labels) from a CSV, JSON-lines, or npz file."""
    suffix = path.suffix.lower()
    scores: list[float] = []
    labels: list[int] = []
    if suffix == ".npz":
        import numpy as np

        data = np.load(path)
        return [float(s) for s in data["scores"]], [int(y) for y in data["labels"]]
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                scores.append(float(row[score_key]))
                labels.append(int(row[label_key]))
        return scores, labels
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            scores.append(float(row[score_key]))
            labels.append(int(row[label_key]))
    return scores, labels


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--scores", type=Path, required=True, help="validation scores (.csv, .jsonl, or .npz)"
    )
    parser.add_argument("--score-key", default="score", help="score column or field name")
    parser.add_argument("--label-key", default="label", help="label column or field name")
    parser.add_argument("--output", type=Path, default=None, help="optional JSON output path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _ensure_package()

    from carve.metrics import select_threshold, threshold_metrics

    scores, labels = load_scores(args.scores, args.score_key, args.label_key)
    tau = select_threshold(scores, labels)
    summary = threshold_metrics(scores, labels, tau)
    summary["n"] = len(scores)
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
