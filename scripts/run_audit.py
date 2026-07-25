"""Run the multi-detector audit on a set of quartet records.

Detectors are referenced as ``name=module:attribute`` where the attribute is
a callable mapping a clip path to an accident probability. The operating
point is either fixed with ``--tau`` or calibrated per detector on a
validation record manifest.

Example:
    python scripts/run_audit.py \
        --records out/benchmark/quartets_test.jsonl \
        --calibration-records out/benchmark/quartets_val.jsonl \
        --detector videomae=my_models.detectors:videomae_b \
        --out out/audit
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from carve.audit import Detector, audit_detectors, export_audit
from carve.data.quartets import load_manifest


def load_detector(spec: str) -> tuple[str, Detector]:
    """Resolve ``name=module:attribute`` into a named scoring callable."""
    try:
        name, target = spec.split("=", 1)
        module_name, attribute = target.split(":", 1)
    except ValueError as error:
        raise SystemExit(f"invalid detector spec {spec!r}; use name=module:attribute") from error
    detector = getattr(importlib.import_module(module_name), attribute)
    if not callable(detector):
        raise SystemExit(f"detector {spec!r} does not resolve to a callable")
    return name, detector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--records", required=True, help="JSON-lines audit-pool manifest")
    parser.add_argument(
        "--calibration-records",
        help="JSON-lines validation manifest for per-detector threshold calibration",
    )
    parser.add_argument("--tau", type=float, help="fixed operating point; skips calibration")
    parser.add_argument(
        "--detector",
        action="append",
        required=True,
        metavar="NAME=MODULE:ATTR",
        help="detector to audit; repeatable",
    )
    parser.add_argument("--out", default="out/audit", help="output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.tau is None and not args.calibration_records:
        raise SystemExit("provide either --tau or --calibration-records")
    records = load_manifest(args.records)
    calibration = load_manifest(args.calibration_records) if args.calibration_records else None
    detectors = dict(load_detector(spec) for spec in args.detector)
    table, factors = audit_detectors(
        detectors, records, tau=args.tau, calibration_records=calibration
    )
    export_audit(table, factors, args.out)
    columns = ("clean_auc", "v0_fpr", "cps", "esi", "cgs", "cfi")
    width = max(len(name) for name in table)
    print(f"{'detector':<{width}}  " + "  ".join(f"{c:>9}" for c in columns))
    for name, row in table.items():
        cells = ("      n/a" if row[c] is None else f"{row[c]:>9.3f}" for c in columns)
        print(f"{name:<{width}}  " + "  ".join(cells))
    print(f"\nwrote audit outputs to {args.out}")


if __name__ == "__main__":
    main()
