#!/usr/bin/env python
"""Run the CGAA-IC hardening loop from a configuration file.

The configuration is JSON or YAML with the layout::

    detector:
      checkpoint: MCG-NJU/videomae-base
      num_frames: 64
      pretrained: true
    data:
      real_train: manifests/real_train.jsonl      # {"path", "label", ...} per line
      real_val: manifests/real_val.jsonl
      audit_quartets: manifests/benchmark_val.jsonl
      sources: manifests/sources.jsonl            # {"source_id", "dataset", "path"}
      blocked_compositions: manifests/held_out_keys.txt   # optional, one key per line
    loader: mypackage.video:load_clip             # path -> float tensor [T, C, H, W]
    backend:                                      # zero-argument factories
      reference_generator: mypackage.backend:build_reference_generator
      reference_gate: mypackage.backend:build_reference_gate
      quartet_generator: mypackage.backend:build_quartet_generator
      quartet_gate: mypackage.backend:build_quartet_gate
    loop:                                         # optional HardeningConfig overrides
      rounds: 3
      budget_fraction: 0.25
    output: runs/hardening

Dotted overrides are applied last, e.g. ``--set loop.rounds=2``.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Callable


def _ensure_package() -> None:
    try:
        importlib.import_module("carve")
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _load_config(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "reading YAML configuration requires the 'pyyaml' package; "
                "install it or supply JSON"
            ) from exc
        return yaml.safe_load(text)
    return json.loads(text)


def _apply_overrides(config: dict, overrides: list[str]) -> None:
    for item in overrides:
        key, _, raw = item.partition("=")
        if not _:
            raise ValueError(f"override {item!r} is not of the form key=value")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        node = config
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value


def _resolve(spec: str) -> Any:
    module_name, _, attr = spec.partition(":")
    if not attr:
        raise ValueError(f"expected 'module:attribute', got {spec!r}")
    return getattr(importlib.import_module(module_name), attr)


def _build_component(spec: str) -> Any:
    """Backend entries are zero-argument factories returning the component."""
    return _resolve(spec)()


def _load_blocked(path: str | None) -> list[str]:
    if not path:
        return []
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True, help="JSON or YAML configuration")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="dotted configuration override, repeatable",
    )
    parser.add_argument("--seed", type=int, default=0, help="seed for the run's numpy Generator")
    parser.add_argument(
        "--uniform-baseline",
        action="store_true",
        help="sample compositions uniformly from the feasible set (same budget)",
    )
    parser.add_argument("--output", type=Path, default=None, help="override the output directory")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _ensure_package()

    import numpy as np
    import torch

    from carve.data.quartets import load_labeled_clips, load_manifest, save_manifest
    from carve.models.detector import VideoMAEDetector
    from carve.training.cgaa_ic import HardeningConfig, SourceClip, run_hardening

    config = _load_config(args.config)
    _apply_overrides(config, args.overrides)
    if args.uniform_baseline:
        config.setdefault("loop", {})["uniform_sampling"] = True

    data_cfg = config["data"]
    real_train = load_labeled_clips(data_cfg["real_train"])
    real_val = load_labeled_clips(data_cfg["real_val"])
    audit_records = load_manifest(data_cfg["audit_quartets"])
    source_lines = [
        line
        for line in Path(data_cfg["sources"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    sources = [
        SourceClip(row["source_id"], row.get("dataset", ""), row["path"])
        for row in map(json.loads, source_lines)
    ]
    blocked = _load_blocked(data_cfg.get("blocked_compositions"))

    detector_cfg = config.get("detector", {})
    detector = VideoMAEDetector(
        checkpoint=detector_cfg.get("checkpoint", "MCG-NJU/videomae-base"),
        num_frames=int(detector_cfg.get("num_frames", 64)),
        pretrained=bool(detector_cfg.get("pretrained", True)),
    )
    clip_loader: Callable = _resolve(config["loader"])
    backend = config["backend"]
    loop_cfg = HardeningConfig(**config.get("loop", {}))
    rng = np.random.default_rng(args.seed)

    result = run_hardening(
        detector=detector,
        real_train=real_train,
        real_val=real_val,
        audit_records=audit_records,
        sources=sources,
        reference_generator=_build_component(backend["reference_generator"]),
        reference_gate=_build_component(backend["reference_gate"]),
        quartet_generator=_build_component(backend["quartet_generator"]),
        quartet_gate=_build_component(backend["quartet_gate"]),
        clip_loader=clip_loader,
        rng=rng,
        config=loop_cfg,
        blocked_compositions=blocked,
    )

    output = args.output or Path(config.get("output", "runs/hardening"))
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "seed": args.seed,
        "rounds": [report.as_dict() for report in result.rounds],
        "accepted_quartets": len(result.synthetic),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    save_manifest(result.synthetic, output / "synthetic_manifest.jsonl")
    torch.save(detector.state_dict(), output / "detector_final.pt")
    print(json.dumps(summary["rounds"][-1]["audit"] if summary["rounds"] else {}, indent=2))
    print(f"artifacts written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
