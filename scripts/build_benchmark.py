"""Build the quartet benchmark from a manifest of real source clips.

For every source, up to six feasible edit tuples are sampled; each tuple is
expanded into a candidate quartet by the two-phase generation pipeline and
retained only when the reference, objective, and panel gates all pass.
Held-out compositions are selected once per run by stratified sampling,
stored next to the accepted records, blocked on training and validation
sources, and drawn on test sources by replacing one or two regular
candidates within the per-source budget.

The source manifest is a JSON list of objects with keys ``source_id``,
``dataset``, ``split``, ``path``, ``environment`` (illumination, weather,
road_surface, capture_quality of the real footage), and optional ``flags``
carrying feasibility measurements such as ``participant_scale`` and
``visible_road_fraction``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from carve.core import AccidentSpec, EnvironmentSpec
from carve.data.quartets import save_manifest
from carve.factors import (
    HELD_OUT_PER_TEST_SOURCE,
    WET_PAVEMENT_CAUSES,
    EditContext,
    feasible_compositions,
    held_out_keys,
    sample_held_out_compositions,
)
from carve.gates.objective import ObjectiveGateConfig, calibrate_gamma_f
from carve.gates.panel import PanelRubric
from carve.gates.probe import load_probe
from carve.generation.providers import (
    HttpImageEditBackend,
    HttpVideoEditBackend,
    judge_from_config,
)
from carve.generation.video_edit import Backends, GateSuite, SourceUnit, generate_quartet

Composition = tuple[EnvironmentSpec, AccidentSpec]


def edit_context(source: SourceUnit) -> EditContext:
    """Feasibility context for one source.

    Conditionally allowed factor combinations are made feasible by declaring
    their cause in the script (the matching note is attached to the
    instruction later); source-dependent measurements come from the manifest.
    """
    return EditContext(
        wet_pavement_cause=WET_PAVEMENT_CAUSES[0],
        light_rain_onset=True,
        participant_scale=source.flags.get("participant_scale"),
        visible_road_fraction=source.flags.get("visible_road_fraction"),
    )


def script_note(space: dict, env: EnvironmentSpec) -> str | None:
    """Feasibility note that the edit script must carry, when required."""
    if env.weather == "clear" and env.road_surface == "wet":
        return space["script_notes"]["wet_pavement_cause"].strip()
    if env.weather == "rain" and env.road_surface == "dry":
        return space["script_notes"]["transitional_rain_onset"].strip()
    return None


def sample_source_tuples(
    source: SourceUnit,
    held_out: list[Composition],
    blocked: frozenset[str],
    cap: int,
    rng: np.random.Generator,
) -> list[tuple[Composition, bool]]:
    """Sample the per-source edit tuples under the candidate cap.

    Training and validation sources never draw held-out compositions; test
    sources replace one or two regular candidates with held-out draws while
    keeping the per-source budget unchanged.
    """
    regular = feasible_compositions(edit_context(source), blocked=blocked)
    picks = rng.permutation(len(regular))[:cap]
    tuples: list[tuple[Composition, bool]] = [(regular[int(i)], False) for i in picks]
    if source.split != "test" or not held_out:
        return tuples
    draws = int(rng.integers(HELD_OUT_PER_TEST_SOURCE[0], HELD_OUT_PER_TEST_SOURCE[1] + 1))
    draws = min(draws, len(held_out), len(tuples))
    chosen = rng.permutation(len(held_out))[:draws]
    return [(held_out[int(i)], True) for i in chosen] + tuples[: cap - draws]


def build_gate_suite(config: dict, config_path: str, args: argparse.Namespace) -> GateSuite:
    objective = ObjectiveGateConfig.from_yaml(config_path)
    gamma_f = args.gamma_f if args.gamma_f is not None else objective.gamma_f
    if gamma_f is None and args.calibration_clips:
        clips = sorted(Path(args.calibration_clips).glob("*.mp4"))
        gamma_f = calibrate_gamma_f(
            clips, objective.gamma_f_percentile, objective.analysis_resolution
        )
        print(f"calibrated gamma_f={gamma_f:.4f} on {len(clips)} held-out real clips")
    if gamma_f is None:
        raise SystemExit("gamma_f unset: pass --gamma-f or --calibration-clips")
    judges = [judge_from_config(spec) for spec in config["panel"]["judges"]]
    probe_path = config["gates"]["probe"].get("checkpoint")
    return GateSuite(
        objective=objective,
        gamma_f=float(gamma_f),
        rubric=PanelRubric.from_yaml(args.rubric),
        panel_judges=judges,
        reference_judge=judges[0],
        probe=load_probe(probe_path) if probe_path else None,
    )


def build_backends(config: dict) -> Backends:
    image_cfg = config["generation"]["image_backend"]
    video_cfg = config["generation"]["video_backend"]
    return Backends(
        image=HttpImageEditBackend(
            endpoint=image_cfg["endpoint"], model=image_cfg["model"],
            api_key_env=image_cfg.get("api_key_env"),
        ),
        video=HttpVideoEditBackend(
            endpoint=video_cfg["endpoint"], model=video_cfg["model"],
            api_key_env=video_cfg.get("api_key_env"),
        ),
    )


def load_sources(path: str | Path) -> list[SourceUnit]:
    with open(path, "r", encoding="utf-8") as handle:
        items = json.load(handle)
    return [
        SourceUnit(
            source_id=item["source_id"],
            dataset=item["dataset"],
            split=item["split"],
            path=item["path"],
            environment=EnvironmentSpec(**item["environment"]),
            flags=dict(item.get("flags", {})),
        )
        for item in items
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sources", required=True, help="source manifest JSON")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--factors", default="configs/factors.yaml")
    parser.add_argument("--rubric", default="configs/panel_rubric.yaml")
    parser.add_argument("--out", default="out/benchmark")
    parser.add_argument("--gamma-f", type=float, help="calibrated flow-jump threshold")
    parser.add_argument(
        "--calibration-clips",
        help="directory of real clips held out from generation, for gamma_f calibration",
    )
    parser.add_argument("--seed", type=int, help="overrides the held-out seed in factors.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    with open(args.factors, "r", encoding="utf-8") as handle:
        space = yaml.safe_load(handle)
    seed = args.seed if args.seed is not None else int(space["held_out"]["seed"])
    rng = np.random.default_rng(seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    held_out = sample_held_out_compositions(rng, int(space["held_out"]["count"]))
    blocked = held_out_keys(held_out)
    with open(out / "held_out_compositions.json", "w", encoding="utf-8") as handle:
        json.dump(sorted(blocked), handle, indent=2)

    sources = load_sources(args.sources)
    gates = build_gate_suite(config, args.config, args)
    backends = build_backends(config)
    cap = int(config["generation"]["per_source_candidate_cap"])
    fps = int(config["clip"]["fps"])
    key_frames = int(config["generation"]["key_frame_count"])

    accepted, rejections = [], []
    for source in sources:
        for index, ((env, accident), is_held_out) in enumerate(
            sample_source_tuples(source, held_out, blocked, cap, rng)
        ):
            record = generate_quartet(
                source,
                env,
                accident,
                backends,
                gates,
                out_dir=out / "media",
                quartet_id=f"{source.source_id}-{index:02d}",
                held_out=is_held_out,
                script_note=script_note(space, env),
                fps=fps,
                key_frame_count=key_frames,
                rejection_log=rejections,
            )
            if record is not None:
                accepted.append(record)

    save_manifest(accepted, out / "quartets.jsonl")
    for split in sorted({record.split for record in accepted}):
        save_manifest(
            [r for r in accepted if r.split == split], out / f"quartets_{split}.jsonl"
        )
    with open(out / "rejections.json", "w", encoding="utf-8") as handle:
        json.dump(rejections, handle, indent=2)
    stages = {stage: sum(1 for r in rejections if r["stage"] == stage)
              for stage in ("reference", "objective", "panel")}
    total = len(accepted) + len(rejections)
    print(f"accepted {len(accepted)}/{total} candidate tuples")
    print(f"rejections by stage: {stages}")
    print(f"held-out compositions reserved for test sources: {len(held_out)}")


if __name__ == "__main__":
    main()
