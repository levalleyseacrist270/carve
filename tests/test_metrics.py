"""Synthetic-score sanity tests for the audit metrics, FSS, and the sampler.

All tests run on mocked detector scores written directly onto quartet
records; no video, torch, or generation backend is involved.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from carve.core import AccidentSpec, EnvironmentSpec, QuartetRecord
from carve.factors import (
    ACCIDENT_TYPE_LEVELS,
    ENVIRONMENT_LEVELS,
    EditContext,
    LEVEL_INDEX,
    check_feasibility,
    composition_key,
    feasible_compositions,
    held_out_keys,
    levels_of_environment,
    sample_held_out_compositions,
)
from carve.metrics import (
    LevelDiagnostics,
    audit_quartets,
    auc_score,
    empirical_bayes_shrink,
    factor_stress_scores,
    per_level_diagnostics,
    select_threshold,
    threshold_metrics,
)
from carve.training.sampler import (
    RoundBudget,
    WeakFactorSampler,
    sampling_probabilities,
)

TAU = 0.5

DAY_ENV = EnvironmentSpec("day", "clear", "dry", "clean")
NIGHT_ENV = EnvironmentSpec("night", "rain", "wet", "compressed")
FOG_ENV = EnvironmentSpec("dusk", "fog", "wet", "clean")
REAR_END = AccidentSpec("rear_end", "moderate", "middle")
MINOR = AccidentSpec("side_impact", "minor", "late")
SEVERE = AccidentSpec("t_bone", "severe", "early")


def quartet(
    qid: str,
    p0: float,
    pe: float,
    pa: float,
    pae: float,
    env: EnvironmentSpec = DAY_ENV,
    accident: AccidentSpec = REAR_END,
    held_out: bool = False,
) -> QuartetRecord:
    return QuartetRecord(
        quartet_id=qid,
        source_id=f"src-{qid}",
        dataset="tad",
        split="val",
        env=env,
        accident=accident,
        held_out_composition=held_out,
        scores={"V0": p0, "VE": pe, "VA": pa, "VAE": pae},
    )


# ---------------------------------------------------------------- pool audit


def test_ideal_detector_audit() -> None:
    records = [quartet(f"q{i}", 0.0, 0.0, 1.0, 1.0) for i in range(4)]
    records += [quartet(f"h{i}", 0.0, 0.0, 1.0, 1.0, held_out=True) for i in range(2)]
    summary = audit_quartets(records, TAU)
    assert summary.cps == 1.0
    assert summary.v0_fpr == 0.0
    assert summary.nesi == 0.0
    assert summary.esi == 0.0
    assert summary.cfi == 1.0
    assert summary.pmr == 0.0
    assert summary.cgs.auc_seen == 1.0
    assert summary.cgs.auc_unseen == 1.0
    assert summary.cgs.ratio == 1.0
    assert summary.cgs.ratio_percent == 100.0
    assert summary.cgs.gap == 0.0


def test_conservative_detector_high_cps_but_no_faithfulness() -> None:
    records = [quartet(f"q{i}", 0.05, 0.05, 0.05, 0.05) for i in range(6)]
    summary = audit_quartets(records, TAU)
    assert summary.cps == 1.0
    assert summary.v0_fpr == 0.0
    assert summary.cfi == 0.0
    assert summary.pmr == 1.0


def test_nuisance_sensitive_detector() -> None:
    records = [quartet(f"q{i}", 0.10, 0.62, 0.9, 0.9) for i in range(5)]
    summary = audit_quartets(records, TAU)
    assert summary.cps == pytest.approx(0.0)
    assert summary.nesi == pytest.approx(0.52)
    assert summary.v0_fpr == 0.0


def test_exact_drift_and_faithfulness_values() -> None:
    records = [
        quartet("a", 0.1, 0.3, 0.9, 0.7),
        quartet("b", 0.2, 0.1, 0.6, 0.8),
    ]
    summary = audit_quartets(records, TAU)
    assert summary.nesi == pytest.approx((0.2 + 0.1) / 2)
    assert summary.esi == pytest.approx((0.2 + 0.2) / 2)
    expected_cfi = ((0.9 - 0.1 + 0.7 - 0.3) + (0.6 - 0.2 + 0.8 - 0.1)) / 4
    assert summary.cfi == pytest.approx(expected_cfi)


def test_threshold_boundary_semantics() -> None:
    # Scores exactly at tau: not a false alarm (strict >), but a positive miss (<=).
    records = [quartet("a", TAU, TAU, TAU, TAU)]
    summary = audit_quartets(records, TAU)
    assert summary.cps == 1.0
    assert summary.v0_fpr == 0.0
    assert summary.pmr == 1.0


def test_missing_branch_score_raises() -> None:
    record = quartet("a", 0.1, 0.1, 0.9, 0.9)
    del record.scores["VA"]
    with pytest.raises(KeyError):
        audit_quartets([record], TAU)


# ----------------------------------------------------------------------- AUC


def test_auc_score_cases() -> None:
    assert auc_score([0.1, 0.2], [0.8, 0.9]) == 1.0
    assert auc_score([0.8, 0.9], [0.1, 0.2]) == 0.0
    assert auc_score([0.5, 0.5], [0.5, 0.5]) == 0.5
    assert auc_score([], [0.5]) is None
    assert auc_score([0.5], []) is None


def test_cgs_pools_and_empty_guard() -> None:
    seen = [quartet(f"s{i}", 0.0, 0.0, 1.0, 1.0) for i in range(3)]
    unseen = [quartet(f"u{i}", 0.5, 0.5, 0.5, 0.5, held_out=True) for i in range(2)]
    summary = audit_quartets(seen + unseen, TAU)
    assert summary.cgs.auc_seen == 1.0
    assert summary.cgs.auc_unseen == 0.5
    assert summary.cgs.ratio == pytest.approx(0.5)
    assert summary.cgs.gap == pytest.approx(0.5)

    no_unseen = audit_quartets(seen, TAU)
    assert no_unseen.cgs.auc_unseen is None
    assert no_unseen.cgs.ratio is None
    assert no_unseen.cgs.gap is None


# ------------------------------------------------------- per-level diagnostics


def test_per_level_slicing_without_shrinkage() -> None:
    night = [quartet(f"n{i}", 0.1, 0.9, 0.9, 0.9, env=NIGHT_ENV, accident=MINOR) for i in range(4)]
    day = [quartet(f"d{i}", 0.1, 0.1, 0.2, 0.2, env=DAY_ENV, accident=SEVERE) for i in range(4)]
    diag = per_level_diagnostics(night + day, TAU, shrinkage_strength=0.0)

    assert diag["illumination=night"].cps == pytest.approx(0.0)
    assert diag["illumination=day"].cps == pytest.approx(1.0)
    assert diag["illumination=night"].nesi == pytest.approx(0.8)
    assert diag["illumination=night"].pmr is None
    assert diag["severity=minor"].pmr == pytest.approx(0.0)
    assert diag["severity=severe"].pmr == pytest.approx(1.0)
    assert diag["severity=minor"].cps is None
    assert diag["severity=minor"].esi is not None
    assert diag["illumination=night"].esi is not None
    assert diag["illumination=night"].cgs_gap is None  # no held-out compositions


def test_per_level_cgs_gap_from_held_out_compositions() -> None:
    seen = [quartet(f"s{i}", 0.0, 0.0, 1.0, 1.0) for i in range(4)]
    unseen = [
        quartet(f"u{i}", 0.5, 0.5, 0.5, 0.5, env=FOG_ENV, accident=SEVERE, held_out=True)
        for i in range(2)
    ]
    diag = per_level_diagnostics(seen + unseen, TAU, shrinkage_strength=0.0)
    assert diag["weather=fog"].cgs_gap == pytest.approx(0.5)
    assert diag["accident_type=t_bone"].cgs_gap == pytest.approx(0.5)
    assert diag["illumination=day"].cgs_gap is None


def test_shrinkage_weight_and_monotonicity() -> None:
    assert empirical_bayes_shrink(1.0, 8, 0.0, 8.0) == pytest.approx(0.5)
    assert empirical_bayes_shrink(1.0, 100, 0.0, 0.0) == 1.0
    assert empirical_bayes_shrink(1.0, 0, 0.3, 8.0) == pytest.approx(0.3)
    shrunk = [empirical_bayes_shrink(1.0, n, 0.0, 8.0) for n in (1, 4, 16, 64, 256)]
    assert all(a < b for a, b in zip(shrunk, shrunk[1:]))  # closer to raw as n grows
    by_strength = [empirical_bayes_shrink(1.0, 8, 0.0, s) for s in (0.0, 4.0, 8.0, 32.0)]
    assert all(a > b for a, b in zip(by_strength, by_strength[1:]))  # toward pooled


def test_shrinkage_pulls_slice_toward_pool() -> None:
    night = [quartet(f"n{i}", 0.1, 0.9, 0.9, 0.9, env=NIGHT_ENV) for i in range(4)]
    day = [quartet(f"d{i}", 0.1, 0.1, 0.9, 0.9, env=DAY_ENV) for i in range(4)]
    raw = per_level_diagnostics(night + day, TAU, shrinkage_strength=0.0)
    shrunk = per_level_diagnostics(night + day, TAU, shrinkage_strength=8.0)
    pooled_false_alarm = 0.5  # 4 of 8 VE scores above tau
    assert raw["illumination=night"].cps == pytest.approx(0.0)
    expected = 1.0 - empirical_bayes_shrink(1.0, 4, pooled_false_alarm, 8.0)
    assert shrunk["illumination=night"].cps == pytest.approx(expected)


# ----------------------------------------------------------------------- FSS


def _diag(key: str, n: int = 10, **stats: float) -> LevelDiagnostics:
    return LevelDiagnostics(level=LEVEL_INDEX[key], n=n, **stats)


def test_fss_env_level_renormalizes_without_pmr() -> None:
    diagnostics = {
        "illumination=night": _diag(
            "illumination=night", cps=0.7, nesi=0.2, esi=0.25, cgs_gap=0.05
        ),
        "illumination=day": _diag(
            "illumination=day", cps=0.9, nesi=0.1, esi=0.15, cgs_gap=0.01
        ),
    }
    stress = factor_stress_scores(diagnostics)
    # night is the maximum of every defined term; the missing PMR weight is
    # renormalized away, so its stress is exactly 1.
    assert stress["illumination=night"] == pytest.approx(1.0)
    assert stress["illumination=day"] == pytest.approx(0.0)


def test_fss_accident_level_weight_renormalization() -> None:
    diagnostics = {
        "severity=minor": _diag("severity=minor", esi=0.3, pmr=0.1),
        "severity=severe": _diag("severity=severe", esi=0.1, pmr=0.3),
    }
    stress = factor_stress_scores(diagnostics)
    # Defined terms are ESI (0.20) and PMR (0.25); weights renormalize to
    # sum to one within the 0.45 total.
    assert stress["severity=minor"] == pytest.approx(0.20 / 0.45)
    assert stress["severity=severe"] == pytest.approx(0.25 / 0.45)


def test_fss_esi_normalized_across_groups() -> None:
    diagnostics = {
        "illumination=night": _diag("illumination=night", cps=0.8, nesi=0.1, esi=0.2),
        "severity=minor": _diag("severity=minor", esi=0.3, pmr=0.2),
        "severity=severe": _diag("severity=severe", esi=0.1, pmr=0.2),
    }
    stress = factor_stress_scores(diagnostics)
    # ESI is normalized over all three levels, so the environment level sits
    # mid-scale; its CPS and NESI terms are constant and normalize to zero.
    assert stress["illumination=night"] == pytest.approx((0.20 * 0.5) / 0.60)


def test_fss_stays_in_unit_interval() -> None:
    rng = np.random.default_rng(3)
    diagnostics = {}
    for lvl in ENVIRONMENT_LEVELS:
        diagnostics[lvl.key] = _diag(
            lvl.key,
            cps=float(rng.uniform(0.5, 1.0)),
            nesi=float(rng.uniform(0.0, 0.3)),
            esi=float(rng.uniform(0.0, 0.3)),
        )
    stress = factor_stress_scores(diagnostics)
    assert all(0.0 <= value <= 1.0 for value in stress.values())


# ------------------------------------------------------------------- sampler


def test_sampling_probabilities_power_law() -> None:
    stress = {"a": 0.8, "b": 0.4, "c": 0.2}
    probs = sampling_probabilities(stress, alpha=1.5)
    powered = {k: v**1.5 for k, v in stress.items()}
    total = sum(powered.values())
    for key, value in powered.items():
        assert probs[key] == pytest.approx(value / total)
    assert sum(probs.values()) == pytest.approx(1.0)


def test_sampling_probabilities_zero_stress_uniform() -> None:
    probs = sampling_probabilities({"a": 0.0, "b": 0.0}, alpha=1.5)
    assert probs["a"] == pytest.approx(0.5)
    assert probs["b"] == pytest.approx(0.5)


def test_weak_factor_sampler_matches_distribution() -> None:
    stress = {
        "illumination=night": 0.84,
        "severity=minor": 0.77,
        "weather=fog": 0.64,
        "capture_quality=compressed": 0.48,
    }
    sampler = WeakFactorSampler(stress, alpha=1.5)
    expected = sampling_probabilities(stress, alpha=1.5)
    rng = np.random.default_rng(7)
    draws = 60_000
    counts: dict[str, int] = {key: 0 for key in stress}
    for _ in range(draws):
        counts[sampler.sample_level(rng)] += 1
    for key, prob in expected.items():
        assert counts[key] / draws == pytest.approx(prob, abs=0.01)


def test_weak_factor_sampler_targets_level() -> None:
    sampler = WeakFactorSampler({"illumination=night": 1.0})
    rng = np.random.default_rng(11)
    for env, accident in sampler.sample(rng, 200):
        assert env.illumination == "night"
        assert check_feasibility(env, accident)[0]


def test_uniform_baseline_over_feasible_set() -> None:
    sampler = WeakFactorSampler({"illumination=night": 1.0}, uniform=True)
    rng = np.random.default_rng(13)
    pool_keys = {composition_key(env, acc) for env, acc in sampler.pool}
    counts = {t: 0 for t in ACCIDENT_TYPE_LEVELS}
    draws = 12_000
    for env, accident in sampler.sample(rng, draws):
        assert composition_key(env, accident) in pool_keys
        counts[accident.accident_type] += 1
    for accident_type in ACCIDENT_TYPE_LEVELS:
        assert counts[accident_type] / draws == pytest.approx(1 / 6, abs=0.02)


def test_sampler_blocks_compositions() -> None:
    blocked = composition_key(NIGHT_ENV, AccidentSpec("rear_end", "moderate", "late"))
    sampler = WeakFactorSampler({"illumination=night": 1.0}, blocked=[blocked])
    keys = {composition_key(env, acc) for env, acc in sampler.pool}
    assert blocked not in keys
    assert len(sampler.pool) == len(feasible_compositions()) - 1


def test_feasible_pool_size_under_contexts() -> None:
    assert len(feasible_compositions()) == 1944
    with_cause = EditContext(wet_pavement_cause="recent_rain")
    assert len(feasible_compositions(with_cause)) == 2430
    both = EditContext(wet_pavement_cause="recent_rain", light_rain_onset=True)
    assert len(feasible_compositions(both)) == 2916


# -------------------------------------------------------------------- budget


def test_round_budget_realized_edit_counts() -> None:
    # 25% of the real training clips, counted in generated edit entries and
    # reached in whole quartets of three entries.
    for real_clips, edit_target, quartets, realized in (
        (283, 71, 24, 72),
        (682, 171, 57, 171),
        (5360, 1340, 447, 1341),
    ):
        budget = RoundBudget(real_clips)
        assert budget.edit_target == edit_target
        assert budget.quartet_target == quartets
        accepted = 0
        while not budget.reached(accepted):
            accepted += 1
        assert accepted == quartets
        assert accepted * 3 == realized
        assert budget.candidate_target == math.ceil(quartets * (1 + 1 / 3))


# ---------------------------------------------------------------- threshold


def test_select_threshold_maximizes_f1() -> None:
    scores = [0.1, 0.2, 0.6, 0.8]
    labels = [0, 0, 1, 1]
    tau = select_threshold(scores, labels)
    assert tau == pytest.approx(0.4)
    assert threshold_metrics(scores, labels, tau)["f1"] == pytest.approx(1.0)


def test_select_threshold_requires_both_classes() -> None:
    with pytest.raises(ValueError):
        select_threshold([0.2, 0.8], [1, 1])


# ------------------------------------------------------- held-out compositions


def test_held_out_compositions_cover_types_and_env_levels() -> None:
    comps = sample_held_out_compositions(np.random.default_rng(0))
    assert len(comps) == 18
    types = {acc.accident_type for _, acc in comps}
    assert types == set(ACCIDENT_TYPE_LEVELS)
    env_keys = {lvl.key for env, _ in comps for lvl in levels_of_environment(env)}
    assert env_keys == {lvl.key for lvl in ENVIRONMENT_LEVELS}
    for env, accident in comps:
        assert check_feasibility(env, accident)[0]
    again = sample_held_out_compositions(np.random.default_rng(0))
    assert held_out_keys(comps) == held_out_keys(again)


# ---------------------------------------------------------------- feasibility


def test_feasibility_rules_and_reasons() -> None:
    clear_wet = EnvironmentSpec("day", "clear", "wet", "clean")
    ok, reason = check_feasibility(clear_wet, REAR_END)
    assert not ok and "wet-pavement cause" in reason
    ok, _ = check_feasibility(
        clear_wet, REAR_END, EditContext(wet_pavement_cause="standing_water")
    )
    assert ok

    rain_dry = EnvironmentSpec("day", "rain", "dry", "clean")
    ok, reason = check_feasibility(rain_dry, REAR_END)
    assert not ok and "light-rain" in reason
    ok, _ = check_feasibility(rain_dry, REAR_END, EditContext(light_rain_onset=True))
    assert ok

    rollover = AccidentSpec("rollover", "severe", "middle")
    ok, _ = check_feasibility(DAY_ENV, rollover)
    assert ok  # source measurements unknown: rule defers
    ok, reason = check_feasibility(
        DAY_ENV, rollover, EditContext(participant_scale=0.01, visible_road_fraction=0.5)
    )
    assert not ok and "scale" in reason
    ok, reason = check_feasibility(
        DAY_ENV, rollover, EditContext(participant_scale=0.2, visible_road_fraction=0.05)
    )
    assert not ok and "road space" in reason
    ok, _ = check_feasibility(
        DAY_ENV, rollover, EditContext(participant_scale=0.2, visible_road_fraction=0.5)
    )
    assert ok


# ------------------------------------------------ runtime configuration keys


def test_configuration_keys_change_effective_settings() -> None:
    """Shared configuration blocks must reach the loop, losses, and clips."""
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "_hardening_cli", root / "scripts" / "run_hardening.py"
    )
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    parsed = {
        "losses": {"lambda_ep": 0.6, "lambda_ic": 0.9, "lambda_fm": 0.3, "delta": 0.35},
        "sampler": {"alpha": 2.0, "eta": [0.30, 0.10, 0.20, 0.25, 0.15], "shrinkage": 4.0},
        "training": {
            "lr": 1.0e-4,
            "weight_decay": 0.01,
            "batch_size": 16,
            "epochs": 12,
            "rounds": 2,
            "budget_fraction": 0.5,
        },
        "generation": {"oversample_factor": 1.5},
        "clip": {"onset_window": [26, 38]},
    }
    loop_kwargs, loss_kwargs, onset_range = cli._merge_loop_settings(parsed)
    assert loop_kwargs["alpha"] == 2.0
    assert loop_kwargs["shrinkage"] == 4.0
    assert loop_kwargs["fss_weights"] == (0.30, 0.10, 0.20, 0.25, 0.15)
    assert loop_kwargs["learning_rate"] == pytest.approx(1.0e-4)
    assert loop_kwargs["weight_decay"] == pytest.approx(0.01)
    assert loop_kwargs["batch_size"] == 16
    assert loop_kwargs["pretrain_epochs"] == 12
    assert loop_kwargs["rounds"] == 2
    assert loop_kwargs["budget_fraction"] == 0.5
    assert loop_kwargs["oversample"] == pytest.approx(0.5)  # multiplier minus one
    assert loss_kwargs == {
        "environment_purity": 0.6,
        "intervention_consistency": 0.9,
        "faithfulness_margin": 0.3,
        "margin": 0.35,
    }
    assert onset_range == (26, 38)

    # Absent keys leave the built-in defaults untouched, and an explicit
    # loop block wins over the shared blocks.
    assert cli._merge_loop_settings({}) == ({}, {}, None)
    winner, _, _ = cli._merge_loop_settings(
        {"sampler": {"alpha": 2.0}, "loop": {"alpha": 3.0}}
    )
    assert winner["alpha"] == 3.0

    # The alpha override visibly changes the sampling distribution.
    stress = {"illumination=night": 0.8, "weather=fog": 0.4}
    default_probs = sampling_probabilities(stress)
    overridden = sampling_probabilities(stress, alpha=loop_kwargs["alpha"])
    assert overridden["illumination=night"] > default_probs["illumination=night"]

    # The onset band override moves the clip-window placement.
    from carve.data.clips import positive_window

    default_window = positive_window(100, 1000)
    custom_window = positive_window(100, 1000, onset_range=(30, 30))
    assert default_window.onset_in_clip == 32
    assert custom_window.onset_in_clip == 30
    with pytest.raises(ValueError):
        positive_window(100, 1000, onset_range=(40, 24))
