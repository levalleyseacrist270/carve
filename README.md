# CARVE

Counterfactual audit and hardening for CCTV traffic accident detectors.

Roadside accident detectors are usually evaluated on observational splits in
which accident labels co-occur with illumination, weather, road surface, and
compression conditions. High accuracy on such splits does not show whether a
detector responds to accident evidence or to nuisance conditions correlated
with accidents. CARVE (Counterfactual Accident Robustness via Video Editing)
answers that question with matched video quartets built from a single real
source clip:

| Branch | Content | Label |
|--------|---------|-------|
| `V0`   | real source clip, no accident | 0 |
| `VE`   | environment factors edited, no accident | 0 |
| `VA`   | scripted accident inserted, source environment | 1 |
| `VAE`  | scripted accident under the edited environment | 1 |

Every edit preserves camera pose, road geometry, lane count, and the
trajectories of vehicles outside the scripted collision. On this structure
the toolkit computes factor-resolved robustness metrics, and a closed
hardening loop routes new generation budget toward the factors that currently
break the detector.

## What is included

- **Two-phase generation orchestration.** Phase one turns source key frames,
  layout cues, and an environment script into a reference image; phase two
  performs reference-guided video editing for the `VE`, `VA`, and `VAE`
  branches. The full structured instruction templates for both phases ship in
  `carve/generation/scripts_builder.py`, and all backends sit behind
  provider-agnostic interfaces.
- **Three-layer quality control.** A reference-image gate
  (layout / environment / no-accident / realism), five objective video gates
  (masked DINO similarity, motion-histogram distance, displacement outside the
  edit, flow-jump rate, capture-quality probe), and a three-judge VLM panel
  with a fixed seven-dimension rubric, temperature-zero JSON protocol, and a
  median/IQR acceptance rule.
- **Audit metric suite.** CPS, NESI, ESI, CFI, and CGS as primary metrics,
  plus PMR and FSS as hardening diagnostics, with per-factor breakdowns.
- **Closed-loop hardening.** PMR-FSS weak-factor sampling with
  empirical-Bayes shrinkage and the CGAA-IC training loop with
  environment-purity, intervention-consistency, and faithfulness-margin
  losses. The deployed detector keeps its inference graph unchanged.
- **Evaluation harness.** A multi-detector audit runner producing the audit
  table and per-factor diagnostics as CSV and JSON, and split definitions
  handled at the source-unit level.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Python 3.10 or newer is required. Core dependencies: numpy, opencv-python,
torch, transformers, pyyaml, requests. VLM judges and generation backends are
reached over HTTP behind small adapters; no proprietary SDK is required.

## Package layout

```
carve/
  core.py                  shared types: Branch, EnvironmentSpec, AccidentSpec,
                           QuartetRecord, GateResult
  factors.py               factor space, feasibility mask, held-out sampling
  metrics.py               CPS/NESI/ESI/CFI/CGS, PMR/FSS, threshold selection
  audit.py                 multi-detector audit table and per-factor export
  gates/
    reference.py           reference-image gate (four yes/no checks)
    objective.py           DINO / MHD / displacement / flow-jump / capture gates
    probe.py               capture-quality 3-class probe (train + predict)
    panel.py               three-judge panel: prompt, parsing, acceptance rule
  generation/
    scripts_builder.py     structured scripts and instruction templates
    layout_cues.py         lane masks, boundaries, boxes, perspective cues
    reference_image.py     phase-one orchestration
    video_edit.py          phase-two orchestration and quartet assembly
    providers.py           backend and judge interfaces with HTTP adapters
  data/                    clip protocol, quartet datasets, split handling
  models/                  detector wrapper with an unchanged inference graph
  training/                PMR-FSS sampler, quartet losses, hardening loop
scripts/
  build_benchmark.py       manifest of sources -> gated quartet records
  run_audit.py             quartet records + detectors -> audit outputs
  run_hardening.py         config-driven CGAA-IC loop entry point
  calibrate_threshold.py   operating-point sweep on validation scores
configs/
  default.yaml             all thresholds and hyperparameters
  factors.yaml             factor space, feasibility rules, held-out policy
  panel_rubric.yaml        seven-dimension rubric and acceptance thresholds
tests/                     synthetic-score tests for metrics and sampling
demo/synthetic_track/      fully synthetic demonstration media (see README)
```

## Quickstart

**1. Calibrate the flow-jump threshold and train the capture probe.** Both
use real clips that never enter generation:

```python
from carve.gates import calibrate_gamma_f, train_probe, save_probe

gamma_f = calibrate_gamma_f(held_out_real_clips, percentile=95)
probe = train_probe(labeled_clips)  # [(path, "clean"|"compressed"|"motion_blur")]
save_probe(probe, "probe.pt")
```

**2. Build the benchmark** from a source manifest (JSON list with
`source_id`, `dataset`, `split`, `path`, `environment`, optional `flags`),
with generator endpoints and panel judges set in `configs/default.yaml`:

```bash
python scripts/build_benchmark.py \
    --sources sources.json --gamma-f <calibrated value> --out out/benchmark
```

**3. Audit detectors** on the accepted quartets:

```bash
python scripts/run_audit.py \
    --records out/benchmark/quartets_test.jsonl \
    --calibration-records out/benchmark/quartets_val.jsonl \
    --detector videomae=my_models.detectors:videomae_b \
    --out out/audit
```

**4. Harden.** `carve.training.cgaa_ic.run_hardening` audits the detector on
validation quartets, samples weak factors with PMR-FSS, generates fresh gated
quartets from training-split sources through the same two-phase pipeline
(plug `carve.generation` into its generator and gate interfaces), and updates
the detector with the quartet losses configured in `configs/default.yaml`.
The deployed model keeps the base inference graph.

## Configuration reference

| Key | Meaning | Default |
|-----|---------|---------|
| `clip.fps` / `clip.frames` | decode rate and clip length | 16 / 64 |
| `clip.onset_window` | allowed impact-frame band for positives | [24, 40] |
| `clip.negative_middle_fraction` | middle fraction sampled for negatives | 0.8 |
| `analysis.resolution` | measurement resolution for objective gates | 224 |
| `gates.objective.dino_similarity_min` | masked DINO similarity floor | 0.92 |
| `gates.objective.mhd_max` | motion-histogram distance ceiling | 0.25 |
| `gates.objective.displacement_px_max` | non-edit displacement ceiling (px) | 4 |
| `gates.objective.fjr_max` | flow-jump rate ceiling | 0.12 |
| `gates.objective.gamma_f_percentile` | calibration percentile for gamma_f | 95 |
| `gates.probe.classes` | capture-quality classes | clean/compressed/motion_blur |
| `generation.per_source_candidate_cap` | candidate tuples per source | 6 |
| `generation.oversample_factor` | candidate oversampling in hardening rounds | 1.334 |
| `panel` (rubric file) | acceptance `min_median` / `max_iqr` | 4 / 1 |
| `losses.lambda_ep/ic/fm` | quartet loss weights | 0.5 / 1.0 / 0.25 |
| `losses.delta` | faithfulness margin | 0.4 |
| `sampler.alpha` | weak-factor concentration | 1.5 |
| `sampler.eta` | brittleness term weights | 0.25/0.15/0.20/0.25/0.15 |
| `sampler.shrinkage` | empirical-Bayes shrinkage strength | 8.0 |
| `training.*` | optimizer, schedule, rounds, budget | see file |

## Data and media

This release contains code, configuration, and a fully synthetic
demonstration track under `demo/synthetic_track/` (synthetic source clips,
environment reference frames, preview grids, and the first complete
quartets; see the README there). Benchmark media derived from real footage
are not distributed: the source datasets license their footage for research
use without redistribution of derived media, CCTV frames show road users who
gave no consent to further distribution, and synthetic crash footage anchored
to real scenes warrants access control. Users supply their own licensed
static-camera footage, generator endpoints, and VLM endpoints; the gates and
acceptance criteria in this repository define what enters a benchmark,
independently of the chosen backends.

## License

MIT. See `LICENSE`.

## Citation

Citation information will be added upon publication.
