"""CGAA-IC: counterfactual generation, audit, and augmentation with
intervention consistency.

The closed loop hardens a detector in R rounds. Each round recalibrates the
operating threshold on the real validation split, audits the current
detector on the sampler validation pool (the source dataset's validation
quartets), converts per-factor diagnostics into PMR-FSS stress scores,
samples target factors, generates and gates fresh quartets from
training-split sources, and updates the detector on the real clips plus all
synthetic quartets accepted so far. Threshold selection and sampling never
touch target test clips or test quartets, and the deployed inference graph
is left unchanged.

Generation and gating are injected through small protocols so the loop stays
agnostic to the editing backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Optional, Protocol, Sequence, runtime_checkable

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..core import AccidentSpec, Branch, EnvironmentSpec, GateResult, QuartetRecord
from ..data.quartets import ClipDataset, ClipLoader, LabeledClip, QuartetDataset
from ..metrics import (
    DEFAULT_SHRINKAGE,
    AuditSummary,
    audit_quartets,
    factor_stress_scores,
    per_level_diagnostics,
    select_threshold,
)
from ..models.detector import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_WEIGHT_DECAY,
    AccidentDetector,
    configure_training,
)
from .losses import LossWeights, hardening_objective
from .sampler import (
    DEFAULT_ALPHA,
    DEFAULT_BUDGET_FRACTION,
    DEFAULT_OVERSAMPLE,
    EDITS_PER_QUARTET,
    RoundBudget,
    WeakFactorSampler,
)


@dataclass(frozen=True)
class SourceClip:
    """A real accident-free source clip eligible for counterfactual editing."""

    source_id: str
    dataset: str
    path: str


@runtime_checkable
class ReferenceGenerator(Protocol):
    """Produces an environment reference candidate for a source clip.

    The returned candidate is opaque to the loop; it is handed to the
    reference gate and, if accepted, to the quartet generator.
    """

    def generate(
        self, source: SourceClip, env: EnvironmentSpec, rng: np.random.Generator
    ) -> object: ...


@runtime_checkable
class QuartetGenerator(Protocol):
    """Produces a candidate quartet from a source clip and accepted reference."""

    def generate(
        self,
        source: SourceClip,
        reference: object,
        env: EnvironmentSpec,
        accident: AccidentSpec,
        rng: np.random.Generator,
    ) -> QuartetRecord: ...


@runtime_checkable
class QualityGate(Protocol):
    """Accept-or-reject decision for one candidate reference or quartet."""

    def evaluate(self, candidate: object) -> GateResult: ...


@dataclass
class HardeningConfig:
    """Configuration of the hardening loop; defaults follow the shipped setup."""

    rounds: int = 3
    pretrain_epochs: int = DEFAULT_EPOCHS
    round_epochs: int = 10
    batch_size: int = DEFAULT_BATCH_SIZE
    learning_rate: float = DEFAULT_LEARNING_RATE
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    budget_fraction: float = DEFAULT_BUDGET_FRACTION
    oversample: float = DEFAULT_OVERSAMPLE
    alpha: float = DEFAULT_ALPHA
    shrinkage: float = DEFAULT_SHRINKAGE
    uniform_sampling: bool = False
    loss_weights: LossWeights = field(default_factory=LossWeights)
    device: Optional[str] = None
    num_workers: int = 0


@dataclass
class RoundReport:
    """Audit and generation bookkeeping for one hardening round."""

    index: int
    tau: float
    audit: AuditSummary
    stress: dict[str, float]
    accepted_quartets: int
    attempted_candidates: int
    reference_rejected: int
    quartet_rejected: int

    @property
    def accepted_edits(self) -> int:
        return self.accepted_quartets * EDITS_PER_QUARTET

    def as_dict(self) -> dict:
        return {
            "round": self.index,
            "tau": self.tau,
            "audit": self.audit.as_dict(),
            "stress": dict(self.stress),
            "accepted_quartets": self.accepted_quartets,
            "accepted_edits": self.accepted_edits,
            "attempted_candidates": self.attempted_candidates,
            "reference_rejected": self.reference_rejected,
            "quartet_rejected": self.quartet_rejected,
        }


@dataclass
class HardeningResult:
    """Outcome of the full loop: per-round reports and accepted quartets."""

    rounds: list[RoundReport]
    synthetic: list[QuartetRecord]


def _resolve_device(config: HardeningConfig) -> torch.device:
    if config.device:
        return torch.device(config.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _loader(
    dataset, batch_size: int, rng: np.random.Generator, config: HardeningConfig,
    shuffle: bool,
) -> DataLoader:
    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(int(rng.integers(0, 2**63 - 1)))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=config.num_workers,
        drop_last=False,
    )


def _cycle(loader: DataLoader) -> Iterator:
    while True:
        for batch in loader:
            yield batch


def score_clips(
    detector: AccidentDetector,
    paths: Sequence[str],
    clip_loader: ClipLoader,
    device: torch.device,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> np.ndarray:
    """Score media paths with the detector in eval mode."""
    scores = np.empty(len(paths), dtype=float)
    for start in range(0, len(paths), batch_size):
        chunk = paths[start : start + batch_size]
        videos = torch.stack([clip_loader(p) for p in chunk]).to(device)
        probs = detector.predict(videos)
        scores[start : start + len(chunk)] = probs.detach().cpu().numpy()
    return scores


def score_quartet_records(
    detector: AccidentDetector,
    records: Sequence[QuartetRecord],
    clip_loader: ClipLoader,
    device: torch.device,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> None:
    """Fill ``record.scores`` for all four branches of every record."""
    jobs: list[tuple[QuartetRecord, str, str]] = []
    for record in records:
        for branch in Branch:
            jobs.append((record, branch.value, record.paths[branch.value]))
    scores = score_clips(
        detector, [path for _, _, path in jobs], clip_loader, device, batch_size
    )
    for (record, branch_name, _), score in zip(jobs, scores):
        record.scores[branch_name] = float(score)


def calibrate_operating_point(
    detector: AccidentDetector,
    validation: Sequence[LabeledClip],
    clip_loader: ClipLoader,
    device: torch.device,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> float:
    """Select tau on the real validation split by maximizing F1."""
    scores = score_clips(
        detector, [c.path for c in validation], clip_loader, device, batch_size
    )
    return select_threshold(scores, [c.label for c in validation])


def generate_round(
    sampler: WeakFactorSampler,
    budget: RoundBudget,
    sources: Sequence[SourceClip],
    reference_generator: ReferenceGenerator,
    reference_gate: QualityGate,
    quartet_generator: QuartetGenerator,
    quartet_gate: QualityGate,
    rng: np.random.Generator,
) -> tuple[list[QuartetRecord], dict[str, int]]:
    """Generate and gate candidates until the accepted budget is reached.

    At most ``budget.candidate_target`` tuples are attempted; the loop stops
    early once accepted edits cover the target. Candidates rejected at the
    reference stage never reach video generation.
    """
    accepted: list[QuartetRecord] = []
    stats = {"attempted": 0, "reference_rejected": 0, "quartet_rejected": 0}
    for _ in range(budget.candidate_target):
        if budget.reached(len(accepted)):
            break
        stats["attempted"] += 1
        env, accident = sampler.sample_composition(rng)
        source = sources[int(rng.integers(len(sources)))]
        reference = reference_generator.generate(source, env, rng)
        if not reference_gate.evaluate(reference).passed:
            stats["reference_rejected"] += 1
            continue
        candidate = quartet_generator.generate(source, reference, env, accident, rng)
        if not quartet_gate.evaluate(candidate).passed:
            stats["quartet_rejected"] += 1
            continue
        accepted.append(candidate)
    return accepted, stats


def _fit_supervised(
    detector: AccidentDetector,
    clips: Sequence[LabeledClip],
    clip_loader: ClipLoader,
    config: HardeningConfig,
    epochs: int,
    device: torch.device,
    rng: np.random.Generator,
) -> None:
    """Plain BCE training on real labeled clips."""
    if epochs <= 0:
        return
    loader = _loader(
        ClipDataset(clips, clip_loader), config.batch_size, rng, config, shuffle=True
    )
    optimizer, scheduler = configure_training(
        detector,
        steps_per_epoch=len(loader),
        epochs=epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    detector.train()
    for _ in range(epochs):
        for videos, labels in loader:
            probs = detector(videos.to(device))
            loss = hardening_objective(
                probs, labels.to(device), None, config.loss_weights
            ).total
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()


def _fit_hardening(
    detector: AccidentDetector,
    real_clips: Sequence[LabeledClip],
    synthetic: Sequence[QuartetRecord],
    clip_loader: ClipLoader,
    config: HardeningConfig,
    device: torch.device,
    rng: np.random.Generator,
) -> None:
    """One round's update on real clips plus all accepted synthetic quartets."""
    if not synthetic:
        _fit_supervised(
            detector, real_clips, clip_loader, config, config.round_epochs, device, rng
        )
        return
    real_loader = _loader(
        ClipDataset(real_clips, clip_loader), config.batch_size, rng, config, shuffle=True
    )
    # A quartet batch holds four clips per record; shrink it to keep the
    # per-step clip count close to the configured batch size.
    quartet_batch = max(1, config.batch_size // len(Branch))
    quartet_loader = _loader(
        QuartetDataset(synthetic, clip_loader), quartet_batch, rng, config, shuffle=True
    )
    steps_per_epoch = max(len(real_loader), len(quartet_loader))
    optimizer, scheduler = configure_training(
        detector,
        steps_per_epoch=steps_per_epoch,
        epochs=config.round_epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    real_iter, quartet_iter = _cycle(real_loader), _cycle(quartet_loader)
    detector.train()
    for _ in range(config.round_epochs):
        for _ in range(steps_per_epoch):
            videos, labels = next(real_iter)
            batch = next(quartet_iter)
            real_probs = detector(videos.to(device))
            quartet_probs = tuple(
                detector(batch[branch.value].to(device)) for branch in Branch
            )
            loss = hardening_objective(
                real_probs, labels.to(device), quartet_probs, config.loss_weights
            ).total
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()


def run_hardening(
    detector: AccidentDetector,
    real_train: Sequence[LabeledClip],
    real_val: Sequence[LabeledClip],
    audit_records: Sequence[QuartetRecord],
    sources: Sequence[SourceClip],
    reference_generator: ReferenceGenerator,
    reference_gate: QualityGate,
    quartet_generator: QuartetGenerator,
    quartet_gate: QualityGate,
    clip_loader: ClipLoader,
    rng: np.random.Generator,
    config: Optional[HardeningConfig] = None,
    blocked_compositions: Sequence[str] = (),
) -> HardeningResult:
    """Run pretraining plus R audit-and-hardening rounds.

    ``audit_records`` are the validation quartets of the source dataset (the
    sampler validation pool, where held-out compositions are blocked, so the
    CGS-gap term renormalizes away); ``sources`` are training-split source
    clips feeding generation; ``blocked_compositions`` carries the held-out
    composition keys so sampling never targets them.
    """
    cfg = config or HardeningConfig()
    if not real_train or not real_val or not audit_records or not sources:
        raise ValueError("hardening needs real train/val clips, audit quartets, and sources")
    device = _resolve_device(cfg)
    detector.to(device)

    _fit_supervised(
        detector, real_train, clip_loader, cfg, cfg.pretrain_epochs, device, rng
    )

    reports: list[RoundReport] = []
    synthetic: list[QuartetRecord] = []
    for round_index in range(1, cfg.rounds + 1):
        tau = calibrate_operating_point(
            detector, real_val, clip_loader, device, cfg.batch_size
        )
        score_quartet_records(detector, audit_records, clip_loader, device, cfg.batch_size)
        audit = audit_quartets(audit_records, tau)
        diagnostics = per_level_diagnostics(audit_records, tau, cfg.shrinkage)
        stress = factor_stress_scores(diagnostics)
        sampler = WeakFactorSampler(
            stress,
            alpha=cfg.alpha,
            uniform=cfg.uniform_sampling,
            blocked=blocked_compositions,
        )
        budget = RoundBudget(len(real_train), cfg.budget_fraction, cfg.oversample)
        accepted, stats = generate_round(
            sampler,
            budget,
            sources,
            reference_generator,
            reference_gate,
            quartet_generator,
            quartet_gate,
            rng,
        )
        synthetic.extend(accepted)
        _fit_hardening(detector, real_train, synthetic, clip_loader, cfg, device, rng)
        reports.append(
            RoundReport(
                index=round_index,
                tau=tau,
                audit=audit,
                stress=stress,
                accepted_quartets=len(accepted),
                attempted_candidates=stats["attempted"],
                reference_rejected=stats["reference_rejected"],
                quartet_rejected=stats["quartet_rejected"],
            )
        )
    return HardeningResult(rounds=reports, synthetic=synthetic)


__all__ = [
    "HardeningConfig",
    "HardeningResult",
    "QualityGate",
    "QuartetGenerator",
    "ReferenceGenerator",
    "RoundReport",
    "SourceClip",
    "calibrate_operating_point",
    "generate_round",
    "run_hardening",
    "score_clips",
    "score_quartet_records",
]
