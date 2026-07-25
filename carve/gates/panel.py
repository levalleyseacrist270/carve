"""Three-judge validity panel for generated edits.

Each judge receives the identical evidence packet and the fixed seven-part
rubric, is queried at temperature zero, and must reply with integer scores
from 1 to 5 in a JSON template. A clip is accepted when the minimum over
dimensions of the per-dimension median across judges reaches the rubric
threshold and the maximum per-dimension inter-quartile range stays within the
allowed spread.
"""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import yaml

from carve.core import Branch, GateResult


@dataclasses.dataclass(frozen=True)
class DimensionSpec:
    """One rubric dimension with score anchors and per-branch semantics."""

    key: str
    title: str
    question: str
    anchors: dict[int, str]
    branch_notes: dict[str, str] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class PanelRubric:
    """Fixed rubric, response protocol, and acceptance thresholds."""

    dimensions: tuple[DimensionSpec, ...]
    min_median: float = 4.0
    max_iqr: float = 1.0
    temperature: float = 0.0
    retries: int = 1
    score_range: tuple[int, int] = (1, 5)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PanelRubric":
        """Load the rubric shipped in ``configs/panel_rubric.yaml``."""
        with open(path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        dimensions = tuple(
            DimensionSpec(
                key=d["key"],
                title=d["title"],
                question=d["question"].strip(),
                anchors={int(k): v.strip() for k, v in d["anchors"].items()},
                branch_notes={k: v.strip() for k, v in d.get("branch_notes", {}).items()},
            )
            for d in raw["dimensions"]
        )
        acceptance = raw.get("acceptance", {})
        protocol = raw.get("protocol", {})
        return cls(
            dimensions=dimensions,
            min_median=float(acceptance.get("min_median", 4)),
            max_iqr=float(acceptance.get("max_iqr", 1)),
            temperature=float(protocol.get("temperature", 0.0)),
            retries=int(protocol.get("retries", 1)),
            score_range=tuple(protocol.get("score_range", (1, 5))),
        )


@dataclasses.dataclass
class EvidencePacket:
    """Standardized input shown to every judge for one candidate clip.

    Attributes:
        branch: Quartet branch of the candidate.
        script_text: Structured edit script for the candidate.
        target_label: Expected binary label (0 negative, 1 positive).
        key_frame_grid: Path of a timestamped key-frame grid of the edit.
        pair_strip: Path of matched source/edited thumbnail rows.
        mask_overlay: Path of the edit-region and participant/impact overlay,
            when the edit is localized.
        motion_evidence: Path of a compact motion visualization built from
            frame differences.
        reference_image: Path of the environment reference image; present for
            environment and joint branches only.
        onset_window: Scripted onset frame window for positive branches.
    """

    branch: Branch
    script_text: str
    target_label: int
    key_frame_grid: str
    pair_strip: str
    mask_overlay: str | None = None
    motion_evidence: str | None = None
    reference_image: str | None = None
    onset_window: tuple[int, int] | None = None

    def images(self) -> list[str]:
        """Ordered image paths of the packet, omitting absent items."""
        paths = [self.key_frame_grid, self.pair_strip, self.mask_overlay,
                 self.motion_evidence, self.reference_image]
        return [p for p in paths if p]


def build_panel_prompt(rubric: PanelRubric, packet: EvidencePacket) -> str:
    """Render the fixed judging prompt for one evidence packet."""
    low, high = rubric.score_range
    lines = [
        "You are auditing a generated CCTV video edit against its edit script.",
        "The attached images show, in order: a timestamped key-frame grid of the",
        "edited clip, matched source/edited thumbnail rows, and when present an",
        "edit-region overlay, a motion-evidence visualization, and the target",
        "environment reference image.",
        "",
        f"Branch: {packet.branch.value}",
        f"Target label: {packet.target_label} (0 = no accident, 1 = accident)",
        f"Edit script: {packet.script_text}",
    ]
    if packet.onset_window:
        lines.append(
            f"Scripted onset window: frames {packet.onset_window[0]}-{packet.onset_window[1]}."
        )
    if packet.reference_image:
        lines.append(
            "The reference image guides the requested environment appearance; the"
            " source scene must remain identifiable."
        )
    lines += ["", f"Score each dimension with an integer from {low} to {high}:"]
    for dim in rubric.dimensions:
        lines += ["", f"- {dim.key}: {dim.question}"]
        note = dim.branch_notes.get(packet.branch.value)
        if note:
            lines.append(f"  For branch {packet.branch.value}: {note}")
        for score in sorted(dim.anchors):
            lines.append(f"  {score}: {dim.anchors[score]}")
    template = ", ".join(f'"{dim.key}": <int>' for dim in rubric.dimensions)
    lines += [
        "",
        "Reply with exactly one JSON object and nothing else, following this",
        f"template: {{{template}}}",
    ]
    return "\n".join(lines)


def parse_score_json(text: str, keys: Sequence[str], score_range: tuple[int, int]) -> dict[str, int]:
    """Parse a judge reply into validated integer scores.

    Accepts replies that wrap the JSON object in prose or code fences.

    Raises:
        ValueError: If no JSON object is found, a key is missing, or a score
            is not an integer within ``score_range``.
    """
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match is None:
        raise ValueError("reply contains no JSON object")
    payload = json.loads(match.group(0))
    low, high = score_range
    scores: dict[str, int] = {}
    for key in keys:
        if key not in payload:
            raise ValueError(f"missing score for dimension '{key}'")
        value = payload[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"score for '{key}' is not an integer: {value!r}")
        if not low <= value <= high:
            raise ValueError(f"score for '{key}' outside [{low}, {high}]: {value}")
        scores[key] = value
    return scores


def query_judge(judge, prompt: str, packet: EvidencePacket, rubric: PanelRubric) -> dict[str, int]:
    """Query one judge with retry on malformed output.

    The retry appends an explicit format reminder; a second malformed reply
    propagates as ``ValueError`` so the caller can reject the candidate.
    """
    keys = [dim.key for dim in rubric.dimensions]
    reply = judge.query(prompt, images=packet.images(), temperature=rubric.temperature)
    for attempt in range(rubric.retries + 1):
        try:
            return parse_score_json(reply, keys, rubric.score_range)
        except (ValueError, json.JSONDecodeError):
            if attempt >= rubric.retries:
                raise
            reminder = prompt + "\n\nReturn only the JSON object, with integer values."
            reply = judge.query(reminder, images=packet.images(), temperature=rubric.temperature)
    raise ValueError("unreachable")  # pragma: no cover


def aggregate_scores(
    per_judge: Mapping[str, Mapping[str, int]], keys: Sequence[str]
) -> tuple[dict[str, float], dict[str, float]]:
    """Per-dimension medians and inter-quartile ranges across judges.

    The IQR uses the linear-interpolation quantile rule on the ordered
    discrete scores, applied identically for every clip.
    """
    medians, iqrs = {}, {}
    for key in keys:
        values = np.asarray([scores[key] for scores in per_judge.values()], dtype=float)
        medians[key] = float(np.median(values))
        iqrs[key] = float(np.percentile(values, 75) - np.percentile(values, 25))
    return medians, iqrs


def panel_semantic_score(medians: Mapping[str, float]) -> float:
    """Continuous 0-100 fidelity summary: mean normalized per-dimension median."""
    values = np.asarray(list(medians.values()), dtype=float)
    return float(100.0 / len(values) * np.sum((values - 1.0) / 4.0))


def run_panel_gate(packet: EvidencePacket, judges: Sequence, rubric: PanelRubric) -> GateResult:
    """Run the full panel on one candidate and apply the acceptance rule.

    Args:
        packet: Evidence packet for the candidate.
        judges: Judge providers; each must expose
            ``query(prompt, images, temperature) -> str`` and a ``name``.
        rubric: Fixed rubric with acceptance thresholds.

    Returns:
        A stage ``"panel"`` result whose details carry raw per-judge scores,
        per-dimension medians and IQRs, and the continuous fidelity score. A
        judge whose reply stays malformed after the retry fails the gate.
    """
    keys = [dim.key for dim in rubric.dimensions]
    prompt = build_panel_prompt(rubric, packet)
    raw: dict[str, dict[str, int]] = {}
    for judge in judges:
        try:
            raw[judge.name] = query_judge(judge, prompt, packet, rubric)
        except (ValueError, json.JSONDecodeError) as error:
            return GateResult(
                passed=False,
                stage="panel",
                details={"error": f"judge '{judge.name}' returned malformed scores: {error}",
                         "raw_scores": raw},
            )
    medians, iqrs = aggregate_scores(raw, keys)
    min_median = min(medians.values())
    max_iqr = max(iqrs.values())
    passed = min_median >= rubric.min_median and max_iqr <= rubric.max_iqr
    return GateResult(
        passed=passed,
        stage="panel",
        details={
            "raw_scores": raw,
            "medians": medians,
            "iqrs": iqrs,
            "min_median": min_median,
            "max_iqr": max_iqr,
            "s_panel": panel_semantic_score(medians),
            "thresholds": {"min_median": rubric.min_median, "max_iqr": rubric.max_iqr},
        },
    )
