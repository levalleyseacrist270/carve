"""Reference-image gate applied before any video generation.

The environment reference image produced in phase one is accepted only when
four independent yes/no checks all pass: source layout preservation, realized
target environment, absence of spurious crash evidence, and CCTV plausibility.
Failing references are discarded, so no video-generation budget is spent on a
broken environment prior.
"""

from __future__ import annotations

import json
import re
from typing import Sequence

from carve.core import EnvironmentSpec, GateResult

_ANSWER_TEMPLATE = (
    'Reply with exactly one JSON object and nothing else, following this '
    'template: {"answer": "yes" or "no", "reason": "<one short sentence>"}.'
)

_CONTEXT = (
    "The first image is a candidate reference image for a video edit. The "
    "remaining images are key frames of the real source CCTV clip, possibly "
    "with lane and road-boundary overlays."
)


def _environment_clause(environment: EnvironmentSpec) -> str:
    return (
        f"illumination {environment.illumination}, weather {environment.weather}, "
        f"road surface {environment.road_surface}, and capture quality "
        f"{environment.capture_quality}"
    )


def reference_checks(environment: EnvironmentSpec) -> list[tuple[str, str]]:
    """Ordered (name, question) pairs for the four reference sub-checks.

    Every question is phrased so that ``yes`` means the check passes.
    """
    return [
        (
            "layout",
            "Does the candidate reference image preserve the road topology, lane "
            "structure, and camera perspective of the source key frames?",
        ),
        (
            "environment",
            "Does the candidate reference image realize the requested environment: "
            f"{_environment_clause(environment)}?",
        ),
        (
            "no_accident",
            "Is the candidate reference image free of any collision, crash debris, "
            "damaged vehicle, or other accident evidence?",
        ),
        (
            "realism",
            "Is the candidate reference image plausible as a frame captured by a "
            "fixed roadside CCTV camera?",
        ),
    ]


def parse_yes_no(text: str) -> bool:
    """Parse a strict yes/no JSON reply.

    Raises:
        ValueError: If no JSON object is found or the answer field is neither
            ``"yes"`` nor ``"no"``.
    """
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match is None:
        raise ValueError("reply contains no JSON object")
    answer = str(json.loads(match.group(0)).get("answer", "")).strip().lower()
    if answer not in {"yes", "no"}:
        raise ValueError(f"answer is not yes/no: {answer!r}")
    return answer == "yes"


def _ask(judge, question: str, images: Sequence[str], retries: int = 1) -> tuple[bool, str]:
    """Run one sub-check with a single retry on malformed output."""
    prompt = f"{_CONTEXT}\n\n{question}\n\n{_ANSWER_TEMPLATE}"
    reply = judge.query(prompt, images=list(images), temperature=0.0)
    for attempt in range(retries + 1):
        try:
            return parse_yes_no(reply), reply
        except (ValueError, json.JSONDecodeError):
            if attempt >= retries:
                return False, reply
            reminder = prompt + "\n\nReturn only the JSON object."
            reply = judge.query(reminder, images=list(images), temperature=0.0)
    return False, reply  # pragma: no cover


def run_reference_gate(
    reference_image: str,
    key_frames: Sequence[str],
    environment: EnvironmentSpec,
    judge,
    retries: int = 1,
) -> GateResult:
    """Evaluate the reference gate as the conjunction of four sub-checks.

    Args:
        reference_image: Path of the candidate reference image.
        key_frames: Paths of source key frames, optionally with layout
            overlays baked in.
        environment: Requested environment for the reference image.
        judge: Provider exposing ``query(prompt, images, temperature) -> str``.
        retries: Retries per sub-check on malformed replies.

    Returns:
        A stage ``"reference"`` result with per-check verdicts and raw
        replies. A sub-check whose reply stays malformed counts as failed.
    """
    images = [reference_image, *key_frames]
    details: dict = {}
    passed = True
    for name, question in reference_checks(environment):
        verdict, reply = _ask(judge, question, images, retries)
        details[name] = {"passed": verdict, "reply": reply}
        passed = passed and verdict
        if not verdict:
            break  # later checks cannot rescue a failed conjunction
    return GateResult(passed=passed, stage="reference", details=details)
