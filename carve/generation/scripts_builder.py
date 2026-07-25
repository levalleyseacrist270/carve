"""Structured edit scripts and generation instructions.

Every generated branch is driven by text built here: a phase-one instruction
for the environment reference image, a strict minimal-change instruction for
environment-branch video edits, and an event instruction for accident and
joint branches. The instruction wording is fixed; only factor values, script
fields, and feasibility notes are substituted.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from carve.core import AccidentSpec, Branch, EnvironmentSpec

# Visible impact windows per onset level for 64-frame clips decoded at 16 fps.
# Mirrors configs/factors.yaml; pass explicit windows to override.
ONSET_WINDOWS: dict[str, tuple[int, int]] = {
    "early": (24, 28),
    "middle": (29, 34),
    "late": (35, 40),
}

_ENVIRONMENT_EDIT_TEMPLATE = (
    "Preserve camera pose, road geometry, lane count, vehicle identities, and "
    "all object trajectories. Treat the provided reference image only as a "
    "target environment appearance prior. Keep accident state unchanged as "
    "none. Change only illumination to {illumination}, weather to {weather}, "
    "road surface to {road_surface}, and capture quality to {capture_quality}. "
    "Do not add or remove vehicles. Do not alter scene layout or camera motion."
)

_EVENT_EDIT_TEMPLATE = (
    "Insert the scripted collision (type, impact region, onset, and severity "
    "from the event script) by perturbing only the selected participants' "
    "trajectories; preserve camera pose, road geometry, lane count, and the "
    "trajectories of all non-participant vehicles{joint_clause}."
)

_JOINT_CLAUSE = ", and realize the requested environment from the reference image"

_REFERENCE_IMAGE_TEMPLATE = (
    "Generate one reference image of this scene under the target environment, "
    "using the provided key frames and layout overlays. Preserve camera "
    "perspective, road topology, lane structure, and all vehicles exactly as "
    "in the key frames. Change only appearance nuisance factors: illumination "
    "to {illumination}, weather to {weather}, road surface to {road_surface}, "
    "and capture quality to {capture_quality}. Do not add any collision, "
    "damage, or other accident evidence. Do not add or remove vehicles."
)


def load_onset_windows(factors_path: str | Path) -> dict[str, tuple[int, int]]:
    """Read the onset-level frame windows from the factor-space file."""
    with open(factors_path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return {level: tuple(window) for level, window in raw["onset_windows"].items()}


def onset_window(
    level: str, windows: dict[str, tuple[int, int]] | None = None
) -> tuple[int, int]:
    """Map an onset level to its visible impact frame window."""
    table = windows or ONSET_WINDOWS
    if level not in table:
        raise ValueError(f"unknown onset level {level!r}; expected one of {sorted(table)}")
    return table[level]


def environment_script(environment: EnvironmentSpec, note: str | None = None) -> str:
    """Structured environment script listing the requested factor levels.

    Args:
        environment: Target environment factors.
        note: Optional feasibility note (for example the stated cause of a wet
            road under clear weather) appended verbatim.
    """
    script = (
        f"illumination={environment.illumination}; weather={environment.weather}; "
        f"road_surface={environment.road_surface}; "
        f"capture_quality={environment.capture_quality}"
    )
    return f"{script}. Note: {note}" if note else script


def accident_script(
    accident: AccidentSpec, windows: dict[str, tuple[int, int]] | None = None
) -> str:
    """Structured event script with the onset level resolved to a frame window."""
    start, end = onset_window(accident.onset, windows)
    participants = ", ".join(str(p) for p in accident.participants) or "unspecified"
    impact = accident.impact_region if accident.impact_region is not None else "unspecified"
    return (
        f"type={accident.accident_type}; severity={accident.severity}; "
        f"onset={accident.onset} (frames {start}-{end}); "
        f"participants={participants}; impact_region={impact}"
    )


def reference_image_instruction(environment: EnvironmentSpec, note: str | None = None) -> str:
    """Phase-one instruction for the environment reference image."""
    text = _REFERENCE_IMAGE_TEMPLATE.format(
        illumination=environment.illumination,
        weather=environment.weather,
        road_surface=environment.road_surface,
        capture_quality=environment.capture_quality,
    )
    return f"{text} {note}" if note else text


def environment_edit_instruction(environment: EnvironmentSpec, note: str | None = None) -> str:
    """Strict minimal-change instruction for the environment branch."""
    text = _ENVIRONMENT_EDIT_TEMPLATE.format(
        illumination=environment.illumination,
        weather=environment.weather,
        road_surface=environment.road_surface,
        capture_quality=environment.capture_quality,
    )
    return f"{text} {note}" if note else text


def event_edit_instruction(
    accident: AccidentSpec,
    branch: Branch,
    windows: dict[str, tuple[int, int]] | None = None,
) -> str:
    """Event instruction for the accident-only or joint branch.

    The joint branch additionally requires realizing the requested
    environment from the shared reference image; the accident-only branch
    preserves the source environment and carries no reference.
    """
    if branch not in (Branch.VA, Branch.VAE):
        raise ValueError(f"event instructions apply to VA/VAE branches, got {branch}")
    text = _EVENT_EDIT_TEMPLATE.format(
        joint_clause=_JOINT_CLAUSE if branch is Branch.VAE else ""
    )
    return f"{text} Event script: {accident_script(accident, windows)}."


def branch_instruction(
    branch: Branch,
    environment: EnvironmentSpec,
    accident: AccidentSpec | None,
    note: str | None = None,
    windows: dict[str, tuple[int, int]] | None = None,
) -> str:
    """Dispatch to the instruction builder for a generated branch."""
    if branch is Branch.VE:
        return environment_edit_instruction(environment, note)
    if accident is None:
        raise ValueError(f"branch {branch} requires an accident script")
    return event_edit_instruction(accident, branch, windows)
