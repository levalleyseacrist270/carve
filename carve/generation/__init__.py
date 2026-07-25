"""Two-phase generation orchestration for counterfactual quartets.

Phase one turns source key frames, layout cues, and an environment script
into a gated reference image. Phase two performs reference-guided video
editing for the environment, accident, and joint branches, then applies the
objective and panel gates before a quartet is accepted.
"""

from carve.generation.layout_cues import (
    LayoutCues,
    accident_region_mask,
    extract_layout_cues,
    render_trajectory_sketch,
)
from carve.generation.providers import (
    GoogleCompatibleJudge,
    HttpImageEditBackend,
    HttpVideoEditBackend,
    ImageEditBackend,
    LocalOpenWeightsJudge,
    OpenAiCompatibleJudge,
    VideoEditBackend,
    VideoEditRequest,
    VlmJudge,
    judge_from_config,
)
from carve.generation.reference_image import ReferenceResult, generate_reference
from carve.generation.scripts_builder import (
    ONSET_WINDOWS,
    accident_script,
    branch_instruction,
    environment_edit_instruction,
    environment_script,
    event_edit_instruction,
    onset_window,
    reference_image_instruction,
)
from carve.generation.video_edit import (
    Backends,
    GateSuite,
    SourceUnit,
    build_evidence_packet,
    generate_quartet,
)

__all__ = [
    "Backends",
    "GateSuite",
    "GoogleCompatibleJudge",
    "HttpImageEditBackend",
    "HttpVideoEditBackend",
    "ImageEditBackend",
    "LayoutCues",
    "LocalOpenWeightsJudge",
    "ONSET_WINDOWS",
    "OpenAiCompatibleJudge",
    "ReferenceResult",
    "SourceUnit",
    "VideoEditBackend",
    "VideoEditRequest",
    "VlmJudge",
    "accident_region_mask",
    "accident_script",
    "branch_instruction",
    "build_evidence_packet",
    "environment_edit_instruction",
    "environment_script",
    "event_edit_instruction",
    "extract_layout_cues",
    "generate_quartet",
    "generate_reference",
    "judge_from_config",
    "onset_window",
    "reference_image_instruction",
    "render_trajectory_sketch",
]
