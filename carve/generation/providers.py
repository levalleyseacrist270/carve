"""Provider abstractions for generation backends and VLM judges.

The pipeline is backend-agnostic: any image editor, video editor, or vision-
language judge can be plugged in by satisfying the protocols below. Generic
JSON-over-HTTP adapters are provided for OpenAI-compatible endpoints,
Google-compatible endpoints, and locally hosted open-weights servers. HTTP
support is optional; the adapters raise a clear error when the ``requests``
package is unavailable instead of failing at import time.
"""

from __future__ import annotations

import base64
import dataclasses
import mimetypes
import os
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

try:  # optional dependency: only HTTP adapters need it
    import requests
except ImportError:  # pragma: no cover - environment dependent
    requests = None

_TIMEOUT_S = 300


@dataclasses.dataclass
class VideoEditRequest:
    """One phase-two video-editing request.

    Attributes:
        source_path: Real source clip to edit.
        instruction: Structured text instruction for the edit.
        out_path: Where the edited clip must be written.
        reference_image: Environment reference image, when the branch uses one.
        preservation_mask: Image path marking the region that must stay
            untouched (the complement of the edit region).
        lane_mask: Lane-mask image path, when available.
        trajectory_sketch: Coarse participant-trajectory sketch image path,
            for accident branches.
        extra: Backend-specific pass-through options.
    """

    source_path: str
    instruction: str
    out_path: str
    reference_image: str | None = None
    preservation_mask: str | None = None
    lane_mask: str | None = None
    trajectory_sketch: str | None = None
    extra: dict = dataclasses.field(default_factory=dict)


@runtime_checkable
class ImageEditBackend(Protocol):
    """Phase-one backend: key frames + layout cues + script -> reference image."""

    def generate(
        self,
        key_frames: Sequence[str],
        layout_overlays: Sequence[str],
        instruction: str,
        out_path: str,
    ) -> str:
        """Generate a reference image and return its path."""
        ...


@runtime_checkable
class VideoEditBackend(Protocol):
    """Phase-two backend: source clip + conditioning + script -> edited clip."""

    def edit(self, request: VideoEditRequest) -> str:
        """Run one video edit and return the output clip path."""
        ...


@runtime_checkable
class VlmJudge(Protocol):
    """Vision-language judge used by the reference and panel gates."""

    name: str

    def query(self, prompt: str, images: Sequence[str], temperature: float = 0.0) -> str:
        """Return the raw text reply for a prompt with attached images."""
        ...


def _require_requests() -> None:
    if requests is None:
        raise RuntimeError(
            "HTTP providers require the 'requests' package; install it or plug in"
            " a custom provider object"
        )


def _data_uri(path: str | Path) -> str:
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    payload = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def _resolve_key(api_key: str | None, api_key_env: str | None) -> str | None:
    if api_key:
        return api_key
    if api_key_env:
        return os.environ.get(api_key_env)
    return None


@dataclasses.dataclass
class HttpImageEditBackend:
    """Generic JSON-over-HTTP adapter for an image-editing endpoint.

    Posts the instruction plus base64 conditioning images and expects a JSON
    reply carrying the generated image as base64 under ``image_b64``.
    """

    endpoint: str
    model: str
    api_key: str | None = None
    api_key_env: str | None = None

    def generate(
        self,
        key_frames: Sequence[str],
        layout_overlays: Sequence[str],
        instruction: str,
        out_path: str,
    ) -> str:
        _require_requests()
        payload = {
            "model": self.model,
            "instruction": instruction,
            "key_frames": [_data_uri(p) for p in key_frames],
            "layout_overlays": [_data_uri(p) for p in layout_overlays],
        }
        reply = _post_json(self.endpoint, payload, _resolve_key(self.api_key, self.api_key_env))
        Path(out_path).write_bytes(base64.b64decode(reply["image_b64"]))
        return out_path


@dataclasses.dataclass
class HttpVideoEditBackend:
    """Generic JSON-over-HTTP adapter for a video-editing endpoint.

    Posts the instruction plus base64 conditioning media and expects a JSON
    reply carrying the edited clip as base64 under ``video_b64``.
    """

    endpoint: str
    model: str
    api_key: str | None = None
    api_key_env: str | None = None

    def edit(self, request: VideoEditRequest) -> str:
        _require_requests()
        payload: dict = {
            "model": self.model,
            "instruction": request.instruction,
            "source_video_b64": base64.b64encode(
                Path(request.source_path).read_bytes()
            ).decode("ascii"),
            **request.extra,
        }
        for field in ("reference_image", "preservation_mask", "lane_mask", "trajectory_sketch"):
            value = getattr(request, field)
            if value:
                payload[field] = _data_uri(value)
        reply = _post_json(self.endpoint, payload, _resolve_key(self.api_key, self.api_key_env))
        Path(request.out_path).write_bytes(base64.b64decode(reply["video_b64"]))
        return request.out_path


def _post_json(url: str, payload: dict, api_key: str | None) -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    response = requests.post(url, json=payload, headers=headers, timeout=_TIMEOUT_S)
    response.raise_for_status()
    return response.json()


@dataclasses.dataclass
class OpenAiCompatibleJudge:
    """Judge adapter for OpenAI-compatible chat-completions endpoints."""

    endpoint: str
    model: str
    name: str = "openai_compatible"
    api_key: str | None = None
    api_key_env: str | None = None

    def query(self, prompt: str, images: Sequence[str], temperature: float = 0.0) -> str:
        _require_requests()
        content: list[dict] = [{"type": "text", "text": prompt}]
        content += [
            {"type": "image_url", "image_url": {"url": _data_uri(p)}} for p in images
        ]
        payload = {
            "model": self.model,
            "temperature": temperature,
            "messages": [{"role": "user", "content": content}],
        }
        url = self.endpoint.rstrip("/") + "/chat/completions"
        reply = _post_json(url, payload, _resolve_key(self.api_key, self.api_key_env))
        return reply["choices"][0]["message"]["content"]


@dataclasses.dataclass
class GoogleCompatibleJudge:
    """Judge adapter for Google-style ``generateContent`` endpoints."""

    endpoint: str
    model: str
    name: str = "google_compatible"
    api_key: str | None = None
    api_key_env: str | None = None

    def query(self, prompt: str, images: Sequence[str], temperature: float = 0.0) -> str:
        _require_requests()
        parts: list[dict] = [{"text": prompt}]
        for path in images:
            mime = mimetypes.guess_type(str(path))[0] or "image/png"
            data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
            parts.append({"inline_data": {"mime_type": mime, "data": data}})
        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"temperature": temperature},
        }
        key = _resolve_key(self.api_key, self.api_key_env)
        url = f"{self.endpoint.rstrip('/')}/models/{self.model}:generateContent"
        if key:
            url = f"{url}?key={key}"
        reply = _post_json(url, payload, None)
        return reply["candidates"][0]["content"]["parts"][0]["text"]


@dataclasses.dataclass
class LocalOpenWeightsJudge(OpenAiCompatibleJudge):
    """Judge adapter for a locally hosted open-weights VLM server.

    Local inference servers commonly expose the OpenAI-compatible route, so
    this adapter only changes the defaults.
    """

    endpoint: str = "http://localhost:8000/v1"
    model: str = "local-vlm"
    name: str = "local_open_weights"


_JUDGE_KINDS = {
    "openai_compatible": OpenAiCompatibleJudge,
    "google_compatible": GoogleCompatibleJudge,
    "local_open_weights": LocalOpenWeightsJudge,
}


def judge_from_config(spec: dict) -> VlmJudge:
    """Instantiate a judge from one ``panel.judges`` entry of the config."""
    kind = spec.get("kind")
    if kind not in _JUDGE_KINDS:
        raise ValueError(f"unknown judge kind: {kind!r}; expected one of {sorted(_JUDGE_KINDS)}")
    cls = _JUDGE_KINDS[kind]
    kwargs = {k: v for k, v in spec.items() if k != "kind"}
    kwargs.setdefault("name", kind)
    return cls(**kwargs)
