"""Capture-quality probe: a lightweight 3-class clip classifier.

The probe distinguishes clean, compressed, and motion-blurred CCTV footage
and is used only for quality filtering: an edit passes the capture gate when
the predicted class matches the requested capture-quality factor. It is
trained once on manually labeled real clips and then frozen.
"""

from __future__ import annotations

import dataclasses
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import yaml
from torch import nn

from carve.gates.objective import load_frames

DEFAULT_CLASSES: tuple[str, ...] = ("clean", "compressed", "motion_blur")


@dataclasses.dataclass(frozen=True)
class ProbeConfig:
    """Training and inference settings for the capture-quality probe."""

    classes: tuple[str, ...] = DEFAULT_CLASSES
    frames_per_clip: int = 8
    resolution: int = 224
    epochs: int = 20
    lr: float = 1.0e-3
    batch_size: int = 16
    seed: int = 0

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ProbeConfig":
        """Load the ``gates.probe`` block plus the analysis resolution."""
        with open(path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        block = dict(raw.get("gates", {}).get("probe", {}))
        block.pop("checkpoint", None)
        if "classes" in block:
            block["classes"] = tuple(block["classes"])
        block["resolution"] = raw.get("analysis", {}).get("resolution", cls.resolution)
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in block.items() if k in fields})


class CaptureQualityProbe(nn.Module):
    """Small strided CNN over individual frames with mean-pooled clip logits."""

    def __init__(self, classes: Sequence[str] = DEFAULT_CLASSES):
        super().__init__()
        self.classes = tuple(classes)
        widths = (3, 32, 64, 128, 256)
        layers: list[nn.Module] = []
        for cin, cout in zip(widths[:-1], widths[1:]):
            layers += [
                nn.Conv2d(cin, cout, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            ]
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(widths[-1], len(self.classes))

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """Return per-frame logits for a ``(N, 3, H, W)`` batch."""
        return self.head(self.pool(self.features(frames)).flatten(1))


def _clip_tensor(clip: str | Path | np.ndarray, config: ProbeConfig) -> torch.Tensor:
    """Sample frames from a clip and normalize them to a ``(F, 3, H, W)`` tensor."""
    frames = clip if isinstance(clip, np.ndarray) else load_frames(clip, config.resolution)
    indices = np.linspace(0, len(frames) - 1, num=config.frames_per_clip).astype(int)
    array = frames[indices].astype(np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(0, 3, 1, 2)
    mean = tensor.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = tensor.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (tensor - mean) / std


def train_probe(
    labeled_clips: Sequence[tuple[str, str]],
    config: ProbeConfig | None = None,
    device: str | None = None,
) -> CaptureQualityProbe:
    """Train the probe on ``(clip_path, class_name)`` pairs.

    Every sampled frame inherits the clip label; the loss is cross-entropy
    over per-frame logits, which keeps the probe insensitive to clip length.
    """
    config = config or ProbeConfig()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config.seed)
    probe = CaptureQualityProbe(config.classes).to(device).train()
    optimizer = torch.optim.Adam(probe.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()
    class_index = {name: i for i, name in enumerate(probe.classes)}
    items = list(labeled_clips)
    rng = random.Random(config.seed)

    for _ in range(config.epochs):
        rng.shuffle(items)
        for start in range(0, len(items), config.batch_size):
            batch = items[start : start + config.batch_size]
            frames = torch.cat([_clip_tensor(path, config) for path, _ in batch])
            labels = torch.cat(
                [
                    torch.full((config.frames_per_clip,), class_index[label], dtype=torch.long)
                    for _, label in batch
                ]
            )
            optimizer.zero_grad()
            loss = criterion(probe(frames.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
    return probe.eval()


@torch.no_grad()
def predict(
    probe: CaptureQualityProbe,
    clip: str | Path | np.ndarray,
    config: ProbeConfig | None = None,
) -> str:
    """Predict the capture-quality class of a clip from mean frame logits."""
    config = config or ProbeConfig(classes=probe.classes)
    device = next(probe.parameters()).device
    logits = probe(_clip_tensor(clip, config).to(device)).mean(dim=0)
    return probe.classes[int(logits.argmax())]


def save_probe(probe: CaptureQualityProbe, path: str | Path) -> None:
    """Serialize probe weights and class names."""
    torch.save({"state_dict": probe.state_dict(), "classes": probe.classes}, path)


def load_probe(path: str | Path, device: str | None = None) -> CaptureQualityProbe:
    """Load a probe saved with :func:`save_probe`."""
    payload = torch.load(path, map_location=device or "cpu")
    probe = CaptureQualityProbe(payload["classes"])
    probe.load_state_dict(payload["state_dict"])
    return probe.eval()
