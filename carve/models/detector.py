"""Clip-level accident detectors with a probability head.

Hardening changes training only: a hardened detector keeps the base model's
inference graph, so deployment cost and latency are unchanged.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

DEFAULT_CHECKPOINT = "MCG-NJU/videomae-base"
DEFAULT_NUM_FRAMES = 64
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_WEIGHT_DECAY = 0.05
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 30


class AccidentDetector(nn.Module):
    """Base interface: video batch in, accident probability per clip out.

    ``forward`` takes a float tensor of shape ``[B, T, C, H, W]`` and returns
    probabilities of shape ``[B]`` in [0, 1].
    """

    def forward(self, video: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    @torch.no_grad()
    def predict(self, video: torch.Tensor) -> torch.Tensor:
        """Score a batch in eval mode, restoring the previous train state."""
        was_training = self.training
        self.eval()
        try:
            return self.forward(video)
        finally:
            if was_training:
                self.train()


class VideoMAEDetector(AccidentDetector):
    """VideoMAE backbone with a linear + sigmoid accident head.

    The backbone uses fixed sinusoidal position embeddings, so the frame
    count is configurable; it must be even to match the tubelet size.
    """

    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        num_frames: int = DEFAULT_NUM_FRAMES,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        try:
            from transformers import VideoMAEConfig, VideoMAEModel
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "VideoMAEDetector requires the 'transformers' package; "
                "install it with: pip install transformers"
            ) from exc
        if num_frames % 2 != 0:
            raise ValueError("num_frames must be even for the VideoMAE tubelet size")
        if pretrained:
            self.backbone = VideoMAEModel.from_pretrained(
                checkpoint, num_frames=num_frames
            )
        else:
            config = VideoMAEConfig(num_frames=num_frames)
            self.backbone = VideoMAEModel(config)
        self.head = nn.Linear(self.backbone.config.hidden_size, 1)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        if video.dim() != 5:
            raise ValueError(f"expected video of shape [B, T, C, H, W], got {tuple(video.shape)}")
        features = self.backbone(pixel_values=video).last_hidden_state
        pooled = features.mean(dim=1)
        return torch.sigmoid(self.head(pooled)).squeeze(-1)


def configure_training(
    model: nn.Module,
    steps_per_epoch: int,
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    min_lr: float = 0.0,
) -> tuple[AdamW, CosineAnnealingLR]:
    """AdamW plus per-step cosine decay over the full schedule."""
    if steps_per_epoch <= 0 or epochs <= 0:
        raise ValueError("steps_per_epoch and epochs must be positive")
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=max(1, epochs * steps_per_epoch), eta_min=min_lr
    )
    return optimizer, scheduler


__all__ = [
    "AccidentDetector",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_CHECKPOINT",
    "DEFAULT_EPOCHS",
    "DEFAULT_LEARNING_RATE",
    "DEFAULT_NUM_FRAMES",
    "DEFAULT_WEIGHT_DECAY",
    "VideoMAEDetector",
    "configure_training",
]
