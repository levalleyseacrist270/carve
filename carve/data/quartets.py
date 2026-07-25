"""Quartet and clip datasets plus manifest serialization.

Video decoding is injected as a ``loader`` callable mapping a media path to a
float tensor of shape ``[T, C, H, W]`` (values in [0, 1], 64 frames at the
224 x 224 analysis resolution), so the datasets stay agnostic to codec and
storage layout.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence, Union

import torch
from torch.utils.data import Dataset

from ..core import AccidentSpec, Branch, EnvironmentSpec, QuartetRecord

ClipLoader = Callable[[str], torch.Tensor]
PathLike = Union[str, Path]


@dataclass(frozen=True)
class LabeledClip:
    """One real clip with its binary accident label."""

    path: str
    label: int
    source_id: str = ""
    dataset: str = ""


class ClipDataset(Dataset):
    """Labeled real clips for supervised pretraining and calibration."""

    def __init__(self, clips: Sequence[LabeledClip], loader: ClipLoader) -> None:
        self.clips = list(clips)
        self.loader = loader

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        clip = self.clips[index]
        video = self.loader(clip.path)
        return video, torch.tensor(float(clip.label), dtype=torch.float32)


class QuartetDataset(Dataset):
    """Matched quartets for the quartet-aware training losses.

    Each item is a dict with the four branch tensors keyed by branch name, so
    the default collate produces per-branch batches aligned across branches.
    """

    def __init__(self, records: Sequence[QuartetRecord], loader: ClipLoader) -> None:
        self.records = list(records)
        self.loader = loader

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        item: dict[str, torch.Tensor] = {}
        for branch in Branch:
            try:
                path = record.paths[branch.value]
            except KeyError as exc:
                raise KeyError(
                    f"quartet {record.quartet_id!r} has no path for branch {branch.value!r}"
                ) from exc
            item[branch.value] = self.loader(path)
        return item


def record_to_dict(record: QuartetRecord) -> dict:
    """JSON-compatible form of a quartet record."""
    accident = asdict(record.accident)
    accident["participants"] = list(record.accident.participants)
    if record.accident.impact_region is not None:
        accident["impact_region"] = list(record.accident.impact_region)
    return {
        "quartet_id": record.quartet_id,
        "source_id": record.source_id,
        "dataset": record.dataset,
        "split": record.split,
        "env": asdict(record.env),
        "accident": accident,
        "held_out_composition": record.held_out_composition,
        "paths": dict(record.paths),
        "scores": dict(record.scores),
    }


def record_from_dict(data: dict) -> QuartetRecord:
    """Inverse of :func:`record_to_dict`."""
    accident = dict(data["accident"])
    accident["participants"] = tuple(accident.get("participants", ()))
    region = accident.get("impact_region")
    accident["impact_region"] = tuple(region) if region is not None else None
    return QuartetRecord(
        quartet_id=data["quartet_id"],
        source_id=data["source_id"],
        dataset=data["dataset"],
        split=data["split"],
        env=EnvironmentSpec(**data["env"]),
        accident=AccidentSpec(**accident),
        held_out_composition=bool(data.get("held_out_composition", False)),
        paths=dict(data.get("paths", {})),
        scores={k: float(v) for k, v in data.get("scores", {}).items()},
    )


def save_manifest(records: Iterable[QuartetRecord], path: PathLike) -> None:
    """Write quartet records as one JSON object per line."""
    with Path(path).open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record_to_dict(record), sort_keys=True) + "\n")


def load_manifest(path: PathLike) -> list[QuartetRecord]:
    """Read a JSON-lines quartet manifest."""
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(record_from_dict(json.loads(line)))
    return records


def load_labeled_clips(path: PathLike) -> list[LabeledClip]:
    """Read a JSON-lines manifest of real labeled clips."""
    clips = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            clips.append(
                LabeledClip(
                    path=data["path"],
                    label=int(data["label"]),
                    source_id=data.get("source_id", ""),
                    dataset=data.get("dataset", ""),
                )
            )
    return clips


__all__ = [
    "ClipDataset",
    "ClipLoader",
    "LabeledClip",
    "QuartetDataset",
    "load_labeled_clips",
    "load_manifest",
    "record_from_dict",
    "record_to_dict",
    "save_manifest",
]
