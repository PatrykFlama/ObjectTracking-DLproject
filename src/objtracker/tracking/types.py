from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch import Tensor


@dataclass(frozen=True)
class Detections:
    """Single-frame detector outputs in pixel xyxy format."""

    boxes: Tensor
    scores: Tensor
    labels: Tensor | None = None

    def __post_init__(self) -> None:
        boxes = self.boxes
        scores = self.scores

        if boxes.ndim != 2 or boxes.shape[1] != 4:
            msg = "Detections.boxes must have shape (N, 4)"
            raise ValueError(msg)
        if scores.ndim != 1:
            msg = "Detections.scores must have shape (N,)"
            raise ValueError(msg)
        if boxes.shape[0] != scores.shape[0]:
            msg = "Detections.boxes and Detections.scores must have the same length"
            raise ValueError(msg)

        if not torch.is_floating_point(boxes):
            boxes = boxes.float()
        if not torch.is_floating_point(scores):
            scores = scores.float()
        scores = scores.to(device=boxes.device)

        labels = self.labels
        if labels is None:
            labels = torch.zeros(boxes.shape[0], dtype=torch.long, device=boxes.device)
        else:
            if labels.ndim != 1:
                msg = "Detections.labels must have shape (N,)"
                raise ValueError(msg)
            if labels.shape[0] != boxes.shape[0]:
                msg = "Detections.boxes and Detections.labels must have the same length"
                raise ValueError(msg)
            labels = labels.to(device=boxes.device, dtype=torch.long)

        object.__setattr__(self, "boxes", boxes)
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "labels", labels)

    def __len__(self) -> int:
        return int(self.boxes.shape[0])

    @classmethod
    def from_dict(cls, prediction: Mapping[str, Tensor]) -> Detections:
        return cls(
            boxes=prediction["boxes"],
            scores=prediction["scores"],
            labels=prediction.get("labels"),
        )


@dataclass(frozen=True)
class Track:
    """A tracker output for one object in the current frame."""

    track_id: int
    box: Tensor
    score: float
    label: int


def as_detections(data: Detections | Mapping[str, Tensor]) -> Detections:
    if isinstance(data, Detections):
        return data
    return Detections.from_dict(data)
