from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from objtracker.tracking.types import Detections, Track, as_detections

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch import Tensor


@dataclass(frozen=True)
class ByteTrackConfig:
    """Configuration passed through to `trackers.ByteTrackTracker`."""

    lost_track_buffer: int = 30
    frame_rate: float = 30.0
    track_activation_threshold: float = 0.7
    minimum_consecutive_frames: int = 2
    minimum_iou_threshold: float = 0.1
    high_conf_det_threshold: float = 0.6


class ByteTrack:
    """Thin adapter around `trackers.ByteTrackTracker`."""

    def __init__(self, config: ByteTrackConfig | None = None):
        from trackers import ByteTrackTracker

        self.config = config or ByteTrackConfig()
        self._tracker = ByteTrackTracker(**self.config.__dict__)

    def reset(self) -> None:
        self._tracker.reset()

    def update(self, detections: Detections | Mapping[str, Tensor]) -> list[Track]:
        frame_detections = as_detections(detections)
        tracked_detections = self._tracker.update(
            _to_supervision_detections(frame_detections)
        )
        return _to_tracks(
            tracked_detections,
            device=frame_detections.boxes.device,
            dtype=frame_detections.boxes.dtype,
        )


def _to_supervision_detections(detections: Detections) -> Any:
    import supervision as sv

    labels = detections.labels
    assert labels is not None
    return sv.Detections(
        xyxy=_to_numpy(detections.boxes, "float32"),
        confidence=_to_numpy(detections.scores, "float32"),
        class_id=_to_numpy(labels, "int64"),
    )


def _to_tracks(
    detections: Any,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> list[Track]:
    tracker_ids = detections.tracker_id
    if tracker_ids is None:
        return []

    return [
        Track(
            track_id=int(track_id),
            box=torch.as_tensor(detections.xyxy[index], dtype=dtype, device=device),
            score=(
                0.0
                if detections.confidence is None
                else float(detections.confidence[index])
            ),
            label=0 if detections.class_id is None else int(detections.class_id[index]),
        )
        for index, track_id in enumerate(tracker_ids)
        if track_id >= 0
    ]


def _to_numpy(tensor: Tensor, dtype: str) -> Any:
    return tensor.detach().cpu().numpy().astype(dtype, copy=True)
