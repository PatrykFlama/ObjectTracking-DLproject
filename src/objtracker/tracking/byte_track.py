from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from objtracker.tracking.supervision_adapter import to_supervision_detections, to_tracks
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

    def update(
        self, detections: Detections | Mapping[str, Tensor], frame: Any | None = None
    ) -> list[Track]:
        frame_detections = as_detections(detections)
        tracked_detections = self._tracker.update(
            to_supervision_detections(frame_detections)
        )
        return to_tracks(
            tracked_detections,
            device=frame_detections.boxes.device,
            dtype=frame_detections.boxes.dtype,
        )
