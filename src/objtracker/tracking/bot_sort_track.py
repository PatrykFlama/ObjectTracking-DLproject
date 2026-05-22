from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from objtracker.tracking.supervision_adapter import to_supervision_detections, to_tracks
from objtracker.tracking.types import Detections, Track, as_detections

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch import Tensor


@dataclass(frozen=True)
class BoTSORTTrackConfig:
    """Configuration passed through to `trackers.BoTSORTTracker`."""

    lost_track_buffer: int = 30
    frame_rate: float = 30.0
    track_activation_threshold: float = 0.7
    minimum_consecutive_frames: int = 2
    minimum_iou_threshold_first_assoc: float = 0.2
    minimum_iou_threshold_second_assoc: float = 0.5
    minimum_iou_threshold_unconfirmed_assoc: float = 0.3
    high_conf_det_threshold: float = 0.6
    enable_cmc: bool = True
    cmc_method: Literal["orb", "sift", "sparseOptFlow", "ecc"] = "sparseOptFlow"
    cmc_downscale: int = 2
    instant_first_frame_activation: bool = True


class BoTSORTTrack:
    """Thin adapter around `trackers.BoTSORTTracker`."""

    def __init__(self, config: BoTSORTTrackConfig | None = None):
        from trackers import BoTSORTTracker

        self.config = config or BoTSORTTrackConfig()
        self._tracker = BoTSORTTracker(**self.config.__dict__)

    def reset(self) -> None:
        self._tracker.reset()

    def update(self, detections: Detections | Mapping[str, Tensor]) -> list[Track]:
        frame_detections = as_detections(detections)
        tracked_detections = self._tracker.update(
            to_supervision_detections(frame_detections)
        )
        return to_tracks(
            tracked_detections,
            device=frame_detections.boxes.device,
            dtype=frame_detections.boxes.dtype,
        )
