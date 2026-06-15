from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from objtracker.tracking.types import as_detections

if TYPE_CHECKING:
    from objtracker.models.base import MultiObjectDetector
    from objtracker.tracking import Detections, MultiObjectTracker, Track


@dataclass(frozen=True)
class PipelineResult:
    detections: Detections
    tracks: list[Track]


class TrackingPipeline:
    def __init__(self, detector: MultiObjectDetector, tracker: MultiObjectTracker):
        self.detector = detector
        self.tracker = tracker

    def reset(self) -> None:
        self.tracker.reset()

    def detect(self, frame: Any) -> Detections:
        return as_detections(self.detector.detect(frame))

    def update(self, frame: Any) -> PipelineResult:
        detections = self.detect(frame)
        return PipelineResult(
            detections=detections,
            tracks=self.tracker.update(detections, frame=frame),
        )
