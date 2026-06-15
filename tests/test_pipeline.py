from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
import torch

from objtracker.pipeline import TrackingPipeline
from objtracker.tracking import Detections, Track
from objtracker.tracking.types import as_detections

if TYPE_CHECKING:
    from collections.abc import Mapping


class FakeDetector:
    def detect(self, frame: Any) -> dict[str, torch.Tensor]:
        assert frame == "frame"
        return {
            "boxes": torch.tensor([[0, 0, 10, 10]], dtype=torch.float32),
            "scores": torch.tensor([0.9], dtype=torch.float32),
        }


class FakeTracker:
    def __init__(self) -> None:
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def update(
        self,
        detections: Detections | Mapping[str, torch.Tensor],
        frame: Any | None = None,
    ) -> list[Track]:
        detections = as_detections(detections)
        labels = detections.labels
        assert labels is not None
        assert len(detections) == 1
        return [
            Track(
                track_id=1,
                box=detections.boxes[0],
                score=float(detections.scores[0]),
                label=int(labels[0]),
            )
        ]


def test_pipeline_connects_detector_to_tracker() -> None:
    pipeline = TrackingPipeline(detector=FakeDetector(), tracker=FakeTracker())

    result = pipeline.update("frame")

    assert len(result.detections) == 1
    assert len(result.tracks) == 1
    assert result.tracks[0].track_id == 1
    assert torch.equal(result.tracks[0].box, torch.tensor([0, 0, 10, 10]))
    assert result.tracks[0].score == pytest.approx(0.9)
    assert result.tracks[0].label == 0


def test_pipeline_reset_resets_tracker() -> None:
    tracker = FakeTracker()
    pipeline = TrackingPipeline(detector=FakeDetector(), tracker=tracker)

    pipeline.reset()

    assert tracker.reset_calls == 1
