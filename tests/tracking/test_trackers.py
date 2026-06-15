from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from objtracker.tracking import Detections, Track, build_tracker
from objtracker.tracking.factory import TRACKERS

TRACKER_KWARGS: dict[str, dict[str, Any]] = {
    "botsort": {
        "track_activation_threshold": 0.5,
        "high_conf_det_threshold": 0.5,
    },
    "bytetrack": {
        "track_activation_threshold": 0.5,
        "high_conf_det_threshold": 0.5,
    },
    "deepsort": {},
}


def _detections(
    boxes: list[list[float]],
    scores: list[float],
    dtype: torch.dtype = torch.float32,
) -> Detections:
    return Detections(
        boxes=torch.tensor(boxes, dtype=dtype),
        scores=torch.tensor(scores, dtype=dtype),
    )


@pytest.mark.parametrize("tracker_name", sorted(TRACKERS))
def test_registered_tracker_produces_tracks(tracker_name: str) -> None:
    tracker = build_tracker(tracker_name, **TRACKER_KWARGS[tracker_name])

    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    tracks = []
    for _ in range(3):
        tracks = tracker.update(_detections([[0, 0, 10, 10]], [0.9]), frame=dummy_frame)
        if tracks:
            break

    assert tracks
    assert isinstance(tracks[0], Track)
    assert tracks[0].track_id >= 0
    assert tracks[0].score == pytest.approx(0.9)
    assert tracks[0].label == 0


@pytest.mark.parametrize("tracker_name", sorted(TRACKERS))
def test_registered_tracker_accepts_prediction_mapping(tracker_name: str) -> None:
    tracker = build_tracker(tracker_name, **TRACKER_KWARGS[tracker_name])
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    prediction = {
        "boxes": torch.tensor([[0, 0, 10, 10]], dtype=torch.float64),
        "scores": torch.tensor([0.9], dtype=torch.float64),
    }

    tracks = []
    for _ in range(3):
        tracks = tracker.update(prediction, frame=dummy_frame)
        if tracks:
            break

    assert tracks, f"{tracker_name} did not produce tracks from prediction mapping"
    assert tracks[0].box.dtype == torch.float64
    assert tracks[0].box.device == prediction["boxes"].device


@pytest.mark.parametrize("tracker_name", sorted(TRACKERS))
def test_registered_tracker_reset_clears_state(tracker_name: str) -> None:
    tracker = build_tracker(tracker_name, **TRACKER_KWARGS[tracker_name])
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    assert tracker.update(
        _detections([[0, 0, 10, 10]], [0.9]), frame=dummy_frame
    ) or tracker.update(_detections([[0, 0, 10, 10]], [0.9], frame=dummy_frame))

    tracker.reset()

    tracker.update(_detections([[20, 20, 30, 30]], [0.9]), frame=dummy_frame)
    tracks = tracker.update(_detections([[20, 20, 30, 30]], [0.9]), frame=dummy_frame)

    assert tracks


def test_build_tracker_rejects_unknown_tracker() -> None:
    with pytest.raises(ValueError, match="Unsupported tracker"):
        build_tracker("regression")
