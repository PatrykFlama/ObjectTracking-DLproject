from __future__ import annotations

import pytest
import torch

from objtracker.tracking import ByteTrack, ByteTrackConfig, Detections, build_tracker


def _detections(boxes: list[list[float]], scores: list[float]) -> Detections:
    return Detections(
        boxes=torch.tensor(boxes, dtype=torch.float32),
        scores=torch.tensor(scores, dtype=torch.float32),
    )


def test_bytetrack_returns_confirmed_tracks_after_match() -> None:
    tracker = ByteTrack(
        ByteTrackConfig(
            track_activation_threshold=0.5,
            high_conf_det_threshold=0.5,
        )
    )

    first_tracks = tracker.update(_detections([[0, 0, 10, 10]], [0.9]))
    second_tracks = tracker.update(_detections([[1, 0, 11, 10]], [0.8]))

    assert first_tracks == []
    assert len(second_tracks) == 1
    assert second_tracks[0].track_id >= 0


def test_low_confidence_detection_can_continue_existing_track() -> None:
    tracker = ByteTrack(
        ByteTrackConfig(
            track_activation_threshold=0.5,
            high_conf_det_threshold=0.5,
        )
    )

    tracker.update(_detections([[0, 0, 10, 10]], [0.9]))
    tracks = tracker.update(_detections([[1, 0, 11, 10]], [0.2]))

    assert tracks[0].track_id >= 0
    assert tracks[0].score == pytest.approx(0.2)


def test_accepts_prediction_mapping_and_preserves_output_device_and_dtype() -> None:
    tracker = ByteTrack(
        ByteTrackConfig(
            track_activation_threshold=0.5,
            high_conf_det_threshold=0.5,
        )
    )
    prediction = {
        "boxes": torch.tensor([[0, 0, 10, 10]], dtype=torch.float64),
        "scores": torch.tensor([0.9], dtype=torch.float64),
    }

    tracker.update(prediction)
    tracks = tracker.update(prediction)

    assert tracks[0].box.dtype == torch.float64
    assert tracks[0].box.device == prediction["boxes"].device


def test_reset_clears_tracker_state() -> None:
    tracker = ByteTrack(
        ByteTrackConfig(
            track_activation_threshold=0.5,
            high_conf_det_threshold=0.5,
        )
    )

    tracker.update(_detections([[0, 0, 10, 10]], [0.9]))
    assert tracker.update(_detections([[0, 0, 10, 10]], [0.9]))

    tracker.reset()

    assert tracker.update(_detections([[20, 20, 30, 30]], [0.9])) == []


def test_build_tracker_constructs_bytetrack_from_kwargs() -> None:
    tracker = build_tracker("bytetrack", track_activation_threshold=0.7)

    assert isinstance(tracker, ByteTrack)
    assert tracker.config.track_activation_threshold == pytest.approx(0.7)


def test_build_tracker_rejects_unknown_tracker() -> None:
    with pytest.raises(ValueError, match="Unsupported tracker"):
        build_tracker("regression")
