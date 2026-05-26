from __future__ import annotations

from typing import Any

import pytest
import torch

from objtracker.models.factory import (
    DETECTORS,
    DetectorRegistration,
    build_detector,
)


class FakeDetector:
    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold

    def detect(self, frame: Any) -> dict[str, torch.Tensor]:
        assert frame == "frame"
        return {
            "boxes": torch.tensor([[0, 0, 10, 10]], dtype=torch.float32),
            "scores": torch.tensor([self.confidence_threshold], dtype=torch.float32),
        }


def test_build_detector_builds_registered_detector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(DETECTORS, "fake", DetectorRegistration(FakeDetector))

    detector = build_detector("fake", confidence_threshold=0.8)

    assert isinstance(detector, FakeDetector)
    assert detector.detect("frame")["scores"].item() == pytest.approx(0.8)


def test_build_detector_rejects_unknown_detector() -> None:
    with pytest.raises(ValueError, match="Unsupported detector"):
        build_detector("unknown")
