from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from torch import Tensor

from objtracker.tracking.types import Detections

type DetectionOutput = Detections | Mapping[str, Tensor]


class MultiObjectDetector(Protocol):
    """Common interface for single-frame object detectors."""

    def detect(self, frame: Any) -> DetectionOutput:
        """Return detections for one frame."""
        ...
