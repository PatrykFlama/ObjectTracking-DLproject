from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from objtracker.models.rf_detr import RFDETRLightning
from objtracker.models.yolo11 import YOLOLightning

if TYPE_CHECKING:
    from objtracker.models.base import MultiObjectDetector


@dataclass(frozen=True)
class DetectorRegistration:
    detector_cls: type


DETECTORS = {
    "rfdetr": DetectorRegistration(RFDETRLightning),
    "yolo": DetectorRegistration(YOLOLightning),
}


def build_detector(name: str, **kwargs: Any) -> MultiObjectDetector:
    registration = DETECTORS.get(name)
    if registration is None:
        msg = f"Unsupported detector: {name}"
        raise ValueError(msg)
    return cast("MultiObjectDetector", registration.detector_cls(**kwargs))
