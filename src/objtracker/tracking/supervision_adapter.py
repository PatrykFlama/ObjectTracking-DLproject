from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from objtracker.tracking.types import Detections, Track

if TYPE_CHECKING:
    from torch import Tensor


def to_supervision_detections(detections: Detections) -> Any:
    import supervision as sv

    labels = detections.labels
    assert labels is not None
    return sv.Detections(
        xyxy=_to_numpy(detections.boxes, "float32"),
        confidence=_to_numpy(detections.scores, "float32"),
        class_id=_to_numpy(labels, "int64"),
    )


def to_tracks(
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
