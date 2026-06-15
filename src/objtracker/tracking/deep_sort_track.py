from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from objtracker.tracking.types import Detections, as_detections
from objtracker.tracking.types import Track as ProjectTrack

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch import Tensor


@dataclass(frozen=True)
class DeepSORTTrackConfig:
    """Configuration passed through to `deep_sort_realtime.DeepSort`."""

    max_age: int = 30
    n_init: int = 3
    nms_max_overlap: float = 1.0
    max_cosine_distance: float = 0.2
    nn_budget: int | None = None
    embedder: str = "mobilenet"
    half: bool = True
    bgr: bool = True
    embedder_gpu: bool = True


class DeepSORTTrack:
    """Adapter for the deep-sort-realtime library."""

    def __init__(self, config: DeepSORTTrackConfig | None = None):
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
        except ImportError as e:
            raise ImportError(
                "Please install deep-sort-realtime via `uv add deep-sort-realtime`"
            ) from e

        self.config = config or DeepSORTTrackConfig()

        self._tracker = DeepSort(**self.config.__dict__)

    def reset(self) -> None:
        """Clear all active tracks to start a new video sequence."""
        self._tracker.delete_all_tracks()

    def update(
        self, detections: Detections | Mapping[str, Tensor], frame: Any | None = None
    ) -> list[ProjectTrack]:
        if frame is None:
            raise ValueError(
                "DeepSORT requires the image frame to extract visual features."
            )

        frame_detections = as_detections(detections)

        boxes = frame_detections.boxes.detach().cpu().numpy()
        scores = frame_detections.scores.detach().cpu().numpy()
        labels = (
            frame_detections.labels.detach().cpu().numpy()
            if frame_detections.labels is not None
            else [0] * len(boxes)
        )

        raw_detections = []
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            raw_detections.append(([x1, y1, w, h], float(score), int(label)))

        ds_tracks = self._tracker.update_tracks(raw_detections, frame=frame)

        out_tracks = []
        device = frame_detections.boxes.device
        dtype = frame_detections.boxes.dtype

        for track in ds_tracks:
            if not track.is_confirmed():
                continue

            track_id = int(track.track_id)
            ltrb = track.to_ltrb()

            label = track.det_class if track.det_class is not None else 0
            score = track.det_conf if track.det_conf is not None else 0.0

            out_tracks.append(
                ProjectTrack(
                    track_id=track_id,
                    box=torch.as_tensor(ltrb, dtype=dtype, device=device),
                    score=float(score),
                    label=int(label),
                )
            )

        return out_tracks
