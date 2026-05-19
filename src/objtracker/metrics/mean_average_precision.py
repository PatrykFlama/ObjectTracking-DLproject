from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor
from torchmetrics.detection.mean_ap import MeanAveragePrecision

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any, Literal


def build_mean_average_precision(
    box_format: Literal["xyxy", "xywh", "cxcywh"],
) -> MeanAveragePrecision:
    return MeanAveragePrecision(box_format=box_format, iou_type="bbox")


def compute_mean_average_precision(metric: MeanAveragePrecision) -> dict[str, Tensor]:
    return cast("Any", metric).compute()


def update_mean_average_precision(
    metric: MeanAveragePrecision,
    predictions: list[dict[str, Tensor]],
    targets: list[dict[str, Tensor]],
) -> None:
    cast("Any", metric).update(predictions, targets)


def yolo_detections_to_map_predictions(
    detections: Sequence[Tensor],
    num_classes: int,
) -> list[dict[str, Tensor]]:
    map_predictions = []

    for detection in detections:
        labels = detection[:, 5].long()
        if num_classes == 1:
            labels = torch.zeros(
                len(detection),
                dtype=torch.long,
                device=detection.device,
            )
        map_predictions.append(
            {
                "boxes": detection[:, :4],
                "scores": detection[:, 4],
                "labels": labels,
            }
        )

    return map_predictions


def yolo_targets_to_map_targets(
    targets: Tensor,
    batch_size: int,
    image_size: tuple[int, int],
) -> list[dict[str, Tensor]]:
    image_height, image_width = image_size
    scale = targets.new_tensor([image_width, image_height, image_width, image_height])
    map_targets = []

    for image_idx in range(batch_size):
        image_targets = targets[targets[:, 0].long() == image_idx]
        boxes = _cxcywh_to_xyxy(image_targets[:, 2:] * scale)
        map_targets.append({"boxes": boxes, "labels": image_targets[:, 1].long()})

    return map_targets


def rfdetr_outputs_to_map_predictions(
    outputs: dict[str, Tensor],
    num_classes: int,
) -> list[dict[str, Tensor]]:
    scores, labels = outputs["pred_logits"].sigmoid().max(dim=-1)
    if num_classes == 1:
        labels = torch.zeros_like(labels)

    return [
        {
            "boxes": boxes,
            "scores": image_scores,
            "labels": image_labels.long(),
        }
        for boxes, image_scores, image_labels in zip(
            outputs["pred_boxes"], scores, labels
        )
    ]


def rfdetr_targets_to_map_targets(
    targets: Sequence[dict[str, Tensor]],
) -> list[dict[str, Tensor]]:
    return [
        {
            "boxes": target["boxes"],
            "labels": target["labels"].long(),
        }
        for target in targets
    ]


def _cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    if boxes.numel() == 0:
        return boxes.new_zeros((0, 4))

    cx, cy, width, height = boxes.unbind(dim=-1)
    return torch.stack(
        (
            cx - width / 2,
            cy - height / 2,
            cx + width / 2,
            cy + height / 2,
        ),
        dim=-1,
    )
