from __future__ import annotations

import torch

from objtracker.metrics.mean_average_precision import (
    rfdetr_outputs_to_map_predictions,
    yolo_targets_to_map_targets,
)


def test_yolo_targets_to_map_targets_converts_normalized_cxcywh_to_xyxy() -> None:
    targets = torch.tensor([[0, 0, 0.5, 0.5, 0.2, 0.4]])

    map_targets = yolo_targets_to_map_targets(
        targets,
        batch_size=1,
        image_size=(100, 200),
    )

    assert torch.allclose(
        map_targets[0]["boxes"],
        torch.tensor([[80.0, 30.0, 120.0, 70.0]]),
    )
    assert map_targets[0]["labels"].tolist() == [0]


def test_rfdetr_outputs_to_map_predictions_uses_single_class_labels() -> None:
    outputs = {
        "pred_logits": torch.tensor([[[0.1], [0.9]]]),
        "pred_boxes": torch.tensor([[[0.5, 0.5, 0.2, 0.4], [0.3, 0.3, 0.1, 0.1]]]),
    }

    predictions = rfdetr_outputs_to_map_predictions(outputs, num_classes=1)

    assert predictions[0]["labels"].tolist() == [0, 0]
    assert torch.equal(predictions[0]["boxes"], outputs["pred_boxes"][0])
