from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from objtracker.datasets.yolo_formater import YoloDataset, yolo_collate_fn

if TYPE_CHECKING:
    from pathlib import Path


class TestYoloDataset:
    """Tiny regression tests for YOLO MOT15 formatting."""

    def test_returns_resized_image_and_normalized_boxes(
        self, mot15_sequence_dir: Path
    ) -> None:
        dataset = YoloDataset(
            root_dir=str(mot15_sequence_dir),
            sequence="SEQ-01",
            image_size=32,
        )

        image, boxes = dataset[0]

        assert image.shape == (3, 32, 32)
        assert boxes == pytest.approx(torch.tensor([[0, 0.2, 0.2, 0.2, 0.2]]))

    def test_ignores_low_confidence_boxes(self, mot15_sequence_dir: Path) -> None:
        dataset = YoloDataset(root_dir=str(mot15_sequence_dir), sequence="SEQ-01")

        _, boxes = dataset[0]

        assert boxes.shape == (1, 5)


class TestYoloCollateFn:
    """Unit tests for the YOLO collate helper."""

    def test_adds_batch_index_to_targets(self) -> None:
        batch = [
            (torch.zeros(3, 8, 8), torch.tensor([[0, 0.1, 0.2, 0.3, 0.4]])),
            (torch.ones(3, 8, 8), torch.tensor([[0, 0.5, 0.6, 0.7, 0.8]])),
        ]

        images, targets = yolo_collate_fn(batch)

        assert images.shape == (2, 3, 8, 8)
        assert targets == pytest.approx(
            torch.tensor(
                [
                    [0, 0, 0.1, 0.2, 0.3, 0.4],
                    [1, 0, 0.5, 0.6, 0.7, 0.8],
                ]
            )
        )

    def test_empty_targets_return_empty_six_column_tensor(self) -> None:
        batch = [(torch.zeros(3, 8, 8), torch.zeros((0, 5)))]

        _, targets = yolo_collate_fn(batch)

        assert targets.shape == (0, 6)
