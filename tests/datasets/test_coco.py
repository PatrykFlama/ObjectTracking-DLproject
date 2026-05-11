from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from objtracker.datasets.coco_dataset import CocoDataset

if TYPE_CHECKING:
    from pathlib import Path


class TestCocoDataset:
    """Tiny regression tests for RF-DETR-style MOT15 formatting."""

    def test_returns_target_dict_with_normalized_boxes(
        self, mot15_sequence_dir: Path
    ) -> None:
        dataset = CocoDataset(
            root_dir=str(mot15_sequence_dir),
            sequence="SEQ-01",
            image_size=32,
        )

        image, target = dataset[0]

        assert image.shape == (3, 32, 32)
        assert target["boxes"] == pytest.approx(torch.tensor([[0.2, 0.2, 0.2, 0.2]]))
        assert target["labels"].tolist() == [0]
        assert target["track_ids"].tolist() == [7]
        assert target["image_id"].tolist() == [1]
