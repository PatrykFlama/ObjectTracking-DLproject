from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from objtracker.datasets.mot15 import MOT15DataModule
from objtracker.datasets.mot15_dataset import MOT15Dataset

if TYPE_CHECKING:
    from pathlib import Path


class TestMOT15Dataset:
    """Tiny regression tests for MOT15 frame loading."""

    def test_len_counts_good_confidence_frames(self, mot15_sequence_dir: Path) -> None:
        dataset = MOT15Dataset(root_dir=str(mot15_sequence_dir), sequence="SEQ-01")

        assert len(dataset) == 2
        assert dataset.frames == [1, 2]

    def test_returns_image_and_xyxy_target(self, mot15_sequence_dir: Path) -> None:
        dataset = MOT15Dataset(root_dir=str(mot15_sequence_dir), sequence="SEQ-01")

        image, target = dataset[0]

        assert image.shape == (50, 100, 3)
        assert target["boxes"] == pytest.approx(torch.tensor([[10, 5, 30, 15]]))
        assert target["track_ids"].tolist() == [7]
        assert target["image_id"].tolist() == [1]


class TestMOT15DataModule:
    """Unit tests for lightweight datamodule helpers."""

    def test_stores_batch_size(self) -> None:
        datamodule = MOT15DataModule(batch_size=8)

        assert datamodule.batch_size == 8

    def test_transfer_batch_to_device_moves_target_tensors(self) -> None:
        datamodule = MOT15DataModule()
        samples = torch.zeros(1, 3, 4, 4)
        targets = [{"boxes": torch.zeros(1, 4), "track_ids": torch.tensor([1])}]

        moved_samples, moved_targets = datamodule.transfer_batch_to_device(
            (samples, targets),
            torch.device("cpu"),
            dataloader_idx=0,
        )

        assert moved_samples.device.type == "cpu"
        assert moved_targets[0]["boxes"].device.type == "cpu"
        assert moved_targets[0]["track_ids"].device.type == "cpu"
