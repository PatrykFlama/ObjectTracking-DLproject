from pathlib import Path

import kagglehub
from torch.utils.data import ConcatDataset, DataLoader

from objtracker.datasets.mot15 import MOT15DataModule
from objtracker.datasets.yolo_formater import YoloDataset, yolo_collate_fn

# MOT15 train sequences split into train / val
TRAIN_SEQUENCES = ["ADL-Rundle-6", "ETH-Bahnhof", "KITTI-17", "PETS09-S2L1", "TUD-Stadtmitte"]
VAL_SEQUENCES   = ["ADL-Rundle-8", "ETH-Sunnyday", "KITTI-13"]


class YoloDataModule(MOT15DataModule):
    def __init__(self, batch_size: int = 4, image_size: int = 640):
        super().__init__(batch_size=batch_size)
        self.image_size = image_size

    def setup(self, stage=None):
        cached_path = kagglehub.dataset_download(self.dataset_repo)

        base_dir = Path(cached_path)
        mot15_train = base_dir / "MOT15" / "train"
        train_path = mot15_train if mot15_train.exists() else base_dir / "train"

        if stage in ("fit", None):
            self.train_dataset = ConcatDataset([
                YoloDataset(root_dir=str(train_path), sequence=seq, image_size=self.image_size)
                for seq in TRAIN_SEQUENCES
                if (train_path / seq).exists()
            ])
            self.val_dataset = ConcatDataset([
                YoloDataset(root_dir=str(train_path), sequence=seq, image_size=self.image_size)
                for seq in VAL_SEQUENCES
                if (train_path / seq).exists()
            ])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=yolo_collate_fn,
            num_workers=4,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=yolo_collate_fn,
            num_workers=4,
            persistent_workers=True,
        )

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        images, targets = batch
        return images.to(device), targets.to(device)