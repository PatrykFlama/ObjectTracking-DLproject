from pathlib import Path

import kagglehub
import pytorch_lightning as pl
from rfdetr.utilities.tensors import collate_fn
from torch.utils.data import DataLoader

from objtracker.datasets.mot15_dataset import MOT15Dataset


class MOT15DataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 4):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_repo = "mdraselsarker/mot15-challenge-dataset"

    def prepare_data(self):
        print("Checking Kaggle cache for MOT15...")
        kagglehub.dataset_download(self.dataset_repo)

    def setup(self, stage=None):
        cached_path = kagglehub.dataset_download(self.dataset_repo)

        base_dir = Path(cached_path)
        mot15_train = base_dir / "MOT15" / "train"
        train_path = mot15_train if mot15_train.exists() else base_dir / "train"

        if stage in ("fit", None):
            self.train_dataset = MOT15Dataset(
                root_dir=str(train_path), sequence="ADL-Rundle-6"
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
        )

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        samples, targets = batch
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        return samples, targets
