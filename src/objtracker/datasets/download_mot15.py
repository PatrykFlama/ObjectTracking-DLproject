from pathlib import Path

import kagglehub
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from objtracker.datasets import MOT15Dataset


class MOT15DataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 4):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_repo = "mdraselsarker/mot15-challenge-dataset"

    def prepare_data(self):
        """Download MOT15 to the local Kaggle cache if needed."""
        print("Checking Kaggle cache for MOT15...")
        kagglehub.dataset_download(self.dataset_repo)

    def setup(self, stage=None):
        """Initialize datasets from the cached MOT15 archive."""
        cached_path = kagglehub.dataset_download(self.dataset_repo)

        base_dir = Path(cached_path)
        mot15_train = base_dir / "MOT15" / "train"
        train_path = (
            mot15_train if mot15_train.exists() else base_dir / "train"
        )

        if stage in ("fit", None):
            self.train_dataset = MOT15Dataset(
                root_dir=str(train_path), sequence="ADL-Rundle-6"
            )

    def train_dataloader(self):
        """Create the training data loader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=lambda batch: tuple(zip(*batch)),
            num_workers=4,
        )
