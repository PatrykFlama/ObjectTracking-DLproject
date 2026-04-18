from pathlib import Path

import kagglehub

from objtracker.datasets.coco_dataset import CocoDataset
from objtracker.datasets.mot15 import MOT15DataModule


class CocoDataModule(MOT15DataModule):
    def setup(self, stage=None):
        """Initialize datasets from the cached MOT15 archive."""
        cached_path = kagglehub.dataset_download(self.dataset_repo)

        base_dir = Path(cached_path)
        mot15_train = base_dir / "MOT15" / "train"
        train_path = mot15_train if mot15_train.exists() else base_dir / "train"

        if stage in ("fit", None):
            self.train_dataset = CocoDataset(
                root_dir=str(train_path), sequence="ADL-Rundle-6"
            )
