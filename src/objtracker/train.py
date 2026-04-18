import sys
from pathlib import Path

# Support direct execution: `uv run src/objtracker/train.py`.
if __package__ is None or __package__ == "":
    src_root = Path(__file__).resolve().parents[1]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from objtracker.datasets.coco import CocoDataModule
from objtracker.models.rf_detr import RFDETRLightning

if __name__ == "__main__":
    print("Initializing Data Pipeline...")
    data_module = CocoDataModule(batch_size=4)

    print("Initializing Model...")
    model = RFDETRLightning(model_size="nano", lr=1e-4)

    print("Initializing Weights & Biases...")
    # Setup the WandB Logger
    wandb_logger = WandbLogger(
        project="MOT15-Tracking",  # Name of the project in your WandB dashboard
        name="RFDETR-nano-baseline",  # Name of this specific run
        log_model="all",  # Automatically saves your best model weights to the cloud!
    )

    print("Starting Lightning Trainer...")
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="auto",
        log_every_n_steps=1,
        logger=wandb_logger,  # Hand the logger to the Trainer
    )

    trainer.fit(model, datamodule=data_module)
