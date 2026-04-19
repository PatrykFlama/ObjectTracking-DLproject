import sys
from pathlib import Path

# Support direct execution: `uv run src/objtracker/train.py`.
if __package__ is None or __package__ == "":
    src_root = Path(__file__).resolve().parents[1]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from objtracker.datasets.coco import CocoDataModule
from objtracker.models.rf_detr import RFDETRLightning

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    artifacts_dir = project_root / "artifacts"
    tensorboard_dir = artifacts_dir / "tensorboard"
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    print("Initializing Model...")
    model = RFDETRLightning(model_size="nano", lr=1e-4)

    print("Initializing Data Pipeline...")
    data_module = CocoDataModule(
        batch_size=4,
        image_size=model.model_context.resolution,
    )

    print("Initializing Weights & Biases...")
    # Setup the WandB Logger
    wandb_logger = WandbLogger(
        project="MOT15-Tracking",  # Name of the project in your WandB dashboard
        name="RFDETR-nano-baseline",  # Name of this specific run
        log_model="all",  # Automatically saves your best model weights to the cloud!
    )
    tensorboard_logger = TensorBoardLogger(
        save_dir=str(tensorboard_dir),
        name="MOT15-Tracking",
        version="RFDETR-nano-baseline",
    )

    print("Starting Lightning Trainer...")
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="auto",
        log_every_n_steps=1,
        logger=[wandb_logger, tensorboard_logger],
    )

    trainer.fit(model, datamodule=data_module)
