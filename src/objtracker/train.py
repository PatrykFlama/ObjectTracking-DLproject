import sys
from datetime import datetime
from pathlib import Path

# Support direct execution: `uv run src/objtracker/train.py`.
if __package__ is None or __package__ == "":
    src_root = Path(__file__).resolve().parents[1]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

import pytorch_lightning as pl

from objtracker.datasets.coco import CocoDataModule
from objtracker.models.rf_detr import RFDETRLightning
from objtracker.utils import build_tensorboard_logger, build_wandb_logger

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]

    print("Initializing Model...")
    model = RFDETRLightning(model_size="nano", lr=1e-4)

    print("Initializing Data Pipeline...")
    data_module = CocoDataModule(
        batch_size=4,
        image_size=model.model_context.resolution,
    )

    print("Initializing Weights & Biases...")
    project_name = "MOT15-Tracking"
    run_name = "RFDETR-nano-baseline_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    wandb_logger = build_wandb_logger(
        project_root=project_root,
        project_name=project_name,
        run_name=run_name,
        log_model="all",
    )
    tensorboard_logger = build_tensorboard_logger(
        project_root=project_root,
        project_name=project_name,
        run_name=run_name,
    )

    print("Starting Lightning Trainer...")
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="auto",
        log_every_n_steps=1,
        logger=[wandb_logger, tensorboard_logger],
    )

    trainer.fit(model, datamodule=data_module)
