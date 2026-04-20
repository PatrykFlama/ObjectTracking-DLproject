import argparse
import sys
from datetime import datetime
from pathlib import Path

if __package__ is None or __package__ == "":
    src_root = Path(__file__).resolve().parents[1]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

import pytorch_lightning as pl

from objtracker.datasets.coco import CocoDataModule
from objtracker.datasets.yolo import YoloDataModule
from objtracker.models.rf_detr import RFDETRLightning
from objtracker.models.yolo11 import YOLOLightning
from objtracker.utils import build_tensorboard_logger, build_wandb_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Train object detection model on MOT15")
    parser.add_argument("--model", choices=["rfdetr", "yolo"], default="rfdetr")
    parser.add_argument("--model-size", default="nano", help="nano/small/medium (rfdetr) or n/s/m/l/x (yolo)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]

    print("Initializing Model...")
    if args.model == "rfdetr":
        model = RFDETRLightning(model_size=args.model_size, lr=args.lr)
        data_module = CocoDataModule(
            batch_size=args.batch_size,
            image_size=model.model_context.resolution,
        )
    elif args.model == "yolo":
        model = YOLOLightning(model_size=args.model_size, lr=args.lr)
        data_module = YoloDataModule(
            batch_size=args.batch_size,
            image_size=640,
        )

    print("Initializing Data Pipeline...")

    print("Initializing Weights & Biases...")
    project_name = "MOT15-Tracking"
    run_name = f"{args.model.upper()}-{args.model_size}-baseline_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
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
        max_epochs=args.epochs,
        accelerator="auto",
        log_every_n_steps=1,
        logger=[wandb_logger, tensorboard_logger],
        num_sanity_val_steps = 0,
    )

    trainer.fit(model, datamodule=data_module)  