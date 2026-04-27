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
    parser = argparse.ArgumentParser(
        description="Train object detection model on MOT15"
    )
    parser.add_argument("--model", choices=["rfdetr", "yolo"], default="rfdetr")
    parser.add_argument(
        "--model-size",
        default="nano",
        help="nano/small/medium (rfdetr) or n/s/m/l/x (yolo)",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--backbone-lr-mult", type=float, default=None)
    parser.add_argument("--scheduler", choices=["none", "cosine"], default=None)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--warmup-ratio", type=float, default=None)
    parser.add_argument("--min-lr-ratio", type=float, default=None)
    parser.add_argument(
        "--training-profile",
        choices=["baseline", "tuned"],
        default="baseline",
        help=(
            "baseline keeps the original AdamW setup; tuned enables decay groups "
            "and cosine warmup"
        ),
    )
    parser.add_argument("--gradient-clip-val", type=float, default=None)
    parser.add_argument("--accumulate-grad-batches", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project-name", default="MOT15-Tracking")
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def build_optimizer_kwargs(args):
    defaults = {
        "baseline": {
            "weight_decay": 0.0,
            "backbone_lr_mult": 1.0,
            "scheduler": "none",
            "warmup_ratio": 0.0,
            "min_lr_ratio": 1.0,
            "use_param_groups": False,
        },
        "tuned": {
            "weight_decay": 1e-4,
            "backbone_lr_mult": 0.1,
            "scheduler": "cosine",
            "warmup_ratio": 0.05,
            "min_lr_ratio": 0.05,
            "use_param_groups": True,
        },
    }[args.training_profile]

    return {
        "weight_decay": (
            args.weight_decay
            if args.weight_decay is not None
            else defaults["weight_decay"]
        ),
        "backbone_lr_mult": (
            args.backbone_lr_mult
            if args.backbone_lr_mult is not None
            else defaults["backbone_lr_mult"]
        ),
        "scheduler": args.scheduler or defaults["scheduler"],
        "warmup_steps": args.warmup_steps,
        "warmup_ratio": (
            args.warmup_ratio
            if args.warmup_ratio is not None
            else defaults["warmup_ratio"]
        ),
        "min_lr_ratio": (
            args.min_lr_ratio
            if args.min_lr_ratio is not None
            else defaults["min_lr_ratio"]
        ),
        "use_param_groups": defaults["use_param_groups"],
    }


if __name__ == "__main__":
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)
    project_root = Path(__file__).resolve().parents[2]
    optimizer_kwargs = build_optimizer_kwargs(args)

    print("Initializing Model...")
    if args.model == "rfdetr":
        model = RFDETRLightning(
            model_size=args.model_size,
            lr=args.lr,
            **optimizer_kwargs,
        )
        data_module = CocoDataModule(
            batch_size=args.batch_size,
            image_size=model.model_context.resolution,
        )
    elif args.model == "yolo":
        model = YOLOLightning(
            model_size=args.model_size,
            lr=args.lr,
            **optimizer_kwargs,
        )
        data_module = YoloDataModule(
            batch_size=args.batch_size,
            image_size=640,
        )

    print("Initializing Data Pipeline...")

    print("Initializing Weights & Biases...")
    project_name = args.project_name
    run_name = args.run_name or (
        f"{args.model.upper()}-{args.model_size}-{args.training_profile}_"
        + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
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
        num_sanity_val_steps=0,
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.accumulate_grad_batches,
    )

    trainer.fit(model, datamodule=data_module)
