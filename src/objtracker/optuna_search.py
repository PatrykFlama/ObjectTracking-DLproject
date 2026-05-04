from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if __package__ is None or __package__ == "":
    src_root = Path(__file__).resolve().parents[1]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

import optuna
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from objtracker.train import build_model_and_data, namespace_from_args

if TYPE_CHECKING:
    from argparse import Namespace


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(
        description="Run Optuna hyperparameter search for MOT15 training."
    )
    parser.add_argument("--model", choices=["rfdetr", "yolo"], default="rfdetr")
    parser.add_argument(
        "--model-size",
        default="nano",
        help="nano/small/medium (rfdetr) or n/s/m/l/x (yolo)",
    )
    parser.add_argument(
        "--training-profile",
        choices=["baseline", "tuned"],
        default="tuned",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--study-name", default="objtracker-optuna")
    parser.add_argument("--storage", default=None)
    parser.add_argument("--project-name", default="MOT15-Tracking-Optuna")
    parser.add_argument("--metric", default="val_loss")
    parser.add_argument("--gradient-clip-val", type=float, default=None)
    parser.add_argument("--accumulate-grad-batches", type=int, default=1)
    parser.add_argument("--prune", action="store_true")
    return parser.parse_args()


def suggest_optimizer_kwargs(trial: optuna.Trial, training_profile: str) -> dict:
    scheduler = trial.suggest_categorical("scheduler", ["none", "cosine"])
    warmup_ratio = 0.0
    min_lr_ratio = 1.0
    if scheduler == "cosine":
        warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.15)
        min_lr_ratio = trial.suggest_float("min_lr_ratio", 0.01, 0.25, log=True)

    return {
        "lr": trial.suggest_float("lr", 1e-6, 1e-3, log=True),
        "weight_decay": trial.suggest_float(
            "weight_decay",
            1e-6,
            1e-2,
            log=True,
        ),
        "backbone_lr_mult": trial.suggest_float(
            "backbone_lr_mult",
            0.05,
            1.0,
            log=True,
        ),
        "scheduler": scheduler,
        "warmup_steps": 0,
        "warmup_ratio": warmup_ratio,
        "min_lr_ratio": min_lr_ratio,
        "training_profile": training_profile,
    }


class PyTorchLightningPruningCallback(pl.Callback):
    def __init__(self, trial: optuna.Trial, monitor: str):
        self.trial = trial
        self.monitor = monitor

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        metric = trainer.callback_metrics.get(self.monitor)
        if metric is None:
            return
        self.trial.report(float(metric), step=trainer.current_epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned


def objective(trial: optuna.Trial, args: Namespace) -> float:
    pl.seed_everything(args.seed + trial.number, workers=True)
    trial_kwargs = suggest_optimizer_kwargs(trial, args.training_profile)
    trial_args = namespace_from_args(
        args,
        **trial_kwargs,
        run_name=f"{args.study_name}-trial-{trial.number:03d}",
    )
    model, data_module = build_model_and_data(trial_args)

    callbacks: list[pl.Callback] = [
        EarlyStopping(monitor=args.metric, mode="min", patience=3)
    ]
    if args.prune:
        callbacks.append(PyTorchLightningPruningCallback(trial, args.metric))

    logger = TensorBoardLogger(
        save_dir=str(Path("logs") / args.project_name),
        name=args.study_name,
        version=f"trial-{trial.number:03d}",
    )
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        log_every_n_steps=1,
        logger=logger,
        callbacks=callbacks,
        num_sanity_val_steps=0,
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.accumulate_grad_batches,
        enable_checkpointing=False,
    )
    trainer.fit(model, datamodule=data_module)

    metric = trainer.callback_metrics.get(args.metric)
    if metric is None:
        msg = f"Metric {args.metric!r} was not logged by the trainer."
        raise RuntimeError(msg)
    return float(metric)


def main() -> None:
    args = parse_args()
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="minimize",
        load_if_exists=True,
    )
    study.optimize(
        lambda trial: objective(trial, args),
        n_trials=args.n_trials,
        timeout=args.timeout,
    )

    print(f"Best trial: {study.best_trial.number}")
    print(f"Best {args.metric}: {study.best_value}")
    print("Best params:")
    for name, value in study.best_params.items():
        print(f"  {name}: {value}")


if __name__ == "__main__":
    main()
