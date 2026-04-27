from typing import Any, cast

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG

from objtracker.models.optim import (
    OptimizerConfig,
    configure_adamw_with_optional_scheduler,
)


class YOLOLightning(pl.LightningModule):
    def __init__(
        self,
        model_size="nano",
        lr=1e-4,
        num_classes=1,
        weight_decay=1e-4,
        backbone_lr_mult=0.1,
        scheduler="none",
        warmup_steps=0,
        warmup_ratio=0.05,
        min_lr_ratio=0.05,
        use_param_groups=True,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.optimizer_config = OptimizerConfig(
            lr=lr,
            weight_decay=weight_decay,
            backbone_lr_mult=backbone_lr_mult,
            scheduler=scheduler,
            warmup_steps=warmup_steps,
            warmup_ratio=warmup_ratio,
            min_lr_ratio=min_lr_ratio,
            use_param_groups=use_param_groups,
        )

        size_map = {
            "n": "n",
            "nano": "n",
            "s": "s",
            "small": "s",
            "m": "m",
            "medium": "m",
            "l": "l",
            "large": "l",
            "x": "x",
            "xlarge": "x",
        }
        if model_size not in size_map:
            msg = f"Unsupported YOLO model size: {model_size}"
            raise ValueError(msg)
        size = size_map[model_size]

        yolo = YOLO(f"yolo11{size}.pt")
        self.model = cast("Any", yolo.model)
        self.model.nc = num_classes

        for param in self.model.parameters():
            param.requires_grad = True

        self.model.train()

    def forward(self, images):
        return self.model(images)

    def setup(self, stage=None):
        self.model.args = get_cfg(DEFAULT_CFG)
        self.model.criterion = self.model.init_criterion()

    def on_train_start(self):
        self.model.train()
        device = self.device
        c = self.model.criterion
        c.device = device
        for k, v in vars(c).items():
            if isinstance(v, torch.Tensor):
                setattr(c, k, v.to(device))
            elif isinstance(v, torch.nn.Module):
                v.to(device)
        detect = self.model.model[-1]
        if hasattr(detect, "stride") and detect.stride is not None:
            c.stride = detect.stride.to(device)

    def _criterion_to_device(self, device):
        c = self.model.criterion
        if c is None:
            return
        for name, val in vars(c).items():
            if isinstance(val, torch.Tensor):
                setattr(c, name, val.to(device))
        if hasattr(c, "assigner"):
            c.assigner.to(device)

    def on_validation_start(self):
        self._criterion_to_device(self.device)

    def on_train_epoch_start(self):
        self.model.train()

    def _compute_loss(self, images, targets):
        outputs = self.model(images)
        batch_dict = {
            "batch_idx": targets[:, 0],
            "cls": targets[:, 1],
            "bboxes": targets[:, 2:],
        }
        loss, _ = self.model.criterion(outputs, batch_dict)
        return loss

    def training_step(self, batch, batch_idx):
        images, targets = batch
        self.model.train()

        loss = self._compute_loss(images, targets)
        total_loss = loss.sum()

        self.log_dict(
            {
                "train_loss_box": loss[0],
                "train_loss_cls": loss[1],
                "train_loss_dfl": loss[2],
            },
            prog_bar=False,
        )
        self.log("train_loss", total_loss, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        self.model.train()

        loss = self._compute_loss(images, targets)
        total_loss = loss.sum()

        self.log_dict(
            {
                "val_loss_box": loss[0],
                "val_loss_cls": loss[1],
                "val_loss_dfl": loss[2],
            },
            prog_bar=False,
        )
        self.log("val_loss", total_loss, prog_bar=True)
        return total_loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return configure_adamw_with_optional_scheduler(
            self,
            self.optimizer_config,
            estimated_steps=getattr(self.trainer, "estimated_stepping_batches", None),
        )
