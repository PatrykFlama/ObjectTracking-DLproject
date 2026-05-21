from typing import Any, cast

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG

from objtracker.metrics.mean_average_precision import (
    build_mean_average_precision,
    compute_mean_average_precision,
    update_mean_average_precision,
    yolo_outputs_to_map_predictions,
    yolo_targets_to_map_targets,
)
from objtracker.models.optim import (
    OptimizerConfig,
    configure_adamw_with_optional_scheduler,
)
from objtracker.paths import CHECKPOINTS_DIR


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
        confidence_threshold=0.4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
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

        weights_name = f"yolo11{size}.pt"
        weights_path = CHECKPOINTS_DIR / weights_name
        self.yolo = YOLO(str(weights_path if weights_path.exists() else weights_name))
        self.model = cast("Any", self.yolo.model)
        self.model.nc = num_classes

        for param in self.model.parameters():
            param.requires_grad = True

        self.model.train()
        self.val_map = build_mean_average_precision(box_format="xyxy")

    def forward(self, images):
        return self.model(images)

    def detect(self, frame: Any):
        from objtracker.tracking.types import Detections

        result = cast(
            "Any",
            self.yolo.predict(
                frame,
                conf=self.confidence_threshold,
                verbose=False,
            )[0],
        )
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return Detections(
                boxes=torch.empty((0, 4), device=self.device),
                scores=torch.empty(0, device=self.device),
            )

        return Detections(
            boxes=boxes.xyxy.to(device=self.device),
            scores=boxes.conf.to(device=self.device),
            labels=boxes.cls.to(device=self.device, dtype=torch.long),
        )

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

    def _batch_dict(self, targets):
        return {
            "batch_idx": targets[:, 0],
            "cls": targets[:, 1],
            "bboxes": targets[:, 2:],
        }

    def _loss_from_outputs(self, outputs, targets):
        loss, _ = self.model.criterion(outputs, self._batch_dict(targets))
        return loss

    def _compute_loss(self, images, targets):
        return self._loss_from_outputs(self.model(images), targets)

    def _validation_map_predictions(self, outputs):
        return yolo_outputs_to_map_predictions(outputs, self.num_classes)

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
        self.model.eval()

        outputs = self.model(images)
        loss = self._loss_from_outputs(outputs, targets)
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
        update_mean_average_precision(
            self.val_map,
            self._validation_map_predictions(outputs),
            yolo_targets_to_map_targets(
                targets,
                len(images),
                tuple(images.shape[-2:]),
            ),
        )
        return total_loss

    def on_validation_epoch_end(self):
        result = compute_mean_average_precision(self.val_map)
        self.log("val_map", result["map"], prog_bar=False, sync_dist=True)
        self.log("val_map_50", result["map_50"], prog_bar=True, sync_dist=True)
        self.val_map.reset()

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return configure_adamw_with_optional_scheduler(
            self,
            self.optimizer_config,
            estimated_steps=getattr(self.trainer, "estimated_stepping_batches", None),
        )
