from typing import Any, cast

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.nms import non_max_suppression

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
    ):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
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
        yolo = YOLO(str(weights_path if weights_path.exists() else weights_name))
        self.model = cast("Any", yolo.model)
        self.model.nc = num_classes

        for param in self.model.parameters():
            param.requires_grad = True

        self.model.train()
        self.val_map: MeanAveragePrecision = MeanAveragePrecision(
            box_format="xyxy",
            iou_type="bbox",
        )

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

    def _validation_map_predictions(self, images):
        was_training = self.model.training
        self.model.eval()
        try:
            outputs = self.model(images)
            detections = non_max_suppression(
                outputs,
                conf_thres=0.001,
                iou_thres=0.7,
                agnostic=self.num_classes == 1,
                max_det=300,
            )
        finally:
            self.model.train(was_training)

        map_predictions = []
        for detection in detections:
            labels = detection[:, 5].long()
            if self.num_classes == 1:
                labels = torch.zeros(
                    len(detection),
                    dtype=torch.long,
                    device=detection.device,
                )
            map_predictions.append(
                {
                    "boxes": detection[:, :4],
                    "scores": detection[:, 4],
                    "labels": labels,
                }
            )
        return map_predictions

    def _validation_map_targets(self, targets, batch_size, image_size):
        image_height, image_width = image_size
        scale = targets.new_tensor(
            [image_width, image_height, image_width, image_height]
        )
        map_targets = []

        for image_idx in range(batch_size):
            image_targets = targets[targets[:, 0].long() == image_idx]
            boxes = image_targets[:, 2:] * scale
            if boxes.numel() > 0:
                cx, cy, width, height = boxes.unbind(dim=-1)
                boxes = torch.stack(
                    (
                        cx - width / 2,
                        cy - height / 2,
                        cx + width / 2,
                        cy + height / 2,
                    ),
                    dim=-1,
                )
            else:
                boxes = targets.new_zeros((0, 4))
            map_targets.append({"boxes": boxes, "labels": image_targets[:, 1].long()})

        return map_targets

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
        self.val_map.update(
            self._validation_map_predictions(images),
            self._validation_map_targets(
                targets,
                len(images),
                tuple(images.shape[-2:]),
            ),
        )
        return total_loss

    def on_validation_epoch_end(self):
        result = cast("Any", self.val_map).compute()
        self.log("val_map", result["map"], prog_bar=False, sync_dist=True)
        self.log("val_map_50", result["map_50"], prog_bar=True, sync_dist=True)
        self.val_map.reset()

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return configure_adamw_with_optional_scheduler(
            self,
            self.optimizer_config,
            estimated_steps=getattr(self.trainer, "estimated_stepping_batches", None),
        )
