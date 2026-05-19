from typing import Any, cast

import pytorch_lightning as pl
import rfdetr as rfd
import torch
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from rfdetr.models.lwdetr import build_criterion_and_postprocessors
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from objtracker.models.optim import (
    OptimizerConfig,
    configure_adamw_with_optional_scheduler,
)


class RFDETRLightning(pl.LightningModule):
    def __init__(
        self,
        model_size="nano",
        lr=1e-4,
        weight_decay=1e-4,
        backbone_lr_mult=0.1,
        scheduler="none",
        warmup_steps=0,
        warmup_ratio=0.05,
        min_lr_ratio=0.05,
        use_param_groups=True,
        resolution=640,
        num_classes=1,
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

        models = {
            "nano": rfd.RFDETRNano,
            "small": rfd.RFDETRSmall,
            "medium": rfd.RFDETRMedium,
        }
        if model_size not in models:
            msg = f"Unsupported RF-DETR model size: {model_size}"
            raise ValueError(msg)
        self.rfdetr_model = models[model_size](
            resolution=resolution,
            num_classes=num_classes,
        )

        self.model_context = self.rfdetr_model.model
        self.model = self.model_context.model
        self.criterion, _ = build_criterion_and_postprocessors(self.model_context.args)
        self.val_map: MeanAveragePrecision = MeanAveragePrecision(
            box_format="cxcywh",
            iou_type="bbox",
        )

    def forward(self, images):
        return self.model(images)

    def _loss_from_outputs(self, outputs, targets):
        loss_dict = self.criterion(outputs, targets)
        if not isinstance(loss_dict, dict):
            msg = "Expected model to return a dict of loss tensors"
            raise TypeError(msg)
        weight_dict = self.criterion.weight_dict
        loss_values = [
            loss_value * weight_dict[name]
            for name, loss_value in loss_dict.items()
            if name in weight_dict
        ]
        if not loss_values:
            msg = "No weighted loss terms were produced by criterion"
            raise RuntimeError(msg)
        return torch.stack(cast("list[torch.Tensor]", loss_values)).sum()

    def _compute_loss(self, images, targets):
        outputs = self.model(images, targets)
        return self._loss_from_outputs(outputs, targets)

    def _validation_map_predictions(self, outputs):
        scores, labels = outputs["pred_logits"].sigmoid().max(dim=-1)
        if self.num_classes == 1:
            labels = torch.zeros_like(labels)

        return [
            {
                "boxes": boxes,
                "scores": image_scores,
                "labels": image_labels.long(),
            }
            for boxes, image_scores, image_labels in zip(
                outputs["pred_boxes"], scores, labels
            )
        ]

    def _validation_map_targets(self, targets):
        return [
            {
                "boxes": target["boxes"],
                "labels": target["labels"].long(),
            }
            for target in targets
        ]

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        images, targets = batch
        total_loss = self._compute_loss(images, targets)
        self.log("train_loss", total_loss, prog_bar=True)
        return total_loss

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        images, targets = batch
        outputs = self.model(images, targets)
        total_loss = self._loss_from_outputs(outputs, targets)
        self.log("val_loss", total_loss, prog_bar=True)
        self.val_map.update(
            self._validation_map_predictions(outputs),
            self._validation_map_targets(targets),
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
