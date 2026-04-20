from typing import cast

import pytorch_lightning as pl
import rfdetr as rfd
import torch
from rfdetr.models.lwdetr import build_criterion_and_postprocessors


class RFDETRLightning(pl.LightningModule):
    def __init__(self, model_size="nano", lr=1e-4):
        super().__init__()
        self.lr = lr

        models = {
            "nano": rfd.RFDETRNano,
            "small": rfd.RFDETRSmall,
            "medium": rfd.RFDETRMedium,
        }
        self.rfdetr_model = models[model_size]()

        self.model_context = self.rfdetr_model.model
        self.model = self.model_context.model
        self.criterion, _ = build_criterion_and_postprocessors(self.model_context.args)

    def forward(self, images):
        return self.model(images)

    def _compute_loss(self, images, targets):
        outputs = self.model(images, targets)
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

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        images, targets = batch
        total_loss = self._compute_loss(images, targets)
        self.log("train_loss", total_loss, prog_bar=True)
        return total_loss

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        images, targets = batch
        total_loss = self._compute_loss(images, targets)
        self.log("val_loss", total_loss, prog_bar=True)
        return total_loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)