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
        # RF-DETR variant objects wrap the actual torch module under .model.model.
        self.model_context = self.rfdetr_model.model
        self.model = self.model_context.model
        self.criterion, _ = build_criterion_and_postprocessors(self.model_context.args)

    def forward(self, images):
        return self.model(images)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        images, targets = batch

        outputs = self.model(images, targets)
        loss_dict = self.criterion(outputs, targets)
        if not isinstance(loss_dict, dict):
            msg = "Expected model to return a dict of loss tensors during training"
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

        tensor_losses = cast("list[torch.Tensor]", loss_values)

        total_loss = torch.stack(tensor_losses).sum()

        self.log("train_loss", total_loss, prog_bar=True)

        return total_loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
