from typing import cast

import pytorch_lightning as pl
import rfdetr as rfd
import torch


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
        self.model = self.rfdetr_model.model.model

    def forward(self, images):
        return self.model(images)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        images, targets = batch

        loss_dict = self.model(images, targets)
        if not isinstance(loss_dict, dict):
            msg = "Expected model to return a dict of loss tensors during training"
            raise TypeError(msg)
        loss_values = list(loss_dict.values())
        if not all(isinstance(loss_value, torch.Tensor) for loss_value in loss_values):
            msg = "Expected loss dict values to be torch.Tensor instances"
            raise TypeError(msg)

        tensor_losses = cast("list[torch.Tensor]", loss_values)

        total_loss = torch.stack(tensor_losses).sum()

        self.log("train_loss", total_loss, prog_bar=True)

        return total_loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
