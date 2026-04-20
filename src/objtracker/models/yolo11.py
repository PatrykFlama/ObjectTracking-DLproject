import pytorch_lightning as pl
import torch
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG


class YOLOLightning(pl.LightningModule):
    def __init__(self, model_size="nano", lr=1e-4, num_classes=1):
        super().__init__()
        self.lr = lr

        size_map = {
            "nano":   "n",
            "small":  "s",
            "medium": "m",
            "large":  "l",
            "xlarge": "x",
        }
        size = size_map[model_size]

        yolo = YOLO(f"yolo11{size}.pt")
        self.model = yolo.model
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
        if hasattr(detect, 'stride') and detect.stride is not None:
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
            "cls":       targets[:, 1],
            "bboxes":    targets[:, 2:],
        }
        loss, _ = self.model.criterion(outputs, batch_dict)
        return loss

    def training_step(self, batch, batch_idx):
        images, targets = batch
        self.model.train()

        loss = self._compute_loss(images, targets)
        total_loss = loss.sum()

        self.log_dict({
            "train_loss_box": loss[0],
            "train_loss_cls": loss[1],
            "train_loss_dfl": loss[2],
        }, prog_bar=False)
        self.log("train_loss", total_loss, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        self.model.train()

        loss = self._compute_loss(images, targets)
        total_loss = loss.sum()

        self.log_dict({
            "val_loss_box": loss[0],
            "val_loss_cls": loss[1],
            "val_loss_dfl": loss[2],
        }, prog_bar=False)
        self.log("val_loss", total_loss, prog_bar=True)
        return total_loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)