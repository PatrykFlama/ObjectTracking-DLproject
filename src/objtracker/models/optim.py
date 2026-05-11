from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

if TYPE_CHECKING:
    from pytorch_lightning.utilities.types import (
        OptimizerLRScheduler,
        OptimizerLRSchedulerConfig,
    )
    from torch.optim import Optimizer


@dataclass(frozen=True)
class OptimizerConfig:
    lr: float = 1e-4
    weight_decay: float = 1e-4
    backbone_lr_mult: float = 0.1
    use_param_groups: bool = True
    scheduler: str = "none"
    warmup_steps: int = 0
    warmup_ratio: float = 0.05
    min_lr_ratio: float = 0.05


def _is_no_decay_param(name: str, param: nn.Parameter) -> bool:
    return param.ndim <= 1 or name.endswith(".bias") or ".norm" in name.lower()


def _is_backbone_param(name: str) -> bool:
    lower_name = name.lower()
    return any(key in lower_name for key in ("backbone", "encoder", "stem"))


def build_adamw_optimizer(
    module: nn.Module,
    config: OptimizerConfig,
) -> torch.optim.AdamW:
    if not config.use_param_groups:
        return torch.optim.AdamW(
            module.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

    groups: dict[tuple[float, float], list[nn.Parameter]] = {}
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        lr = (
            config.lr * config.backbone_lr_mult
            if _is_backbone_param(name)
            else config.lr
        )
        weight_decay = 0.0 if _is_no_decay_param(name, param) else config.weight_decay
        groups.setdefault((lr, weight_decay), []).append(param)

    return torch.optim.AdamW(
        [
            {"params": params, "lr": lr, "weight_decay": weight_decay}
            for (lr, weight_decay), params in groups.items()
        ],
        lr=config.lr,
        weight_decay=config.weight_decay,
    )


def build_warmup_cosine_scheduler(
    optimizer: Optimizer,
    total_steps: int,
    warmup_steps: int = 0,
    warmup_ratio: float = 0.05,
    min_lr_ratio: float = 0.05,
) -> LambdaLR:
    total_steps = max(1, total_steps)
    if warmup_steps <= 0:
        warmup_steps = int(total_steps * warmup_ratio)
    warmup_steps = min(max(0, warmup_steps), total_steps - 1)
    min_lr_ratio = min(max(min_lr_ratio, 0.0), 1.0)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)

        decay_steps = max(1, total_steps - warmup_steps)
        progress = min(max((step - warmup_steps) / decay_steps, 0.0), 1.0)
        cosine_ratio = 0.5 * (1.0 + cos(pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_ratio

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def configure_adamw_with_optional_scheduler(
    module: nn.Module,
    config: OptimizerConfig,
    estimated_steps: int | None,
) -> OptimizerLRScheduler:
    optimizer = build_adamw_optimizer(module, config)
    if config.scheduler == "none":
        return optimizer
    if config.scheduler != "cosine":
        msg = f"Unsupported scheduler: {config.scheduler}"
        raise ValueError(msg)

    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        total_steps=estimated_steps or 1,
        warmup_steps=config.warmup_steps,
        warmup_ratio=config.warmup_ratio,
        min_lr_ratio=config.min_lr_ratio,
    )
    optimizer_scheduler: OptimizerLRSchedulerConfig = {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        },
    }
    return optimizer_scheduler
