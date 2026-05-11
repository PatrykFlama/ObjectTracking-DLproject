import torch
from torch import nn

from objtracker.models.optim import (
    OptimizerConfig,
    build_adamw_optimizer,
    build_warmup_cosine_scheduler,
)


class TinyDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
        self.head = nn.Linear(4, 2)


def test_build_adamw_optimizer_uses_tuned_parameter_groups():
    model = TinyDetector()

    optimizer = build_adamw_optimizer(
        model,
        OptimizerConfig(lr=1e-3, weight_decay=1e-2, backbone_lr_mult=0.1),
    )

    group_settings = {
        (group["lr"], group["weight_decay"]) for group in optimizer.param_groups
    }

    assert (1e-4, 1e-2) in group_settings
    assert (1e-4, 0.0) in group_settings
    assert (1e-3, 1e-2) in group_settings
    assert (1e-3, 0.0) in group_settings


def test_build_adamw_optimizer_can_match_baseline_single_group():
    model = TinyDetector()

    optimizer = build_adamw_optimizer(
        model,
        OptimizerConfig(lr=1e-3, weight_decay=0.0, use_param_groups=False),
    )

    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["lr"] == 1e-3
    assert optimizer.param_groups[0]["weight_decay"] == 0.0


def test_warmup_cosine_scheduler_warms_then_decays():
    model = TinyDetector()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0)
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        total_steps=10,
        warmup_steps=2,
        min_lr_ratio=0.1,
    )

    lrs = []
    for _ in range(10):
        optimizer.step()
        lrs.append(optimizer.param_groups[0]["lr"])
        scheduler.step()

    assert lrs[0] == 0.5
    assert lrs[1] == 1.0
    assert lrs[-1] < lrs[2]
    assert lrs[-1] >= 0.1
