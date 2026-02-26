import torch
import torch.nn as nn


def test_forward_pass():
    model = nn.Linear(10, 1)
    x = torch.randn(2, 10)
    y = model(x)
    assert y.shape == (2, 1)
