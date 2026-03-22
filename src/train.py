import torch
import torch.nn as nn


def main():
    model = nn.Linear(10, 1)
    x = torch.randn(4, 10)
    y = model(x)
    print("Output shape:", y.shape)


if __name__ == "__main__":
    main()
