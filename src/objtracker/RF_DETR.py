import os
import random
from typing import Any

import numpy as np
import rfdetr as rfd
import torch

RNG = np.random.default_rng()


def set_seed(seed: int) -> None:
    global RNG
    random.seed(seed)
    RNG = np.random.default_rng(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


class RFDETRTrainer:
    def __init__(
        self,
        model_size: str = "nano",
        seed: int = 42,
    ):
        set_seed(seed)
        self.seed = seed

        models = {
            "nano": rfd.RFDETRNano,
            "small": rfd.RFDETRSmall,
            "medium": rfd.RFDETRMedium,
        }

        if model_size not in models:
            allowed = ", ".join(models.keys())
            raise ValueError(
                f"Invalid model_size {model_size!r}. "
                f"Allowed values are: {allowed}."
            )

        self.model_size = model_size
        self.model: Any = models[model_size]()
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

    def train(
        self,
        dataset_dir: str,
        epochs: int = 10,
        batch_size: int = 4,
        lr: float = 1e-4,
        grad_accum_steps: int = 4,
        output_dir: str = "runs/train",
        wandb_project: str | None = None,
        wandb_run: str | None = None,
        wandb_entity: str | None = None,
    ) -> None:

        print(
            f"MODEL TRAINING"
            f"device={self.device} | seed={self.seed}"
            f"model={self.model_size} | epochs={epochs} | lr={lr}"
        )

        self.model.train(
            dataset_dir=dataset_dir,
            epochs=epochs,
            batch_size=batch_size,
            grad_accum_steps=grad_accum_steps,
            lr=lr,
            output_dir=output_dir,
            wandb=wandb_project is not None,
            project=wandb_project,
            run=wandb_run,
        )

    def predict(self, image_path: str):
        return self.model.predict(image_path)

    def predict_batch(self, image_paths: list[str]) -> list:
        return [self.predict(p) for p in image_paths]

    def save(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> None:
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
