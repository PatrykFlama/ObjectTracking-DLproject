import os
from pathlib import Path
from typing import Literal

from pytorch_lightning.loggers import WandbLogger


def build_wandb_logger(
    project_root: Path,
    project_name: str,
    run_name: str,
    log_model: Literal["all"] | bool = "all",
) -> WandbLogger:
    artifacts_dir = project_root / "artifacts"
    wandb_dirs = {
        "WANDB_DIR": artifacts_dir / "wandb",
        "WANDB_CACHE_DIR": artifacts_dir / "wandb-cache",
        "WANDB_CONFIG_DIR": artifacts_dir / "wandb-config",
        "WANDB_DATA_DIR": artifacts_dir / "wandb-data",
    }

    for env_name, env_path in wandb_dirs.items():
        env_path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(env_name, str(env_path))

    return WandbLogger(
        project=project_name,
        name=run_name,
        log_model=log_model,
        save_dir=str(wandb_dirs["WANDB_DIR"]),
    )
