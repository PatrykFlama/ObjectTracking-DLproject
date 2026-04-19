from pathlib import Path

from pytorch_lightning.loggers import TensorBoardLogger


def build_tensorboard_logger(
    project_root: Path,
    project_name: str,
    run_name: str,
) -> TensorBoardLogger:
    tensorboard_dir = project_root / "artifacts" / "tensorboard"
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    return TensorBoardLogger(
        save_dir=str(tensorboard_dir),
        name=project_name,
        version=run_name,
    )
