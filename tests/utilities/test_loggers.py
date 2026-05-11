from __future__ import annotations

import os
from unittest.mock import patch

from objtracker.utils import build_tensorboard_logger, build_wandb_logger


class TestBuildTensorboardLogger:
    """Unit tests for TensorBoard logger setup."""

    def test_creates_tensorboard_artifact_dir(self, tmp_path) -> None:
        with patch("objtracker.utils.tensorboard.TensorBoardLogger"):
            build_tensorboard_logger(
                project_root=tmp_path,
                project_name="objtracker",
                run_name="smoke",
            )

        assert (tmp_path / "artifacts" / "tensorboard").is_dir()

    def test_forwards_logger_arguments(self, tmp_path) -> None:
        with patch("objtracker.utils.tensorboard.TensorBoardLogger") as logger_cls:
            build_tensorboard_logger(
                project_root=tmp_path,
                project_name="objtracker",
                run_name="smoke",
            )

        logger_cls.assert_called_once_with(
            save_dir=str(tmp_path / "artifacts" / "tensorboard"),
            name="objtracker",
            version="smoke",
        )


class TestBuildWandbLogger:
    """Unit tests for Weights & Biases logger setup."""

    def test_creates_wandb_artifact_dirs(self, tmp_path, monkeypatch) -> None:
        for env_name in _WANDB_ENV_NAMES:
            monkeypatch.delenv(env_name, raising=False)

        with patch("objtracker.utils.wandb.WandbLogger"):
            build_wandb_logger(
                project_root=tmp_path,
                project_name="objtracker",
                run_name="smoke",
            )

        for dirname in ["wandb", "wandb-cache", "wandb-config", "wandb-data"]:
            assert (tmp_path / "artifacts" / dirname).is_dir()

    def test_sets_missing_wandb_env_vars(self, tmp_path, monkeypatch) -> None:
        for env_name in _WANDB_ENV_NAMES:
            monkeypatch.delenv(env_name, raising=False)

        with patch("objtracker.utils.wandb.WandbLogger"):
            build_wandb_logger(
                project_root=tmp_path,
                project_name="objtracker",
                run_name="smoke",
            )

        assert _wandb_env("WANDB_DIR") == str(tmp_path / "artifacts" / "wandb")
        assert _wandb_env("WANDB_CACHE_DIR") == str(
            tmp_path / "artifacts" / "wandb-cache"
        )
        assert _wandb_env("WANDB_CONFIG_DIR") == str(
            tmp_path / "artifacts" / "wandb-config"
        )
        assert _wandb_env("WANDB_DATA_DIR") == str(
            tmp_path / "artifacts" / "wandb-data"
        )

    def test_preserves_existing_wandb_env_vars(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("WANDB_DIR", "/existing/wandb")

        with patch("objtracker.utils.wandb.WandbLogger"):
            build_wandb_logger(
                project_root=tmp_path,
                project_name="objtracker",
                run_name="smoke",
            )

        assert _wandb_env("WANDB_DIR") == "/existing/wandb"

    def test_forwards_logger_arguments(self, tmp_path, monkeypatch) -> None:
        for env_name in _WANDB_ENV_NAMES:
            monkeypatch.delenv(env_name, raising=False)

        with patch("objtracker.utils.wandb.WandbLogger") as logger_cls:
            build_wandb_logger(
                project_root=tmp_path,
                project_name="objtracker",
                run_name="smoke",
                log_model=False,
            )

        logger_cls.assert_called_once_with(
            project="objtracker",
            name="smoke",
            log_model=False,
            save_dir=str(tmp_path / "artifacts" / "wandb"),
        )


class TestUtilsExports:
    """Structural tests for the public utility imports."""

    def test_logger_builders_are_importable_from_utils_package(self) -> None:
        assert callable(build_tensorboard_logger)
        assert callable(build_wandb_logger)


_WANDB_ENV_NAMES = [
    "WANDB_DIR",
    "WANDB_CACHE_DIR",
    "WANDB_CONFIG_DIR",
    "WANDB_DATA_DIR",
]


def _wandb_env(name: str) -> str:
    return os.environ[name]
