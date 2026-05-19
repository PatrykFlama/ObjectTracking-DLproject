from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
TRAIN_MODULE = "objtracker.cli.train"


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open() as config_file:
        return yaml.safe_load(config_file)


def build_train_command(exp: dict[str, Any], cfg: dict[str, Any]) -> list[str]:
    command = [
        sys.executable,
        "-m",
        TRAIN_MODULE,
        "--model",
        "rfdetr",
        "--model-size",
        str(exp["model_size"]),
        "--epochs",
        str(exp["epochs"]),
        "--batch-size",
        str(exp.get("batch_size", 4)),
        "--lr",
        str(exp["lr"]),
        "--seed",
        str(cfg["seed"]),
        "--project-name",
        str(cfg["wandb"]["project"]),
        "--run-name",
        str(exp["name"]),
    ]

    grad_accum_steps = exp.get("grad_accum_steps")
    if grad_accum_steps is not None:
        command.extend(["--accumulate-grad-batches", str(grad_accum_steps)])

    training_profile = exp.get("training_profile")
    if training_profile is not None:
        command.extend(["--training-profile", str(training_profile)])

    return command


def select_experiments(
    experiments: list[dict[str, Any]],
    selected_name: str | None,
) -> list[dict[str, Any]]:
    if selected_name is None:
        return experiments

    selected = [exp for exp in experiments if exp["name"] == selected_name]
    if not selected:
        msg = f"No experiment named {selected_name!r} found in config."
        raise ValueError(msg)
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RF-DETR experiments")
    parser.add_argument(
        "--config",
        type=Path,
        default=EXPERIMENT_DIR / "exp-config.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--experiment",
        default=None,
        help="Run only this named experiment",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List configured experiments and exit",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config
    if not config_path.is_absolute():
        config_path = EXPERIMENT_DIR / config_path

    cfg = load_config(config_path)
    experiments = select_experiments(cfg["experiments"], args.experiment)

    if args.list:
        for exp in experiments:
            print(
                f"{exp['name']}: model=rfdetr size={exp['model_size']} "
                f"epochs={exp['epochs']} lr={exp['lr']}"
            )
        return

    print(
        f"[Train] Running {len(experiments)} experiment(s)"
        f" | W&B project: {cfg['wandb']['project']}"
    )

    for exp in experiments:
        command = build_train_command(exp, cfg)
        print(f"\n[Experiment] {exp['name']}")
        print(shlex.join(command))
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)

    print("\n[Train] All experiments completed.")


if __name__ == "__main__":
    main()
