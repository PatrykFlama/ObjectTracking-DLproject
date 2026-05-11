from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
TRAIN_SCRIPT = PROJECT_ROOT / "src" / "objtracker" / "train.py"

CLI_FLAGS = {
    key: f"--{key.replace('_', '-')}"
    for key in [
        "model",
        "model_size",
        "training_profile",
        "lr",
        "weight_decay",
        "backbone_lr_mult",
        "scheduler",
        "warmup_steps",
        "warmup_ratio",
        "min_lr_ratio",
        "gradient_clip_val",
        "accumulate_grad_batches",
        "batch_size",
        "epochs",
        "seed",
        "project_name",
    ]
}


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("rb") as config_file:
        return tomllib.load(config_file)


def materialize_runs(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Merge each run with the defaults, so that each run dict is complete and
    can be directly used to build a train.py command."""
    defaults = config.get("defaults", {})
    runs = []
    for run in config["runs"]:
        merged = defaults | run
        merged["run_name"] = run["name"]
        runs.append(merged)
    return runs


def build_train_command(run: dict[str, Any]) -> list[str]:
    command = [sys.executable, str(TRAIN_SCRIPT)]
    for key, flag in CLI_FLAGS.items():
        if key in run:
            command.extend([flag, str(run[key])])
    command.extend(["--run-name", str(run["run_name"])])
    return command


def select_runs(
    runs: list[dict[str, Any]],
    selected_names: list[str] | None,
) -> list[dict[str, Any]]:
    if not selected_names:
        return runs

    selected = [run for run in runs if run["name"] in selected_names]
    missing = sorted(set(selected_names) - {run["name"] for run in selected})
    if missing:
        msg = f"Unknown experiment run(s): {', '.join(missing)}"
        raise ValueError(msg)
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MOT15 model improvement experiments"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=EXPERIMENT_DIR / "experiment.toml",
        help="Path to experiment TOML config",
    )
    parser.add_argument(
        "--run",
        action="append",
        dest="run_names",
        help="Run only this experiment name; pass multiple times for several runs",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List configured experiment runs and exit",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    runs = materialize_runs(config)

    if args.list:
        for run in runs:
            print(
                f"{run['name']}: model={run['model']} "
                f"size={run['model_size']} profile={run['training_profile']}"
            )
        return

    runs = select_runs(runs, args.run_names)
    for run in runs:
        command = build_train_command(run)
        print(f"\n[Experiment] {run['name']}")
        print(shlex.join(command))
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
