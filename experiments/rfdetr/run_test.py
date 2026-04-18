import argparse
from pathlib import Path

import yaml

from objtracker.models.RF_DETR import RFDETRTrainer, set_seed


def load_config(config_path: Path) -> dict:
    with config_path.open() as f:
        return yaml.safe_load(f)


def run_experiment(exp: dict, cfg: dict, dataset_dir: Path) -> None:
    name: str = exp["name"]
    seed: int = cfg["seed"]
    epochs: int = int(exp["epochs"])
    batch_size_raw = exp.get("batch_size")
    batch_size: int = int(batch_size_raw) if batch_size_raw is not None else 4
    lr: float = float(exp["lr"])
    grad_accum_steps: int = int(exp.get("grad_accum_steps", 4))

    print(exp)
    print(f"\n{'=' * 60}")
    print(f"  Experiment: {name}")
    print(
        f"  seed={seed} | model={exp['model_size']} | "
        f"lr={lr} | epochs={epochs}"
    )
    print(f"{'=' * 60}\n")

    set_seed(seed)

    trainer = RFDETRTrainer(model_size=exp["model_size"], seed=seed)

    trainer.train(
        dataset_dir=str(dataset_dir),
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        grad_accum_steps=grad_accum_steps,
        output_dir=f"{cfg['output_dir']}/{name}",
        wandb_project=cfg["wandb"]["project"],
        wandb_run=name,
        wandb_entity=cfg["wandb"].get("entity"),
    )

    print(f"\n[Done] Experiment '{name}' finished.\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RF-DETR reproducible training"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config YAML (default: config.yaml)",
    )
    parser.add_argument(
        "--experiment",
        default=None,
        help="Run only this named experiment (default: run all)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[0]
    config_path = project_root / args.config
    cfg = load_config(config_path)

    dataset_dir = project_root / cfg["dataset"]["dir"]

    experiments = cfg["experiments"]
    if args.experiment:
        experiments = [e for e in experiments if e["name"] == args.experiment]
        if not experiments:
            raise ValueError(
                f"No experiment named '{args.experiment}' found in config."
            )

    print(
        f"[Train] Running {len(experiments)} experiment(s)"
        f" | W&B project: {cfg['wandb']['project']}"
    )

    for exp in experiments:
        run_experiment(exp, cfg, dataset_dir)

    print("\n[Train] All experiments completed.")


if __name__ == "__main__":
    main()
