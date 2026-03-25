from pathlib import Path

from RF_DETR import RFDETRT

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    dataset_dir = project_root / "dataset"

    rfdetr = RFDETRT("nano")
    rfdetr.train(
        str(dataset_dir),
        epochs=10,
        batch_size=4,
    )
