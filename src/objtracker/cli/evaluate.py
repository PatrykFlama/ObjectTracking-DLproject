import sys
from pathlib import Path

import torch
from tqdm import tqdm

if __package__ is None or __package__ == "":
    src_root = Path(__file__).resolve().parents[2]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

from torchmetrics.detection.mean_ap import MeanAveragePrecision

from objtracker.datasets.yolo import YoloDataModule
from objtracker.models.yolo11 import YOLOLightning
from objtracker.paths import CHECKPOINTS_DIR


def evaluate_yolo(checkpoint_path):
    print("Loading YOLO Model & Data...")

    model = YOLOLightning.load_from_checkpoint(checkpoint_path, map_location="cpu")
    model.eval()

    datamodule = YoloDataModule(batch_size=4)
    datamodule.setup()
    val_loader = datamodule.val_dataloader()

    metric = MeanAveragePrecision(box_format="cxcywh", iou_type="bbox")

    print("Evaluating Validation Set (This may take a few minutes)...")
    with torch.no_grad():
        for batch in tqdm(val_loader):
            images, targets = batch

            outputs = model(images)

            preds = []
            target_list = []

            for i in range(len(images)):
                img_targets = targets[targets[:, 0] == i]

                preds.append(
                    {
                        "boxes": outputs[0][i][:, :4],
                        "scores": outputs[0][i][:, 4],
                        "labels": torch.zeros(len(outputs[0][i]), dtype=torch.int),
                    }
                )

                target_list.append(
                    {
                        "boxes": img_targets[:, 2:],
                        "labels": torch.zeros(len(img_targets), dtype=torch.int),
                    }
                )

            metric.update(preds, target_list)  # type: ignore

    print("\nCalculating Final mAP...")
    result = metric.compute()  # type: ignore
    print(f"mAP (Mean Average Precision): {result['map'].item():.4f}")
    print(f"mAP @ 50% IoU: {result['map_50'].item():.4f}")


def main():
    evaluate_yolo(CHECKPOINTS_DIR / "yolo_n_tuned.ckpt")


if __name__ == "__main__":
    main()
