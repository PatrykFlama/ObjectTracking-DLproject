import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import functional as F
from tqdm import tqdm

if __package__ is None or __package__ == "":
    src_root = Path(__file__).resolve().parents[1]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

from ultralytics import YOLO

from objtracker.models.rf_detr import RFDETRLightning
from objtracker.models.yolo11 import YOLOLightning


def calculate_iou(box1, box2):
    """Calculates Intersection over Union for two bounding boxes [x1, y1, x2, y2]"""
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return intersection_area / float(box1_area + box2_area - intersection_area)


def evaluate_predictions(preds, gts, iou_threshold=0.5):
    """Calculates True Positives, False Positives, and False Negatives"""
    tp = 0
    fp = 0
    matched_gt_indices = set()

    preds = sorted(preds, key=lambda x: x[4], reverse=True)

    for pred in preds:
        best_iou = 0.0
        best_gt_idx = -1

        for idx, gt in enumerate(gts):
            if idx in matched_gt_indices:
                continue
            iou = calculate_iou(pred[:4], gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = idx

        if best_iou >= iou_threshold:
            tp += 1
            matched_gt_indices.add(best_gt_idx)
        else:
            fp += 1

    fn = len(gts) - len(matched_gt_indices)
    return tp, fp, fn


def main():
    sequence_path = Path("dataset/MOT15/train/ETH-Sunnyday")
    img_dir = sequence_path / "img1"
    gt_file = sequence_path / "gt/gt.txt"

    gt_df = pd.read_csv(
        gt_file,
        header=None,
        names=[
            "frame",
            "id",
            "bb_left",
            "bb_top",
            "bb_width",
            "bb_height",
            "conf",
            "x",
            "y",
            "z",
        ],
    )

    print("Loading Models to CPU...")
    yolo_pl = YOLOLightning.load_from_checkpoint(
        "artifacts/yolo_n_tuned.ckpt", map_location="cpu"
    )
    yolo_model = YOLO("yolo11n.pt")
    yolo_model.model.load_state_dict(yolo_pl.model.state_dict())

    rfdetr_model = RFDETRLightning.load_from_checkpoint(
        "artifacts/rfdetr_nano_tuned.ckpt", map_location="cpu"
    )
    rfdetr_model.eval()

    frames = sorted(img_dir.glob("*.jpg"))
    conf_threshold = 0.4

    results = {
        "YOLO": {"tp": 0, "fp": 0, "fn": 0},
        "RF-DETR": {"tp": 0, "fp": 0, "fn": 0},
    }

    print("Evaluating ETH-Sunnyday...")
    for frame_path in tqdm(frames):
        frame_idx = int(frame_path.stem)

        frame_gts = gt_df[gt_df["frame"] == frame_idx]
        gts = []
        for _, row in frame_gts.iterrows():
            if row["conf"] == 0:
                continue
            x1, y1 = row["bb_left"], row["bb_top"]
            x2, y2 = x1 + row["bb_width"], y1 + row["bb_height"]
            gts.append([x1, y1, x2, y2])

        yolo_res = yolo_model.predict(
            str(frame_path), conf=conf_threshold, verbose=False
        )[0]
        yolo_preds = []
        if len(yolo_res.boxes) > 0:
            boxes = yolo_res.boxes.xyxy.cpu().numpy()
            confs = yolo_res.boxes.conf.cpu().numpy()
            for b, c in zip(boxes, confs):
                yolo_preds.append([b[0], b[1], b[2], b[3], c])

        orig_img = Image.open(frame_path).convert("RGB")
        orig_w, orig_h = orig_img.size
        img_tensor = (
            torch.from_numpy(np.array(orig_img)).permute(2, 0, 1).float() / 255.0
        )
        img_tensor = F.resize(img_tensor, [640, 640])
        img_tensor = F.normalize(
            img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ).unsqueeze(0)

        with torch.no_grad():
            rf_outputs = rfdetr_model(img_tensor)

        logits = rf_outputs["pred_logits"][0].sigmoid().max(-1).values
        rf_boxes = rf_outputs["pred_boxes"][0]

        keep = logits > conf_threshold
        rf_boxes = rf_boxes[keep]
        rf_confs = logits[keep]

        rf_preds = []
        for box, conf in zip(rf_boxes, rf_confs):
            cx, cy, w, h = box.tolist()
            x1 = (cx - w / 2) * orig_w
            y1 = (cy - h / 2) * orig_h
            x2 = (cx + w / 2) * orig_w
            y2 = (cy + h / 2) * orig_h
            rf_preds.append([x1, y1, x2, y2, conf.item()])

        y_tp, y_fp, y_fn = evaluate_predictions(yolo_preds, gts)
        results["YOLO"]["tp"] += y_tp
        results["YOLO"]["fp"] += y_fp
        results["YOLO"]["fn"] += y_fn

        r_tp, r_fp, r_fn = evaluate_predictions(rf_preds, gts)
        results["RF-DETR"]["tp"] += r_tp
        results["RF-DETR"]["fp"] += r_fp
        results["RF-DETR"]["fn"] += r_fn

    print("\n" + "=" * 40)
    print("FINAL EVALUATION ON ETH-SUNNYDAY")
    print("=" * 40)

    for name, metrics in results.items():
        tp, fp, fn = metrics["tp"], metrics["fp"], metrics["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        print(f"--- {name} ---")
        print(f"True Positives  (Found)  : {tp}")
        print(f"False Positives (Ghosts) : {fp}")
        print(f"False Negatives (Missed) : {fn}")
        print(f"Precision : {precision:.2%}")
        print(f"Recall    : {recall:.2%}")
        print(f"F1-Score  : {f1:.2%}\n")


if __name__ == "__main__":
    main()
