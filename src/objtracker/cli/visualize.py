import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as F

if __package__ is None or __package__ == "":
    src_root = Path(__file__).resolve().parents[2]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

from ultralytics import YOLO

from objtracker.models.rf_detr import RFDETRLightning
from objtracker.models.yolo11 import YOLOLightning


def draw_boxes(image_path, boxes, output_path, color=(0, 0, 255)):
    img = cv2.imread(str(image_path))
    assert img is not None, f"Image not found at {image_path}"
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.imwrite(str(output_path), img)
    print(f"Saved visual to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to test image")
    parser.add_argument("--model", choices=["yolo", "rfdetr"], required=True)
    parser.add_argument("--weights", required=True, help="Path to .pt or .ckpt")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold")
    args = parser.parse_args()

    out_path = Path("artifacts/outputs") / f"{args.model}_pred_{Path(args.image).name}"
    out_path.parent.mkdir(exist_ok=True)

    if args.model == "yolo":
        print("Loading PyTorch Lightning checkpoint...")
        pl_model = YOLOLightning.load_from_checkpoint(args.weights, map_location="cpu")
        model = YOLO("artifacts/checkpoints/yolo11n.pt")
        model.model.load_state_dict(pl_model.model.state_dict())  # type: ignore

        print("Running Inference...")
        results = model.predict(args.image, conf=args.conf)

        annotated_img = results[0].plot()

        cv2.imwrite(str(out_path), annotated_img)
        print(f"Saved visual to: {out_path}")

    elif args.model == "rfdetr":
        print("Loading RF-DETR Lightning checkpoint...")
        model = RFDETRLightning.load_from_checkpoint(args.weights, map_location="cpu")
        model.eval()

        print("Running Inference...")
        original_image = cv2.imread(args.image)
        assert original_image is not None, f"Image not found: {args.image}"
        orig_h, orig_w = original_image.shape[:2]

        img_rgb = np.array(Image.open(args.image).convert("RGB"))
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        img_tensor = F.resize(img_tensor, [640, 640])
        img_tensor = F.normalize(
            img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        img_tensor = img_tensor.unsqueeze(0)

        with torch.no_grad():
            outputs = model(img_tensor)

        logits = outputs["pred_logits"][0]
        boxes = outputs["pred_boxes"][0]

        probas = logits.sigmoid().max(-1).values

        keep = probas > args.conf
        boxes_kept = boxes[keep]

        final_boxes = []
        for box in boxes_kept:
            cx, cy, w, h = box.tolist()
            x1 = (cx - w / 2) * orig_w
            y1 = (cy - h / 2) * orig_h
            x2 = (cx + w / 2) * orig_w
            y2 = (cy + h / 2) * orig_h
            final_boxes.append([x1, y1, x2, y2])

        draw_boxes(args.image, final_boxes, out_path, color=(0, 255, 0))


if __name__ == "__main__":
    main()
