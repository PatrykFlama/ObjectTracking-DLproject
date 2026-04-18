from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from objtracker.datasets.mot15_dataset import MOT15Dataset


class CocoDataset(MOT15Dataset):
    def __getitem__(self, index: Any) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        frame_id = self.frames[index]

        img_path = Path(self.img_dir) / f"{frame_id:06d}.jpg"
        if not img_path.exists():
            msg = f"Could not read image file: {img_path}"
            raise FileNotFoundError(msg)
        image = np.array(Image.open(img_path).convert("RGB"))

        # Grab image dimensions for normalization
        img_h, img_w = image.shape[:2]

        records = self.gt[self.gt["frame"] == frame_id]

        boxes = []
        labels = []
        track_ids = []

        for _, row in records.iterrows():
            # 1. Get raw width and height
            bb_w = row["bb_width"]
            bb_h = row["bb_height"]

            # 2. Calculate center X and center Y
            cx = row["bb_left"] + (bb_w / 2)
            cy = row["bb_top"] + (bb_h / 2)

            # 3. Normalize all values between 0 and 1
            cx /= img_w
            cy /= img_h
            bb_w /= img_w
            bb_h /= img_h

            boxes.append([cx, cy, bb_w, bb_h])
            labels.append(0)  # 0 represents "Pedestrian"
            track_ids.append(int(row["id"]))

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),  # Added for DETR
            "track_ids": torch.tensor(track_ids, dtype=torch.int64),
            "image_id": torch.tensor([frame_id]),
        }

        if self.transforms:
            image = self.transforms(image)

        # DETR models expect images to be shape [Channels, Height, Width]
        # OpenCV loads them as [Height, Width, Channels]. We permute it here.
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)

        # Normalize image pixels from 0-255 to 0.0-1.0 (Standard DL practice)
        image = image / 255.0

        return image, target
