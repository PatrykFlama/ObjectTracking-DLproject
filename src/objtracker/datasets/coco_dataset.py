from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as F

from objtracker.datasets.mot15_dataset import MOT15Dataset


class CocoDataset(MOT15Dataset):
    def __init__(
        self,
        root_dir: str,
        sequence: str,
        transforms=None,
        image_size: int = 640,
    ):
        super().__init__(root_dir=root_dir, sequence=sequence, transforms=transforms)
        self.image_size = image_size
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

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

        box_tensor = (
            torch.tensor(boxes, dtype=torch.float32)
            if boxes
            else torch.zeros((0, 4), dtype=torch.float32)
        )
        label_tensor = (
            torch.tensor(labels, dtype=torch.int64)
            if labels
            else torch.zeros((0,), dtype=torch.int64)
        )
        track_id_tensor = (
            torch.tensor(track_ids, dtype=torch.int64)
            if track_ids
            else torch.zeros((0,), dtype=torch.int64)
        )

        target = {
            "boxes": box_tensor,
            "labels": label_tensor,
            "track_ids": track_id_tensor,
            "image_id": torch.tensor([frame_id]),
        }

        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image = F.resize(image, [self.image_size, self.image_size])
        image = F.normalize(image, mean=self.mean, std=self.std)

        if self.transforms:
            image = self.transforms(image)

        return image, target
