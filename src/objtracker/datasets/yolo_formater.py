from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as F

from objtracker.datasets.mot15_dataset import MOT15Dataset


class YoloDataset(MOT15Dataset):
    def __init__(
        self,
        root_dir: str,
        sequence: str,
        transforms=None,
        image_size: int = 640,
    ):
        super().__init__(root_dir=root_dir, sequence=sequence, transforms=transforms)
        self.image_size = image_size

    def __getitem__(self, index: Any) -> tuple[torch.Tensor, torch.Tensor]:
        frame_id = self.frames[index]

        img_path = Path(self.img_dir) / f"{frame_id:06d}.jpg"
        if not img_path.exists():
            msg = f"Could not read image file: {img_path}"
            raise FileNotFoundError(msg)
        image = np.array(Image.open(img_path).convert("RGB"))

        img_h, img_w = image.shape[:2]
        records = self.gt[self.gt["frame"] == frame_id]

        boxes = []
        for _, row in records.iterrows():
            bb_w = row["bb_width"]
            bb_h = row["bb_height"]
            cx = (row["bb_left"] + bb_w / 2) / img_w
            cy = (row["bb_top"] + bb_h / 2) / img_h
            bb_w /= img_w
            bb_h /= img_h
            boxes.append([0, cx, cy, bb_w, bb_h])  # [class_id, cx, cy, w, h]

        box_tensor = (
            torch.tensor(boxes, dtype=torch.float32)
            if boxes
            else torch.zeros((0, 5), dtype=torch.float32)
        )

        # No ImageNet normalization — YOLO expects raw [0, 1]
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image = F.resize(image, [self.image_size, self.image_size])

        if self.transforms:
            image = self.transforms(image)

        return image, box_tensor


def yolo_collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    images, targets = zip(*batch)
    images = torch.stack(images)

    target_list = []
    for i, t in enumerate(targets):
        if t.shape[0] > 0:
            batch_col = torch.full((t.shape[0], 1), i, dtype=torch.float32)
            target_list.append(torch.cat([batch_col, t], dim=1))

    targets_tensor = (
        torch.cat(target_list, dim=0)
        if target_list
        else torch.zeros((0, 6), dtype=torch.float32)
    )
    return images, targets_tensor