from pathlib import Path
from typing import Any

import cv2
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class MOT15Dataset(Dataset[tuple[torch.Tensor, dict[str, torch.Tensor]]]):
    def __init__(self, root_dir, sequence, transforms=None):
        self.root_dir = root_dir
        self.sequence = sequence
        self.transforms = transforms

        self.img_dir = Path(root_dir) / sequence / "img1"
        self.gt_path = Path(root_dir) / sequence / "gt" / "gt.txt"

        self.gt = pd.read_csv(self.gt_path, header=None, sep=",")
        self.gt.columns = [
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
        ]

        ## use only good quality boxes
        self.gt = self.gt[self.gt["conf"] == 1]
        # use unique sorted frame ids so each dataset item corresponds to frame
        self.frames = sorted(self.gt["frame"].unique())

    def __len__(self):
        return len(self.frames)

    def __getitem__(
        self, index: Any
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        frame_id = self.frames[index]

        img_path = Path(self.img_dir) / f"{frame_id:06d}.jpg"
        image = cv2.imread(str(img_path))
        if image is None:
            msg = f"Could not read image file: {img_path}"
            raise FileNotFoundError(msg)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        records = self.gt[self.gt["frame"] == frame_id]

        boxes = []
        track_ids = []

        for _, row in records.iterrows():
            x1 = row["bb_left"]
            y1 = row["bb_top"]
            x2 = x1 + row["bb_width"]
            y2 = y1 + row["bb_height"]
            boxes.append([x1, y1, x2, y2])
            track_ids.append(int(row["id"]))

        boxes = torch.tensor(boxes, dtype=torch.float32)
        track_ids = torch.tensor(track_ids, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "track_ids": track_ids,
            "image_id": torch.tensor([frame_id]),
        }

        if self.transforms:
            image = self.transforms(image)

        image = torch.tensor(image, dtype=torch.float32)
        return image, target


def get_mot15_loader(root_dir, sequence, batch_size=2):
    dataset = MOT15Dataset(root_dir, sequence)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: tuple(zip(*x)),
    )


if __name__ == "__main__":
    loader = get_mot15_loader("ds/MOT15/train", "ADL-Rundle-6")

    for images, targets in loader:
        print("Batch:", len(images))
        print("Image tensor shape:", images[0].shape)
        print("Boxes:", targets[0]["boxes"])
        print("Track IDs:", targets[0]["track_ids"])
        break
