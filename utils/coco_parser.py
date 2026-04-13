import configparser
import json
import shutil
from pathlib import Path

from PIL import Image

CATEGORIES = [{"id": 1, "name": "pedestrian", "supercategory": "person"}]


def _parse_sequence(seq_dir: Path):
    img_dir = seq_dir / "img1"
    gt_path = seq_dir / "gt" / "gt.txt"
    seq_name = seq_dir.name

    ini_path = seq_dir / "seqinfo.ini"
    if ini_path.exists():
        cfg = configparser.ConfigParser()
        cfg.read(ini_path)
        width = int(cfg["Sequence"]["imWidth"])
        height = int(cfg["Sequence"]["imHeight"])
    else:
        img_files = sorted(img_dir.glob("*"))
        if img_files:
            first = img_files[0]
            with Image.open(first) as img:
                width, height = img.size
        else:
            width = 0
            height = 0

    img_paths = sorted(img_dir.glob("*"))
    frame_to_idx = {}
    images: list[dict] = []
    for img_path in img_paths:
        frame_num = int(img_path.stem)
        frame_to_idx[frame_num] = len(images)
        images.append(
            {
                "_src": img_path,
                "file_name": img_path.name,
                "width": width,
                "height": height,
                "seq_name": seq_name,
                "frame_id": frame_num,
            }
        )

    annotations = []
    if gt_path.exists():
        with Path.open(gt_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")

                if int(float(parts[6])) == 0:
                    continue

                frame = int(parts[0])
                obj_id = int(parts[1])
                x = float(parts[2])
                y = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])
                vis = float(parts[8]) if len(parts) > 8 else 1.0

                if frame not in frame_to_idx:
                    continue

                annotations.append(
                    {
                        "_frame_idx": frame_to_idx[frame],
                        "category_id": 1,
                        "bbox": [x, y, w, h],
                        "area": w * h,
                        "iscrowd": 0,
                        "track_id": obj_id,
                        "visibility": vis,
                    }
                )

    return images, annotations


def _build_coco(sequences_data):
    images = []
    annotations = []
    image_id = 1
    ann_id = 1

    for seq_images, seq_anns in sequences_data:
        local_id_map = {}
        for idx, img in enumerate(seq_images):
            local_id_map[idx] = image_id
            entry = {k: v for k, v in img.items() if k != "_src"}
            entry["id"] = image_id
            images.append(entry)
            image_id += 1

        for ann in seq_anns:
            entry = {k: v for k, v in ann.items() if k != "_frame_idx"}
            entry["id"] = ann_id
            entry["image_id"] = local_id_map[ann["_frame_idx"]]
            annotations.append(entry)
            ann_id += 1

    return {
        "info": {"description": "MOT15 in COCO format"},
        "licenses": [],
        "categories": CATEGORIES,
        "images": images,
        "annotations": annotations,
    }


def mot15_to_rfdetr(
    mot_root: Path,
    output_dir: Path,
    copy_images: bool = True,
):
    mot_root = Path(mot_root)
    output_dir = Path(output_dir)

    split_map = {
        "train": "train",
        "val": "valid",
    }

    for mot_split, rfdetr_split in split_map.items():
        split_dir = mot_root / mot_split

        seq_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
        print(
            f"\n[{mot_split} → {rfdetr_split}] {len(seq_dirs)}"
            f"sequences: {[s.name for s in seq_dirs]}"
        )
        sequences_data = []
        for seq_dir in seq_dirs:
            imgs, anns = _parse_sequence(seq_dir)
            sequences_data.append((imgs, anns))

        coco = _build_coco(sequences_data)

        out_split_dir = output_dir / rfdetr_split
        out_split_dir.mkdir(parents=True, exist_ok=True)

        for seq_imgs, _ in sequences_data:
            for img in seq_imgs:
                src = img["_src"]
                dst = out_split_dir / img["file_name"]
                if dst.exists():
                    continue
                if copy_images:
                    shutil.copy2(src, dst)
                else:
                    dst.symlink_to(src.resolve())

        ann_path = out_split_dir / "_annotations.coco.json"
        with Path.open(ann_path, "w") as f:
            json.dump(coco, f, indent=2)

    print(f"\n Dataset: {output_dir.resolve()}")


if __name__ == "__main__":
    mot15_to_rfdetr(
        mot_root=Path("ds/MOT15"),
        output_dir=Path("dataset"),
        copy_images=True,
    )
