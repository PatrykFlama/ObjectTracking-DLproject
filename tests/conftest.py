from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from PIL import Image

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def mot15_sequence_dir(tmp_path: Path) -> Path:
    """Create the smallest MOT15-style sequence needed by dataset tests."""
    sequence_dir = tmp_path / "SEQ-01"
    img_dir = sequence_dir / "img1"
    gt_dir = sequence_dir / "gt"
    img_dir.mkdir(parents=True)
    gt_dir.mkdir()

    # Create two fake images
    Image.new("RGB", (100, 50), color=(255, 0, 0)).save(img_dir / "000001.jpg")
    Image.new("RGB", (100, 50), color=(0, 255, 0)).save(img_dir / "000002.jpg")

    # Create a gt.txt with two boxes in the first frame and one box in the second frame
    (gt_dir / "gt.txt").write_text(
        "\n".join(
            [
                "1,7,10,5,20,10,1,-1,-1,-1",
                "1,8,50,20,10,5,0,-1,-1,-1",
                "2,9,0,0,100,50,1,-1,-1,-1",
            ]
        )
    )
    return tmp_path
