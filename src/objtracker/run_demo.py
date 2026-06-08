import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

if __package__ is None or __package__ == "":
    src_root = Path(__file__).resolve().parents[1]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

from objtracker.models import build_detector
from objtracker.pipeline import TrackingPipeline
from objtracker.tracking import build_tracker


def get_color(track_id):
    """Generate a unique, consistent color for each track ID."""
    rng = np.random.default_rng(track_id)
    color = rng.integers(0, 255, size=(3,)).tolist()
    return tuple(color)


def main():
    parser = argparse.ArgumentParser(description="Run Object Tracking Pipeline Demo")
    parser.add_argument(
        "--sequence",
        type=str,
        default="dataset/MOT15/train/ETH-Sunnyday/img1",
        help="Path to sequence images",
    )
    parser.add_argument(
        "--detector", type=str, default="yolo", choices=["yolo", "rfdetr"]
    )
    parser.add_argument(
        "--tracker", type=str, default="bytetrack", choices=["bytetrack", "botsort"]
    )
    parser.add_argument(
        "--output", type=str, default="demo_output.mp4", help="Output video path"
    )
    args = parser.parse_args()

    print(f"Initializing Pipeline: {args.detector.upper()} + {args.tracker.upper()}")

    pipeline = TrackingPipeline(
        detector=build_detector(args.detector), tracker=build_tracker(args.tracker)
    )

    img_dir = Path(args.sequence)
    frames = sorted(img_dir.glob("*.jpg"))
    if not frames:
        print(f"No images found in {args.sequence}")
        return

    first_frame = cv2.imread(str(frames[0]))
    height, width = first_frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video = cv2.VideoWriter(args.output, fourcc, 30.0, (width, height))

    print(f"Processing {len(frames)} frames...")

    for frame_path in tqdm(frames):
        frame = cv2.imread(str(frame_path))

        pipeline_result = pipeline.update(frame)

        if hasattr(pipeline_result, "tracks"):
            active_tracks = pipeline_result.tracks
        elif hasattr(pipeline_result, "tracked_objects"):
            active_tracks = pipeline_result.tracked_objects
        else:
            print(
                "\nError: Could not find tracks. "
                f"PipelineResult attributes: {dir(pipeline_result)}"
            )
            break

        if active_tracks:
            for track in active_tracks:
                x1, y1, x2, y2 = map(int, track.box.tolist())
                track_id = track.track_id
                color = get_color(track_id)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"ID: {track_id}"
                cv2.putText(
                    frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                )

        out_video.write(frame)

    out_video.release()
    print(f"\nDemo successfully saved to: {args.output}")


if __name__ == "__main__":
    main()
