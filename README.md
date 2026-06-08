# Object Tracking - Deep Learning Project

![CI](https://github.com/PatrykFlama/ObjectTracking-DLproject/actions/workflows/ci.yml/badge.svg)
![Coverage](https://codecov.io/gh/PatrykFlama/ObjectTracking-DLproject/branch/main/graph/badge.svg)

## Overview

This repository contains a modular Deep Learning pipeline for Multi-Object Tracking (MOT). It supports dynamic swapping between state-of-the-art object detectors (YOLO, RF-DETR) and tracking algorithms (ByteTrack, BotSORT).

---

## Installation & Setup

We use `uv` for lightning-fast dependency management.

```bash
# Install the pre-commit hooks to ensure code quality
uvx pre-commit install

# Sync the environment dependencies
uv sync
```

---

## Running the Tracking Pipeline

Our architecture uses a `TrackingPipeline` that cleanly separates the detection phase from the tracking phase. You can hot-swap models using our factory functions.

Here is how you can fire up the pipeline in a Python script:

```python
import cv2
from objtracker.models import build_detector
from objtracker.tracking import build_tracker
from objtracker.pipeline import TrackingPipeline

# 1. Initialize the modular pipeline
pipeline = TrackingPipeline(
    detector=build_detector("yolo"),       # Options: "yolo", "rfdetr"
    tracker=build_tracker("bytetrack"),    # Options: "bytetrack", "botsort"
)

# 2. Process a video frame-by-frame
cap = cv2.VideoCapture("dataset/MOT15/train/ETH-Sunnyday/img1/000001.jpg")
ret, frame = cap.read()

if ret:
    # The pipeline handles detection and tracking ID assignment automatically
    tracked_objects = pipeline.update(frame)

    print(
        f"Tracking IDs in frame: "
        f"{[track.track_id for track in tracked_objects]}"
    )
```

## Running a Video Demo
To easily test the pipeline and generate an MP4 visualization, run our demo script. You can seamlessly swap models via the command line:
```bash
uv run python src/objtracker/run_demo.py --detector yolo --tracker bytetrack

---

## Testing

We use `pytest` for our testing suite. Our tests ensure that the tracker factories correctly initialize, maintain identity states across frames, and properly reset when sequences end.

```bash
# Run the full test suite
uv run pytest
```

---

## Training & Optimization

To train the models or run hyperparameter optimization, use our built-in CLI commands:

```bash
# Standard Training
uv run objtracker-train

# Optuna Hyperparameter Search
uv run objtracker-optuna --n-trials 20 --epochs 10
```

---

## Architecture

The project follows a modular architecture:

```text
Frame
  │
  ▼
Detector (YOLO / RF-DETR)
  │
  ▼
Detections
  │
  ▼
Tracker (ByteTrack / BotSORT)
  │
  ▼
Tracked Objects with IDs
```

This separation allows independent experimentation with detectors and trackers without modifying the rest of the pipeline.

---

## Project Structure

```text
objtracker/
├── models/          # Detector implementations and factories
├── tracking/        # Tracker implementations and factories
├── pipeline/        # TrackingPipeline orchestration
├── training/        # Training utilities
├── optimization/    # Optuna hyperparameter tuning
└── utils/           # Shared utilities

tests/
├── test_models.py
├── test_trackers.py
└── test_pipeline.py
```

---

## Supported Components

### Detectors

* YOLO
* RF-DETR

### Trackers

* ByteTrack
* BotSORT

Additional detectors and trackers can be integrated by implementing the corresponding interfaces and registering them in the factory functions.

---

## Contributing

1. Fork the repository.
2. Create a feature branch.
3. Install dependencies with `uv sync`.
4. Run tests with `uv run pytest`.
5. Submit a pull request.

Please ensure all tests pass and code is formatted before submitting changes.

---
