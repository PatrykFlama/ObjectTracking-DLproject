# Object Tracking - Deep Learning project

![CI](https://github.com/PatrykFlama/ObjectTracking-DLproject/actions/workflows/ci.yml/badge.svg)
![Coverage](https://codecov.io/gh/PatrykFlama/ObjectTracking-DLproject/branch/main/graph/badge.svg)

## Setup
```bash
uvx pre-commit install
uv sync
uv run objtracker-train
```

## Optuna Search
```bash
uv run objtracker-optuna --n-trials 20 --epochs 10
```

## Test
```bash
uv run pytest
```

## Tracking
```python
from objtracker.tracking import Detections, build_tracker

tracker = build_tracker("bytetrack")  # or "botsort"
tracks = tracker.update(
    Detections(
        boxes=boxes_xyxy,
        scores=confidence_scores,
        labels=class_labels,
    )
)
```

## Build
```bash
uv build
```
