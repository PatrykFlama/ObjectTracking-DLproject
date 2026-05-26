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
from objtracker.models import build_detector
from objtracker.pipeline import TrackingPipeline
from objtracker.tracking import build_tracker

pipeline = TrackingPipeline(
    detector=build_detector("yolo"),  # or "rfdetr"
    tracker=build_tracker("bytetrack"),  # or "botsort"
)
result = pipeline.update(frame)
```

## Build
```bash
uv build
```
