# Object Tracking - Deep Learning project

![CI](https://github.com/PatrykFlama/ObjectTracking-DLproject/actions/workflows/ci.yml/badge.svg)
![Coverage](https://codecov.io/gh/PatrykFlama/ObjectTracking-DLproject/branch/main/graph/badge.svg)

## Setup
```bash
uvx pre-commit install
uv sync
uv run python src/objtracker/train.py
```

## Optuna Search
```bash
uv run python src/objtracker/optuna_search.py --n-trials 20 --epochs 10
```

## Test
```bash
uv run pytest
```

## Build
```bash
uv build
```
