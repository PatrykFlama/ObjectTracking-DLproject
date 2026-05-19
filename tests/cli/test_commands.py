from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pandas as pd

from objtracker.paths import CHECKPOINTS_DIR, OUTPUTS_DIR, PROJECT_ROOT

if TYPE_CHECKING:
    import pytest


def test_artifact_paths_are_project_root_relative() -> None:
    assert CHECKPOINTS_DIR == PROJECT_ROOT / "artifacts" / "checkpoints"
    assert OUTPUTS_DIR == PROJECT_ROOT / "artifacts" / "outputs"


def test_train_main_dispatches_to_train(monkeypatch: pytest.MonkeyPatch) -> None:
    from objtracker.cli import train as train_cli

    args = Namespace(model="rfdetr")
    calls = []

    monkeypatch.setattr(train_cli, "parse_args", lambda: args)
    monkeypatch.setattr(train_cli, "train", calls.append)

    train_cli.main()

    assert calls == [args]


def test_optuna_main_uses_parsed_search_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from objtracker.cli import optuna_search

    args = Namespace(
        study_name="study",
        storage=None,
        n_trials=3,
        timeout=10,
        metric="val_loss",
    )
    optimize_calls = []

    class FakeStudy:
        best_trial = SimpleNamespace(number=7)
        best_value = 0.25
        best_params = {"lr": 1e-4}

        def optimize(self, callback, n_trials, timeout):
            optimize_calls.append((callback, n_trials, timeout))

    monkeypatch.setattr(optuna_search, "parse_args", lambda: args)
    monkeypatch.setattr(optuna_search.optuna, "create_study", lambda **_: FakeStudy())

    optuna_search.main()

    assert optimize_calls
    _, n_trials, timeout = optimize_calls[0]
    assert n_trials == 3
    assert timeout == 10


def test_evaluate_main_uses_checkpoint_artifact_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from objtracker.cli import evaluate

    calls = []
    monkeypatch.setattr(evaluate, "evaluate_yolo", calls.append)

    evaluate.main()

    assert calls == [CHECKPOINTS_DIR / "yolo_n_tuned.ckpt"]


def test_visualize_yolo_uses_checkpoint_and_output_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from objtracker.cli import visualize

    load_calls = []
    yolo_calls = []
    write_calls = []

    class FakeYOLOLightning:
        model = SimpleNamespace(state_dict=lambda: {"weight": "value"})

        @classmethod
        def load_from_checkpoint(cls, path, map_location):
            load_calls.append((path, map_location))
            return cls()

    class FakeYOLOModel:
        def load_state_dict(self, state_dict):
            assert state_dict == {"weight": "value"}

    class FakeYOLO:
        model = FakeYOLOModel()

        def __init__(self, path):
            yolo_calls.append(path)

        def predict(self, image, conf):
            assert image == "input.jpg"
            assert conf == 0.7
            return [SimpleNamespace(plot=lambda: "annotated")]

    monkeypatch.setattr(visualize, "YOLOLightning", FakeYOLOLightning)
    monkeypatch.setattr(visualize, "YOLO", FakeYOLO)
    monkeypatch.setattr(
        visualize.cv2,
        "imwrite",
        lambda path, img: write_calls.append((path, img)),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "objtracker-visualize",
            "--image",
            "input.jpg",
            "--model",
            "yolo",
            "--weights",
            "model.ckpt",
            "--conf",
            "0.7",
        ],
    )

    visualize.main()

    assert load_calls == [("model.ckpt", "cpu")]
    assert yolo_calls == [str(CHECKPOINTS_DIR / "yolo11n.pt")]
    assert write_calls == [
        (str(OUTPUTS_DIR / "yolo_pred_input.jpg"), "annotated"),
    ]


def test_compare_models_uses_checkpoint_artifact_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from objtracker.cli import compare_models

    yolo_load_calls = []
    rfdetr_load_calls = []
    yolo_calls = []

    class FakeYOLOLightning:
        model = SimpleNamespace(state_dict=lambda: {"weight": "value"})

        @classmethod
        def load_from_checkpoint(cls, path, map_location):
            yolo_load_calls.append((path, map_location))
            return cls()

    class FakeRFDETR:
        @classmethod
        def load_from_checkpoint(cls, path, map_location):
            rfdetr_load_calls.append((path, map_location))
            return cls()

        def eval(self):
            return None

    class FakeYOLOModel:
        def load_state_dict(self, state_dict):
            assert state_dict == {"weight": "value"}

    class FakeYOLO:
        model = FakeYOLOModel()

        def __init__(self, path):
            yolo_calls.append(path)

    monkeypatch.setattr(compare_models, "YOLOLightning", FakeYOLOLightning)
    monkeypatch.setattr(compare_models, "RFDETRLightning", FakeRFDETR)
    monkeypatch.setattr(compare_models, "YOLO", FakeYOLO)
    monkeypatch.setattr(compare_models.pd, "read_csv", lambda *_, **__: pd.DataFrame())
    monkeypatch.setattr(compare_models.Path, "glob", lambda *_: [])

    compare_models.main()

    assert yolo_load_calls == [
        (CHECKPOINTS_DIR / "yolo_n_tuned.ckpt", "cpu"),
    ]
    assert yolo_calls == [str(CHECKPOINTS_DIR / "yolo11n.pt")]
    assert rfdetr_load_calls == [
        (CHECKPOINTS_DIR / "rfdetr_nano_tuned.ckpt", "cpu"),
    ]
