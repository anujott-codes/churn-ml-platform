import json
import pandas as pd
import numpy as np
import pytest
from pathlib import Path
import optuna

from src.config.feature_config import TARGET_COLUMN
from src.config.tuner_config import PARAMS
from src.components.model_tuner import ModelTuner
from src.exception import ChurnPipelineException


def create_dummy_dataset(path: Path, n_samples: int = 100):
    np.random.seed(42)

    df = pd.DataFrame({
        "feature_1": np.random.randn(n_samples),
        "feature_2": np.random.randn(n_samples),
        "feature_3": np.random.randint(0, 5, n_samples),
        TARGET_COLUMN: np.random.randint(0, 2, n_samples)
    })

    df.to_csv(path, index=False)
    return df


class DummyPreprocessor:
    """Identity transformer for isolation."""
    def fit_transform(self, X):
        return X.copy()

    def transform(self, X):
        return X.copy()


@pytest.fixture(autouse=True)
def patch_data_transformation(monkeypatch):
    """
    Ensures tests do not depend on real DataTransformation logic.
    """
    from src.components import model_tuner

    class DummyDT:
        def get_preprocessor(self):
            return DummyPreprocessor()

    monkeypatch.setattr(model_tuner, "DataTransformation", DummyDT)


def test_load_data_success(tmp_path):
    data_path = tmp_path / "train.csv"
    create_dummy_dataset(data_path)

    tuner = ModelTuner(
        train_data_path=data_path,
        report_dir=tmp_path,
        n_trials=1
    )

    X, y = tuner.load_data()

    assert TARGET_COLUMN not in X.columns
    assert len(X) == len(y)
    assert y.name == TARGET_COLUMN


def test_load_data_missing_target(tmp_path):
    data_path = tmp_path / "train.csv"

    df = pd.DataFrame({
        "feature_1": [1, 2, 3]
    })
    df.to_csv(data_path, index=False)

    tuner = ModelTuner(
        train_data_path=data_path,
        report_dir=tmp_path
    )

    with pytest.raises(ChurnPipelineException):
        tuner.load_data()


def test_objective_returns_float(tmp_path):
    data_path = tmp_path / "train.csv"
    create_dummy_dataset(data_path)

    tuner = ModelTuner(
        train_data_path=data_path,
        report_dir=tmp_path,
        n_trials=1
    )

    X, y = tuner.load_data()

    trial_params = {
        key: (val[0] + val[1]) // 2 if isinstance(val[0], int)
        else float((val[0] + val[1]) / 2)
        for key, val in PARAMS.items()
    }

    trial = optuna.trial.FixedTrial(trial_params)

    auc = tuner.objective(trial, X, y)

    assert isinstance(auc, float)
    assert 0.0 <= auc <= 1.0


def test_tune_creates_reports(tmp_path):
    data_path = tmp_path / "train.csv"
    create_dummy_dataset(data_path)

    tuner = ModelTuner(
        train_data_path=data_path,
        report_dir=tmp_path,
        n_trials=2,
        random_state=42
    )

    best_params = tuner.tune()

    # Validate return
    assert isinstance(best_params, dict)
    assert "objective" in best_params
    assert "random_state" in best_params

    # Validate files created
    assert tuner.best_params_path.exists()
    assert tuner.trials_path.exists()
    assert tuner.summary_path.exists()

    # Validate summary content
    with open(tuner.summary_path) as f:
        summary = json.load(f)

    assert "best_auc" in summary
    assert "n_trials" in summary
    assert summary["n_trials"] == 2


def test_tune_is_deterministic(tmp_path):
    data_path = tmp_path / "train.csv"
    create_dummy_dataset(data_path)

    tuner1 = ModelTuner(
        train_data_path=data_path,
        report_dir=tmp_path / "run1",
        n_trials=1,
        random_state=42
    )

    tuner2 = ModelTuner(
        train_data_path=data_path,
        report_dir=tmp_path / "run2",
        n_trials=1,
        random_state=42
    )

    params1 = tuner1.tune()
    params2 = tuner2.tune()

    assert params1 == params2
