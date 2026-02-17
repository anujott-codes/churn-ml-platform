import json
import pickle
import numpy as np
import pandas as pd
import pytest

from pathlib import Path
from sklearn.linear_model import LogisticRegression

from src.components.model_evaluator import ModelEvaluator
from src.config.feature_config import TARGET_COLUMN
from src.exception import ChurnPipelineException


def create_dummy_pipeline(path: Path):
    """
    Creates and saves a fitted LogisticRegression model
    that exposes predict_proba (required by evaluator).
    """
    np.random.seed(42)

    X = pd.DataFrame(
        np.random.randn(200, 3),
        columns=["feature_1", "feature_2", "feature_3"]
    )
    y = np.random.randint(0, 2, 200)

    model = LogisticRegression()
    model.fit(X, y)

    with open(path, "wb") as f:
        pickle.dump(model, f)

    return model


def create_test_dataset(path: Path, n_samples: int = 100):
    np.random.seed(42)

    df = pd.DataFrame({
        "feature_1": np.random.randn(n_samples),
        "feature_2": np.random.randn(n_samples),
        "feature_3": np.random.randn(n_samples),
        TARGET_COLUMN: np.random.randint(0, 2, n_samples)
    })

    df.to_csv(path, index=False)
    return df


def test_invalid_threshold_raises(tmp_path):
    with pytest.raises(ChurnPipelineException):
        ModelEvaluator(
            model_path=tmp_path / "model.pkl",
            test_data_path=tmp_path / "test.csv",
            reports_dir=tmp_path,
            metrics_dir=tmp_path,
            threshold=1.5,  # invalid
        )

def test_load_pipeline_success(tmp_path):
    model_path = tmp_path / "model.pkl"
    create_dummy_pipeline(model_path)

    evaluator = ModelEvaluator(
        model_path=model_path,
        test_data_path=tmp_path / "test.csv",
        reports_dir=tmp_path,
        metrics_dir=tmp_path,
    )

    pipeline = evaluator.load_pipeline()

    assert hasattr(pipeline, "predict_proba")


def test_load_pipeline_failure(tmp_path):
    evaluator = ModelEvaluator(
        model_path=tmp_path / "missing.pkl",
        test_data_path=tmp_path / "test.csv",
        reports_dir=tmp_path,
        metrics_dir=tmp_path,
    )

    with pytest.raises(ChurnPipelineException):
        evaluator.load_pipeline()

def test_load_test_data_success(tmp_path):
    test_path = tmp_path / "test.csv"
    df = create_test_dataset(test_path, n_samples=50)

    evaluator = ModelEvaluator(
        model_path=tmp_path / "model.pkl",
        test_data_path=test_path,
        reports_dir=tmp_path,
        metrics_dir=tmp_path,
    )

    X, y = evaluator.load_test_data()

    assert len(X) == 50
    assert len(y) == 50
    assert TARGET_COLUMN not in X.columns


def test_load_test_data_missing_target(tmp_path):
    test_path = tmp_path / "test.csv"
    pd.DataFrame({"feature_1": [1, 2, 3]}).to_csv(test_path, index=False)

    evaluator = ModelEvaluator(
        model_path=tmp_path / "model.pkl",
        test_data_path=test_path,
        reports_dir=tmp_path,
        metrics_dir=tmp_path,
    )

    with pytest.raises(ChurnPipelineException):
        evaluator.load_test_data()


def test_compute_precision_at_k_bounds(tmp_path):
    evaluator = ModelEvaluator(
        model_path=tmp_path / "model.pkl",
        test_data_path=tmp_path / "test.csv",
        reports_dir=tmp_path,
        metrics_dir=tmp_path,
    )

    y_true = pd.Series([1, 0, 1, 0, 1])
    y_proba = np.array([0.9, 0.8, 0.7, 0.1, 0.2])

    score = evaluator.compute_precision_at_k(y_true, y_proba)

    assert 0.0 <= score <= 1.0


def test_evaluate_creates_report_and_plots(tmp_path):
    model_path = tmp_path / "model.pkl"
    test_path = tmp_path / "test.csv"

    create_dummy_pipeline(model_path)
    create_test_dataset(test_path, n_samples=100)

    evaluator = ModelEvaluator(
        model_path=model_path,
        test_data_path=test_path,
        reports_dir=tmp_path,
        metrics_dir=tmp_path,
        threshold=0.5,
    )

    report, cm = evaluator.evaluate()

    # ---- report structure ----
    assert isinstance(report, dict)

    required_keys = [
        "roc_auc",
        "pr_auc",
        "precision",
        "recall",
        "f1_score",
        "accuracy",
        "precision_at_k",
        "threshold",
        "test_rows",
        "class_distribution",
        "evaluation_timestamp",
    ]

    for key in required_keys:
        assert key in report

    # ---- metric bounds ----
    assert 0.0 <= report["roc_auc"] <= 1.0
    assert 0.0 <= report["accuracy"] <= 1.0
    assert report["test_rows"] == 100

    # ---- confusion matrix ----
    assert cm.shape == (2, 2)

    # ---- files created ----
    assert evaluator.report_path.exists()
    assert evaluator.conf_matrix_path.exists()
    assert evaluator.roc_curve_path.exists()
    assert evaluator.pr_curve_path.exists()

    # ---- validate saved JSON ----
    with open(evaluator.report_path) as f:
        saved_report = json.load(f)

    assert saved_report["test_rows"] == 100
