import json
import pickle
import pandas as pd
import numpy as np
import pytest
from pathlib import Path
from sklearn.pipeline import Pipeline

from src.components.model_trainer import ModelTrainer
from src.config.feature_config import TARGET_COLUMN
from src.exception import ChurnPipelineException


def create_dummy_dataset(path: Path, n_samples: int = 100):
    np.random.seed(42)

    df = pd.DataFrame({
        # Numerical
        "age": [25, 30, 35, 40],
        "tenure": [1, 2, 3, 4],
        "usage_frequency": [10, 20, 30, 40],
        "support_calls": [0, 1, 2, 3],
        "payment_delay": [0, 1, 0, 2],
        "total_spend": [100, 200, 300, 400],
        "last_interaction": [5, 4, 3, 2],
        "high_support_calls": [0, 0, 1, 1],
        "payment_delay_flag": [0, 1, 0, 1],
        "spend_per_month": [100, 100, 100, 100],

        # Nominal
        "gender": ["Male", "Female", "Male", "Female"],
        "subscription_type": ["Basic", "Pro", "Premium", "Basic"],

        # Ordinal
        "contract_length": ["Monthly", "Quaterly", "Yearly", "Monthly"],

        # Target
        TARGET_COLUMN: [0, 1, 0, 1]
    })

    df.to_csv(path, index=False)
    return df


def create_best_params(path: Path):
    params = {
        "n_estimators": 200,
        "objective": "binary",
        "random_state": 42
    }

    with open(path, "w") as f:
        json.dump(params, f)

    return params


def test_load_data_success(tmp_path):
    data_path = tmp_path / "train.csv"
    create_dummy_dataset(data_path)

    trainer = ModelTrainer(
        train_data_path=data_path,
        best_params_path=tmp_path / "params.json",
        model_dir=tmp_path
    )

    X, y = trainer.load_data()

    assert TARGET_COLUMN not in X.columns
    assert len(X) == len(y)
    assert y.name == TARGET_COLUMN
    assert X.shape[1] == 13


def test_load_data_missing_target(tmp_path):
    data_path = tmp_path / "train.csv"
    pd.DataFrame({"feature_1": [1, 2, 3]}).to_csv(data_path, index=False)

    trainer = ModelTrainer(
        train_data_path=data_path,
        best_params_path=tmp_path / "params.json",
        model_dir=tmp_path
    )

    with pytest.raises(ChurnPipelineException):
        trainer.load_data()


def test_load_best_params_success(tmp_path):
    params_path = tmp_path / "params.json"
    expected_params = create_best_params(params_path)

    trainer = ModelTrainer(
        train_data_path=tmp_path / "train.csv",
        best_params_path=params_path,
        model_dir=tmp_path
    )

    loaded_params = trainer.load_best_params()

    assert loaded_params == expected_params


def test_load_best_params_missing_file(tmp_path):
    trainer = ModelTrainer(
        train_data_path=tmp_path / "train.csv",
        best_params_path=tmp_path / "missing.json",
        model_dir=tmp_path
    )

    with pytest.raises(ChurnPipelineException):
        trainer.load_best_params()


def test_load_best_params_invalid_structure(tmp_path):
    params_path = tmp_path / "params.json"

    # Missing required keys
    with open(params_path, "w") as f:
        json.dump({"n_estimators": 200}, f)

    trainer = ModelTrainer(
        train_data_path=tmp_path / "train.csv",
        best_params_path=params_path,
        model_dir=tmp_path
    )

    with pytest.raises(ChurnPipelineException):
        trainer.load_best_params()


def test_train_model_creates_full_pipeline_and_artifacts(tmp_path):
    data_path = tmp_path / "train.csv"
    params_path = tmp_path / "params.json"

    create_dummy_dataset(data_path)
    create_best_params(params_path)

    trainer = ModelTrainer(
        train_data_path=data_path,
        best_params_path=params_path,
        model_dir=tmp_path
    )

    model_path = trainer.train_model()

    # ---- Model file exists ----
    assert model_path.exists()

    # ---- Metadata exists ----
    assert trainer.metadata_path.exists()

    # ---- Schema exists ----
    assert trainer.schema_path.exists()

    # ---- Validate saved pipeline ----
    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)

    assert isinstance(pipeline, Pipeline)
    assert "preprocessor" in pipeline.named_steps
    assert "model" in pipeline.named_steps

    # ---- Validate metadata ----
    with open(trainer.metadata_path) as f:
        metadata = json.load(f)

    assert metadata["model_type"] == trainer.model_type
    assert metadata["train_rows"] == 4
    assert metadata["n_raw_features"] == 13
    assert "training_timestamp" in metadata
    assert "best_params" in metadata

    # scale_pos_weight must be injected
    assert "scale_pos_weight" in metadata["best_params"]

    # ---- Validate schema ----
    with open(trainer.schema_path) as f:
        schema = json.load(f)

    assert isinstance(schema, list)
    assert len(schema) == 13
    assert TARGET_COLUMN not in schema


def test_build_model_invalid_type(tmp_path):
    trainer = ModelTrainer(
        train_data_path=tmp_path / "train.csv",
        best_params_path=tmp_path / "params.json",
        model_dir=tmp_path,
        model_type="invalid_model"
    )

    with pytest.raises(ChurnPipelineException):
        trainer.build_model({})
