import json
import pickle
import pandas as pd
import numpy as np
import pytest
from pathlib import Path

from src.components.model_trainer import ModelTrainer
from src.config.feature_config import TARGET_COLUMN
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


def create_best_params(path: Path):
    params = {
        "n_estimators": 200,
        "objective": "binary",
        "random_state": 42,
        "max_depth": 3
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
    assert X.shape[1] == 3


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

    with open(params_path, "w") as f:
        json.dump({"n_estimators": 200}, f)

    trainer = ModelTrainer(
        train_data_path=tmp_path / "train.csv",
        best_params_path=params_path,
        model_dir=tmp_path
    )

    with pytest.raises(ChurnPipelineException):
        trainer.load_best_params()


def test_train_model_creates_artifacts(tmp_path):
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

    # Model file created
    assert model_path.exists()

    # Metadata file created
    assert trainer.metadata_path.exists()

    # Schema file created
    assert trainer.schema_path.exists()

    # Validate metadata structure
    with open(trainer.metadata_path) as f:
        metadata = json.load(f)

    assert metadata["model_type"] == trainer.model_type
    assert metadata["train_rows"] == 100
    assert metadata["n_features"] == 3
    assert "training_timestamp" in metadata
    assert "best_params" in metadata

    # Validate schema
    with open(trainer.schema_path) as f:
        schema = json.load(f)

    assert len(schema) == 3
    assert TARGET_COLUMN not in schema

    # Validate model can be loaded
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    assert hasattr(model, "predict")


def test_build_model_invalid_type(tmp_path):
    trainer = ModelTrainer(
        train_data_path=tmp_path / "train.csv",
        best_params_path=tmp_path / "params.json",
        model_dir=tmp_path,
        model_type="invalid_model"
    )

    with pytest.raises(ChurnPipelineException):
        trainer.build_model({})
