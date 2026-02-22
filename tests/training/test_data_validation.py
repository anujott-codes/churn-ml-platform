import json

import pandas as pd
import pytest

from src.components.data_validation import DataValidator
from src.config.schema import RAW_DATA_SCHEMA, TARGET_COLUMN
from src.exception import ChurnPipelineException


@pytest.fixture
def valid_df():
    """Generate valid dataframe based on RAW_DATA_SCHEMA."""
    data = {}

    for col, dtype in RAW_DATA_SCHEMA.items():
        if dtype == "object":
            data[col] = ["A", "B", "C", "D"]
        else:
            data[col] = [1, 2, 3, 4]

    data[TARGET_COLUMN] = [0, 1, 0, 1]

    return pd.DataFrame(data)


@pytest.fixture
def save_csv(tmp_path):
    def _save(df: pd.DataFrame, name="data.csv"):
        path = tmp_path / name
        df.to_csv(path, index=False)
        return path
    return _save


@pytest.fixture
def make_validator(tmp_path):
    """
    Factory fixture to ensure every validator instance
    uses isolated staging + reports directories.
    """
    def _make(csv_path, **kwargs):
        return DataValidator(
            raw_data_path=csv_path,
            staged_dir=tmp_path / "staged",
            reports_dir=tmp_path / "reports",
            **kwargs
        )
    return _make


def test_validation_success(valid_df, save_csv, make_validator, tmp_path):
    csv_path = save_csv(valid_df)

    validator = make_validator(csv_path)
    staged_path = validator.validate()

    # Staged file created
    assert staged_path.exists()
    assert staged_path.parent == tmp_path / "staged"

    # Report created
    reports = list((tmp_path / "reports").glob("*validation_report.json"))
    assert len(reports) == 1

    with open(reports[0]) as f:
        report = json.load(f)

    assert report["validation_status"] == "PASSED"
    assert report["data_shape"]["n_rows"] == len(valid_df)


def test_missing_file(make_validator, tmp_path):
    fake_path = tmp_path / "missing.csv"
    validator = make_validator(fake_path)

    with pytest.raises(ChurnPipelineException):
        validator.validate()


def test_missing_required_column(valid_df, save_csv, make_validator):
    df = valid_df.drop(columns=[list(RAW_DATA_SCHEMA.keys())[0]])
    csv_path = save_csv(df)

    validator = make_validator(csv_path)

    with pytest.raises(ChurnPipelineException):
        validator.validate()


def test_invalid_target_values(valid_df, save_csv, make_validator):
    df = valid_df.copy()
    df[TARGET_COLUMN] = [0, 1, 2, 1]  

    csv_path = save_csv(df)
    validator = make_validator(csv_path)

    with pytest.raises(ChurnPipelineException):
        validator.validate()


def test_empty_dataset(save_csv, make_validator):
    empty_df = pd.DataFrame()
    csv_path = save_csv(empty_df)

    validator = make_validator(csv_path)

    with pytest.raises(ChurnPipelineException):
        validator.validate()


def test_duplicate_rows_allowed(valid_df, save_csv, make_validator):
    df = pd.concat([valid_df, valid_df.iloc[[0]]], ignore_index=True)
    csv_path = save_csv(df)

    validator = make_validator(csv_path)
    staged_path = validator.validate()

    assert staged_path.exists()


def test_severe_class_imbalance_allowed(valid_df, save_csv, make_validator):
    df = valid_df.copy()
    df[TARGET_COLUMN] = [0, 0, 0, 1]

    csv_path = save_csv(df)

    validator = make_validator(
        csv_path,
        min_minority_ratio=0.4
    )

    staged_path = validator.validate()
    assert staged_path.exists()


def test_missing_values_allowed(valid_df, save_csv, make_validator):
    df = valid_df.copy()
    df.iloc[0, 0] = None

    csv_path = save_csv(df)

    validator = make_validator(csv_path)
    staged_path = validator.validate()

    assert staged_path.exists()


def test_validation_does_not_modify_data(valid_df, save_csv, make_validator):
    csv_path = save_csv(valid_df)

    validator = make_validator(csv_path)
    staged_path = validator.validate()

    original = pd.read_csv(csv_path)
    staged = pd.read_csv(staged_path)

    pd.testing.assert_frame_equal(original, staged)
