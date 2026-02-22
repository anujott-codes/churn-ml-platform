
import pandas as pd
import pytest

from src.components.feature_engineering import FeatureEngineer
from src.config.feature_config import (
    FEATURES_TO_CREATE,
    FEATURE_THRESHOLDS,
    DROP_COLUMNS,
    TARGET_COLUMN
)

@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({
        "Customer ID": [1, 2, 3, 3],
        "Tenure": [10, 0, 5, 5],
        "Total Spend": [1000, 500, 300, 300],
        "Support Calls": [1, 5, 3, 3],
        "Payment Delay": [0, 15, 2, 2],
        TARGET_COLUMN: [0, 1, 0, 0]
    })


@pytest.fixture
def staged_csv(tmp_path, sample_dataframe):
    input_path = tmp_path / "staged.csv"
    sample_dataframe.to_csv(input_path, index=False)
    return input_path


@pytest.fixture
def feature_engineer(staged_csv, tmp_path):
    return FeatureEngineer(
        input_path=staged_csv,
        output_filename="processed.csv",
        processed_dir=tmp_path,
        reports_dir=tmp_path
    )


def test_column_standardization(feature_engineer):
    df = feature_engineer._load_data()
    df = feature_engineer._rename_columns(df)

    assert "customer_id" in df.columns
    assert "total_spend" in df.columns
    assert all(col == col.lower() for col in df.columns)


def test_data_cleaning_removes_duplicates(feature_engineer):
    df = feature_engineer._load_data()
    df = feature_engineer._rename_columns(df)
    df = feature_engineer._clean_data(df)

    assert df.duplicated().sum() == 0
    assert len(df) == 3


def test_feature_creation_logic(feature_engineer):
    df = feature_engineer._load_data()
    df = feature_engineer._rename_columns(df)
    df = feature_engineer._clean_data(df)
    df = feature_engineer._business_feature_engineering(df)

    if "high_support_calls" in FEATURES_TO_CREATE:
        threshold = FEATURE_THRESHOLDS["high_support_calls"]
        expected = (df["support_calls"] >= threshold).astype(int)
        assert (df["high_support_calls"] == expected).all()

    if "payment_delay_flag" in FEATURES_TO_CREATE:
        threshold = FEATURE_THRESHOLDS["payment_delay_flag"]
        expected = (df["payment_delay"] > threshold).astype(int)
        assert (df["payment_delay_flag"] == expected).all()

    if "spend_per_month" in FEATURES_TO_CREATE:
        assert "spend_per_month" in df.columns


def test_column_dropping(feature_engineer):
    df = feature_engineer._load_data()
    df = feature_engineer._rename_columns(df)
    df = feature_engineer._clean_data(df)
    df = feature_engineer._business_feature_engineering(df)
    df = feature_engineer._drop_useless_columns(df)

    for col in DROP_COLUMNS:
        assert col not in df.columns


def test_feature_report_generation(feature_engineer, tmp_path):
    df = feature_engineer._load_data()
    df = feature_engineer._rename_columns(df)
    df = feature_engineer._clean_data(df)
    df = feature_engineer._business_feature_engineering(df)
    df = feature_engineer._drop_useless_columns(df)

    report = feature_engineer._generate_feature_report(df)

    # Verify report structure
    assert report["input_file"] == feature_engineer.input_path.name
    assert report["target_column"] == TARGET_COLUMN
    assert isinstance(report["features_created"], list)
    assert report["data_shape_change"]["final"]["rows"] == len(df)

    reports = list(tmp_path.glob("*feature_engineering_report.json"))
    assert len(reports) == 1


def test_feature_engineering_run_end_to_end(feature_engineer, tmp_path):
    output_path = feature_engineer.run()

    assert output_path.exists()
    processed_df = pd.read_csv(output_path)

    assert len(processed_df) > 0
    assert TARGET_COLUMN in processed_df.columns

   
    reports = list(tmp_path.glob("*feature_engineering_report.json"))
    assert len(reports) == 1
