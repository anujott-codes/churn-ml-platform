import pandas as pd
import pytest
import joblib
from pathlib import Path

from src.components.data_transformation import DataTransformation
from src.exception import ChurnPipelineException
from src.config.feature_config import TARGET_COLUMN


@pytest.fixture
def sample_train_data():
    return pd.DataFrame({
        # Numerical 
        "age": [25, 30, 35, 40, 45],
        "tenure": [1, 2, 3, 4, 5],
        "usage_frequency": [10, 20, 30, 40, 50],
        "support_calls": [0, 1, 2, 3, 4],
        "payment_delay": [0, 1, 0, 2, 1],
        "total_spend": [100, 200, 300, 400, 500],
        "last_interaction": [5, 4, 3, 2, 1],
        "high_support_calls": [0, 0, 1, 1, 1],
        "payment_delay_flag": [0, 1, 0, 1, 1],
        "spend_per_month": [100, 100, 100, 100, 100],

        # Nominal 
        "gender": ["Male", "Female", "Male", "Female", "Male"],
        "subscription_type": ["Basic", "Pro", "Basic", "Premium", "Pro"],

        # Ordinal 
        "contract_length": ["Monthly", "Quaterly", "Yearly", "Monthly", "Yearly"],

        # Target
        TARGET_COLUMN: [0, 1, 0, 1, 0]
    })


@pytest.fixture
def sample_test_data():
    return pd.DataFrame({
        "age": [28, 38],
        "tenure": [2, 4],
        "usage_frequency": [15, 35],
        "support_calls": [1, 3],
        "payment_delay": [0, 1],
        "total_spend": [150, 350],
        "last_interaction": [4, 2],
        "high_support_calls": [0, 1],
        "payment_delay_flag": [0, 1],
        "spend_per_month": [75, 87.5],

        "gender": ["Male", "Female"],
        "subscription_type": ["Basic", "Premium"],

        "contract_length": ["Monthly", "Yearly"],

        TARGET_COLUMN: [0, 1]
    })


@pytest.fixture
def processed_files(tmp_path, sample_train_data, sample_test_data):
    train_path = tmp_path / "processed_train.csv"
    test_path = tmp_path / "processed_test.csv"

    sample_train_data.to_csv(train_path, index=False)
    sample_test_data.to_csv(test_path, index=False)

    return train_path, test_path


@pytest.fixture
def transformer(processed_files, tmp_path):
    train_path, test_path = processed_files
    return DataTransformation(
        processed_train_path=train_path,
        processed_test_path=test_path,
        transformed_dir=tmp_path / "transformed",
        preprocessing_dir=tmp_path / "preprocessing"
    )


def test_initialization(processed_files):
    train_path, test_path = processed_files
    dt = DataTransformation(train_path, test_path)

    assert dt.processed_train_path == train_path
    assert dt.processed_test_path == test_path


def test_preprocessor_builds(transformer):
    preprocessor = transformer._build_preprocessor()

    assert preprocessor is not None
    assert len(preprocessor.transformers) == 3


def test_preprocessor_transforms_shape(transformer, sample_train_data):
    preprocessor = transformer._build_preprocessor()

    X = sample_train_data.drop(columns=[TARGET_COLUMN])
    X_transformed = preprocessor.fit_transform(X)

    assert X_transformed.shape[0] == len(sample_train_data)

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_unknown_categories_do_not_crash(transformer, sample_train_data):
    preprocessor = transformer._build_preprocessor()

    X_train = sample_train_data.drop(columns=[TARGET_COLUMN])
    preprocessor.fit(X_train)

    X_test = X_train.copy()
    X_test["gender"] = ["Unknown"] * len(X_test)

    X_transformed = preprocessor.transform(X_test)
    assert X_transformed is not None


def test_data_transformation_end_to_end(transformer, tmp_path):
    result = transformer.initiate_data_transformation()
    
    assert result["train_path"].exists()
    assert result["test_path"].exists()
    assert result["preprocessor_path"].exists()

    train_df = pd.read_csv(result["train_path"])
    test_df = pd.read_csv(result["test_path"])

    assert TARGET_COLUMN in train_df.columns
    assert TARGET_COLUMN in test_df.columns

    preprocessor = joblib.load(result["preprocessor_path"])
    assert preprocessor is not None


def test_missing_input_file_raises_exception(tmp_path):
    train_path = tmp_path / "missing_train.csv"
    test_path = tmp_path / "missing_test.csv"

    dt = DataTransformation(train_path, test_path)

    with pytest.raises(ChurnPipelineException):
        dt.initiate_data_transformation()
