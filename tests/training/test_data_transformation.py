import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from src.components.data_transformation import DataTransformation
from src.config.feature_config import TARGET_COLUMN


@pytest.fixture
def sample_data():
    return pd.DataFrame({
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


def test_get_preprocessor_returns_column_transformer():
    dt = DataTransformation()
    preprocessor = dt.get_preprocessor()

    assert isinstance(preprocessor, ColumnTransformer)


def test_preprocessor_contains_expected_transformers():
    dt = DataTransformation()
    preprocessor = dt.get_preprocessor()

    transformer_names = [name for name, _, _ in preprocessor.transformers]

    assert "num" in transformer_names
    assert "nom" in transformer_names
    assert "ord" in transformer_names
    assert len(transformer_names) == 3


def test_preprocessor_fit_transform_shape(sample_data):
    dt = DataTransformation()
    preprocessor = dt.get_preprocessor()

    X = sample_data.drop(columns=[TARGET_COLUMN])
    transformed = preprocessor.fit_transform(X)

    assert transformed.shape[0] == len(sample_data)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_unknown_nominal_category_handled(sample_data):
    dt = DataTransformation()
    preprocessor = dt.get_preprocessor()

    X = sample_data.drop(columns=[TARGET_COLUMN])
    preprocessor.fit(X)

    X_test = X.copy()
    X_test["gender"] = ["Unknown"] * len(X_test)

    transformed = preprocessor.transform(X_test)

    assert transformed is not None
    assert transformed.shape[0] == len(X_test)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_unknown_ordinal_category_encoded(sample_data):
    dt = DataTransformation()
    preprocessor = dt.get_preprocessor()

    X = sample_data.drop(columns=[TARGET_COLUMN])
    preprocessor.fit(X)

    X_test = X.copy()
    X_test["contract_length"] = ["Invalid"] * len(X_test)

    transformed = preprocessor.transform(X_test)

    assert transformed is not None
    assert transformed.shape[0] == len(X_test)
