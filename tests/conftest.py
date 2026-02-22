import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from src.api.main import create_app
from src.api.dependencies import get_prediction_service


# Fake Services
class FakeModelService:
    def __init__(self):
        self.model = True

    def predict(self, df):
        return [0.8 for _ in range(len(df))]

    def get_model_info(self):
        return {
            "model_name": "CustomerChurnModel",
            "version": "5",
            "alias": "champion",
        }


class FakePredictionService:
    def __init__(self):
        self.model_service = FakeModelService()

    def predict(self, data):
        return {
            "prediction": 1,
            "probability": 0.8,
            "churn": True,
        }

    def predict_batch(self, df):
        return [
            {
                "prediction": 1,
                "probability": 0.8,
                "churn": True,
            }
            for _ in range(len(df))
        ]


@pytest.fixture(autouse=True)
def mock_mlflow():
    from unittest.mock import patch, MagicMock

    with patch("src.api.services.model_service.mlflow.pyfunc.load_model") as mock_load, \
         patch("src.api.services.model_service.MlflowClient") as mock_client_class:

        # ---- Fake Model ----
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.8]

        # ---- Fake Signature ----
        class FakeCol:
            def __init__(self, name):
                self.name = name

        fake_signature = MagicMock()
        fake_signature.inputs = [
            FakeCol("age"),
            FakeCol("gender"),
            FakeCol("tenure"),
            FakeCol("usage_frequency"),
            FakeCol("support_calls"),
            FakeCol("payment_delay"),
            FakeCol("subscription_type"),
            FakeCol("contract_length"),
            FakeCol("total_spend"),
            FakeCol("last_interaction"),
        ]

        fake_metadata = MagicMock()
        fake_metadata.signature = fake_signature

        mock_model.metadata = fake_metadata
        mock_load.return_value = mock_model

        # ---- Fake Registry Client ----
        mock_client = MagicMock()
        mock_version = MagicMock()
        mock_version.version = "5"
        mock_client.get_model_version_by_alias.return_value = mock_version
        mock_client_class.return_value = mock_client

        yield


@pytest.fixture
def client():
    app = create_app()

    def override_dependency():
        return FakePredictionService()

    app.dependency_overrides[get_prediction_service] = override_dependency

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()