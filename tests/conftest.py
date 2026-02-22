import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.dependencies import get_prediction_service


# Fake Service for testing api endpoints
class FakeModelService:
    def __init__(self):
        self.model = True

    def predict(self, df):
        # return probability list
        return [0.8 for _ in range(len(df))]

    def get_model_info(self):
        return {
            "model_name": "CustomerChurnModel",
            "version": "5",
            "alias": "champion"
        }


class FakePredictionService:
    def __init__(self):
        self.model_service = FakeModelService()

    def predict(self, data):
        return {
            "prediction": 1,
            "probability": 0.8,
            "churn": True
        }

    def predict_batch(self, df):
        return [
            {
                "prediction": 1,
                "probability": 0.8,
                "churn": True
            }
            for _ in range(len(df))
        ]


@pytest.fixture
def client():
    def override_dependency():
        return FakePredictionService()

    app.dependency_overrides[get_prediction_service] = override_dependency

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()