from fastapi import Request
from src.api.services.prediction_service import PredictionService


def get_prediction_service(request: Request) -> PredictionService:
    return request.app.state.prediction_service