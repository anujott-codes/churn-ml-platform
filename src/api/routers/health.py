from fastapi import APIRouter, Depends
from src.api.schemas.response import HealthResponse
from src.api.services.prediction_service import PredictionService
from src.api.dependencies import get_prediction_service
from src.api.core.exceptions import ModelNotLoadedException

router = APIRouter(prefix="/api/v1")


@router.get("/health", response_model=HealthResponse)
async def health(
    service: PredictionService = Depends(get_prediction_service)
):
    if service.model_service.model is None:
        raise ModelNotLoadedException("Model not loaded")

    return {"status": "OK"}