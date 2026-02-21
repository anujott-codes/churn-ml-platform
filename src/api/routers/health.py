from fastapi import APIRouter, Depends
from src.api.schemas.response import HealthResponse
from src.api.services.prediction_service import PredictionService
from src.api.dependencies import get_prediction_service

router = APIRouter(prefix="/api/v1")


@router.get("/health", response_model=HealthResponse)
async def health(
    service: PredictionService = Depends(get_prediction_service)
):
    model_loaded = (
        service is not None
        and service.model_service is not None
        and service.model_service.model is not None
    )

    return {
        "status": "OK" if model_loaded else "degraded"
    }