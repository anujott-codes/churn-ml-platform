from fastapi import APIRouter, Depends
from src.api.schemas.response import ModelInfoResponse
from src.api.services.prediction_service import PredictionService
from src.api.dependencies import get_prediction_service


router = APIRouter(prefix="/api/v1")


@router.get("/model-info", response_model=ModelInfoResponse)
async def model_info(
    service: PredictionService = Depends(get_prediction_service)
):
    return service.model_service.get_model_info()