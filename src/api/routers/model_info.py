from fastapi import APIRouter, Depends
from src.api.schemas.response import ModelInfoResponse
from src.api.core.config import settings


router = APIRouter(prefix="/api/v1")


@router.get("/model-info", response_model=ModelInfoResponse)
async def model_info():
    return {
        "model_name": settings.MLFLOW_MODEL_NAME,
        "version": "unknown",
        "alias": settings.MLFLOW_MODEL_ALIAS
    }