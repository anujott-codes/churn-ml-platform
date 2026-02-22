from fastapi import APIRouter, UploadFile, File, Depends
from fastapi.concurrency import run_in_threadpool
import pandas as pd

from src.api.core.config import settings
from src.api.core.logger import get_logger
from src.api.core.exceptions import ValidationException
from src.api.schemas.request import (
    CustomerPredictionRequest,
    BatchPredictionRequest,
)
from src.api.schemas.response import (
    PredictionResponse,
    BatchPredictionResponse,
)
from src.api.services.prediction_service import PredictionService
from src.api.dependencies import get_prediction_service

router = APIRouter(prefix="/api/v1")
logger = get_logger(__name__)


# ------ Single Prediction ------ 
@router.post("/predict", response_model=PredictionResponse)
async def predict(
    payload: CustomerPredictionRequest,
    service: PredictionService = Depends(get_prediction_service),
):
    logger.info("Received single prediction request")

    return await run_in_threadpool(
        service.predict,
        payload.model_dump(),
    )


# ------ Batch Prediction ------ 
@router.post("/predict-batch", response_model=BatchPredictionResponse)
async def predict_batch(
    file: UploadFile = File(...),
    service: PredictionService = Depends(get_prediction_service),
):
    logger.info(f"Received batch prediction file: {file.filename}")

    # Validate file type
    if file.content_type != "text/csv":
        raise ValidationException("Only CSV files are accepted.")

    # Read CSV
    try:
        df = pd.read_csv(file.file)
    except Exception:
        logger.warning("CSV parsing failed")
        raise ValidationException("Failed to parse CSV file.")

    if df.empty:
        raise ValidationException("Uploaded CSV is empty.")

    if len(df) > settings.MAX_BATCH_ROWS:
        raise ValidationException(
            f"CSV exceeds maximum allowed rows ({settings.MAX_BATCH_ROWS:,}). "
            f"Got {len(df):,}."
        )

    # Schema validation
    records = df.to_dict(orient="records")

    try:
        validated = BatchPredictionRequest(records=records)
    except Exception as e:
        logger.warning(f"Batch schema validation failed: {str(e)}")
        raise ValidationException(str(e))

    # Convert validated records back to DataFrame
    df = pd.DataFrame([r.model_dump() for r in validated.records])

    logger.info(f"Processing batch of size: {len(df)}")

    results = await run_in_threadpool(
        service.predict_batch,
        df,
    )

    return {"predictions": results}