from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.concurrency import run_in_threadpool
import pandas as pd

from src.api.core.config import settings
from src.api.core.logger import get_logger
from src.api.schemas.request import CustomerPredictionRequest, BatchPredictionRequest
from src.api.schemas.response import PredictionResponse, BatchPredictionResponse
from src.api.services.prediction_service import PredictionService
from src.api.dependencies import get_prediction_service

router = APIRouter(prefix="/api/v1")
logger = get_logger(__name__)


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    payload: CustomerPredictionRequest,
    service: PredictionService = Depends(get_prediction_service)
):
    try:
        logger.info("Received single prediction request")

        result = await run_in_threadpool(
            service.predict,
            payload.model_dump()
        )

        return result

    except ValueError as e:
        logger.warning(f"Validation error in /predict: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception:
        logger.exception("Unexpected error in /predict")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/predict-batch", response_model=BatchPredictionResponse)
async def predict_batch(
    file: UploadFile = File(...),
    service: PredictionService = Depends(get_prediction_service)
):
    # Validate file type 
    if file.content_type != "text/csv":
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    try:
        logger.info(f"Received batch prediction file: {file.filename}")

        df = pd.read_csv(file.file)
        records = df.to_dict(orient="records")

        try:
            validated = BatchPredictionRequest(records=records)
        except Exception as e:
            logger.warning(f"Batch schema validation failed: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))

        # Convert back to DataFrame AFTER validation
        df = pd.DataFrame([r.model_dump() for r in validated.records])

    except Exception:
        logger.warning("CSV parsing failed")
        raise HTTPException(status_code=400, detail="Failed to parse CSV file.")

    # Basic validations
    if df.empty:
        raise HTTPException(status_code=400, detail="Uploaded CSV is empty.")

    if len(df) > settings.MAX_BATCH_ROWS:
        raise HTTPException(
            status_code=400,
            detail=f"CSV exceeds maximum allowed rows ({settings.MAX_BATCH_ROWS:,}). Got {len(df):,}."
        )

    try:
        logger.info(f"Processing batch of size: {len(df)}")

        results = await run_in_threadpool(
            service.predict_batch,
            df  
        )

        return {"predictions": results}

    except ValueError as e:
        logger.warning(f"Validation error in /predict-batch: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception:
        logger.exception("Unexpected error in /predict-batch")
        raise HTTPException(status_code=500, detail="Internal server error")