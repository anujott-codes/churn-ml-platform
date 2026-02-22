from fastapi import FastAPI
from contextlib import asynccontextmanager
import mlflow

from src.api.core.config import settings
from src.api.core.logger import get_logger
from src.api.services.model_service import ModelService
from src.api.services.prediction_service import PredictionService
from src.api.routers import predict, health, model_info
from src.api.core.exceptions import register_exception_handlers

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application startup and shutdown lifecycle manager.
    """

    # ----- Startup -----
    logger.info("Starting Customer Churn Prediction API...")
    logger.info(f"Environment: {settings.ENV}")
    logger.info(f"MLflow Tracking URI: {settings.MLFLOW_TRACKING_URI}")
    logger.info(f"Loading model from: {settings.MLFLOW_MODEL_URI}")

    try:
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)

        model_service = ModelService(settings.MLFLOW_MODEL_URI)
        model_service.load_model()

        prediction_service = PredictionService(model_service)
        app.state.model_service = model_service
        app.state.prediction_service = prediction_service

        try:
            dummy = {
                "age": 30,
                "gender": "Male",
                "tenure": 12,
                "usage_frequency": 7,
                "support_calls": 5,
                "payment_delay": 11,
                "subscription_type": "Basic",
                "contract_length": "Annual",
                "total_spend": 795,
                "last_interaction": 14
            }
            prediction_service.predict(dummy)
            logger.info("Model warmup inference successful.")
        except Exception:
            logger.warning("Model warmup skipped or failed.")

        logger.info("Model loaded successfully. API is ready.")

    except Exception as e:
        logger.exception("Failed during application startup.")
        raise e  

    yield

    # ----- Shutdown -----
    logger.info("Shutting down Customer Churn Prediction API...")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        description="ML serving layer for Customer Churn prediction",
        version=settings.APP_VERSION,
        docs_url="/docs", 
        redoc_url="/redoc",
        lifespan=lifespan
    )
    # Register exception handlers
    register_exception_handlers(app)

    # Include API routers
    app.include_router(predict.router)
    app.include_router(health.router)
    app.include_router(model_info.router)

    return app


app = create_app()