from fastapi import Request, status
from fastapi.responses import JSONResponse
from src.api.core.logger import get_logger

logger = get_logger(__name__)

class APIException(Exception):
    """Base API exception"""
    pass


class ValidationException(APIException):
    """Raised for client-side validation errors"""
    pass


class PredictionException(APIException):
    """Raised when prediction pipeline fails"""
    pass


class ModelNotLoadedException(APIException):
    """Raised when model is not available"""
    pass

def register_exception_handlers(app):

    @app.exception_handler(ValidationException)
    async def validation_exception_handler(request: Request, exc: ValidationException):
        logger.warning(f"Validation error: {str(exc)}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": str(exc)}
        )

    @app.exception_handler(PredictionException)
    async def prediction_exception_handler(request: Request, exc: PredictionException):
        logger.error(f"Prediction error: {str(exc)}")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": str(exc)}
        )

    @app.exception_handler(ModelNotLoadedException)
    async def model_not_loaded_handler(request: Request, exc: ModelNotLoadedException):
        logger.critical("Model not loaded")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": "Model not loaded"}
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled exception occurred")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"}
        )