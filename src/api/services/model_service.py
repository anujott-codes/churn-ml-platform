import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from src.api.core.logger import get_logger
from src.api.core.exceptions import (
    ModelNotLoadedException,
    ValidationException,
    PredictionException,
)

logger = get_logger(__name__)


class ModelService:
    def __init__(self, model_uri: str):
        self.model_uri = model_uri
        self.model = None
        self.feature_columns = None
        self.signature = None
        self.model_version = None
        self.model_name = None
        self.model_alias = None

    from mlflow.tracking import MlflowClient

    def load_model(self):
        try:
            logger.info(f"Loading model from MLflow: {self.model_uri}")

            self.model = mlflow.pyfunc.load_model(self.model_uri)

            # Extract model name and alias
            if self.model_uri.startswith("models:/"):
                uri_part = self.model_uri.replace("models:/", "")
                if "@" in uri_part:
                    self.model_name, self.model_alias = uri_part.split("@")
                else:
                    self.model_name = uri_part

            # Fetch version from MLflow Registry
            client = MlflowClient()

            if self.model_alias:
                self.model_version = client.get_model_version_by_alias(self.model_name, self.model_alias).version


            # Extract signature and features
            self.signature = self.model.metadata.signature

            if self.signature is None:
                raise ValidationException("Model signature is not defined")

            inputs = self.signature.inputs
            if hasattr(inputs, "inputs"):
                inputs = inputs.inputs

            self.feature_columns = [col.name for col in inputs]

            logger.info(
                f"Model loaded successfully "
                f"(version={self.model_version}, alias={self.model_alias})"
            )

        except Exception:
            logger.exception("Failed to load model")
            raise ModelNotLoadedException("Failed to load model")

    def get_feature_columns(self):
        if self.feature_columns is None:
            raise ModelNotLoadedException(
                "Feature columns not initialized. Model may not be loaded."
            )

        return self.feature_columns

    def predict(self, data):
        if self.model is None:
            raise ModelNotLoadedException("Model is not loaded")

        try:
            return self.model.predict(data)

        except Exception:
            logger.exception("Model prediction failed")
            raise PredictionException("Model prediction failed")
        
    def get_model_info(self):
        if self.model is None:
            raise ModelNotLoadedException("Model not loaded")

        return {
            "model_name": self.model_name,
            "version": self.model_version,
            "alias": self.model_alias,
        }