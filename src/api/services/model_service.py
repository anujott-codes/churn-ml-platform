import mlflow.pyfunc
from src.api.core.logger import get_logger

logger = get_logger(__name__)


class ModelService:
    def __init__(self, model_uri: str):
        self.model_uri = model_uri
        self.model = None
        self.feature_columns = None
        self.signature = None

    def load_model(self):
        try:
            logger.info(f"Loading model from MLflow: {self.model_uri}")
            
            self.model = mlflow.pyfunc.load_model(self.model_uri)

            # Extract signature and feature columns
            self.signature = self.model.metadata.signature
            if self.signature is None:
                raise ValueError("Model signature is not defined in MLflow")

            inputs = self.signature.inputs
            if hasattr(inputs, "inputs"):  
                inputs = inputs.inputs

            self.feature_columns = [col.name for col in inputs]
            if not self.feature_columns:
                raise ValueError("Failed to extract feature columns from signature")

            logger.info(f"Model loaded successfully with features: {self.feature_columns}")

        except Exception:
            logger.exception("Failed to load model from MLflow")
            raise

    def get_feature_columns(self):
        if self.feature_columns is None:
            raise RuntimeError("Feature columns not initialized. Did you call load_model()?")

        return self.feature_columns

    def predict(self, data):
        if self.model is None:
            raise RuntimeError("Model is not loaded")

        return self.model.predict(data)

    def predict_proba(self, data):
        if self.model is None:
            raise RuntimeError("Model is not loaded")

        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(data)

        raise NotImplementedError("This model does not support predict_proba")

    