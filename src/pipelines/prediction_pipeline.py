import mlflow
import mlflow.pyfunc
import pandas as pd
from typing import Union, List, Dict

from src.config.prediction_config import (
    MODEL_NAME,
    MODEL_ALIAS,
    RAW_FEATURES,
    FEATURE_LOGIC,
    ALL_FEATURES
)
from src.exception import ChurnPipelineException
from src.logging.logger import logger


class PredictionPipeline:
    """
    Loads model from MLflow registry using alias.
    Validates schema.
    Supports single and batch inference.
    """

    def __init__(self):
        try:
            self.model = self._load_model()
            logger.info("PredictionPipeline initialized successfully.")
        except Exception as e:
            raise ChurnPipelineException(
                f"Failed to initialize PredictionPipeline: {e}"
            )

    def _load_model(self):
        """
        Loads model from MLflow registry using alias.
        Fails fast if model unavailable.
        """
        try:
            model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
            logger.info(f"Loading model from MLflow using alias: {model_uri}")

            model = mlflow.pyfunc.load_model(model_uri)

            logger.info("Model loaded successfully from MLflow.")
            return model

        except Exception as e:
            raise ChurnPipelineException(
                f"Error loading model from MLflow using alias '{MODEL_ALIAS}': {e}"
            )

    def _validate_raw_input(self, df: pd.DataFrame):
        """
        Strict schema validation.
        """
        try:
            missing_cols = set(RAW_FEATURES) - set(df.columns)
            extra_cols = set(df.columns) - set(RAW_FEATURES)

            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            if extra_cols:
                raise ValueError(f"Unexpected extra columns: {extra_cols}")

            # Enforce strict column order
            df = df[RAW_FEATURES]

            return df

        except Exception as e:
            raise ChurnPipelineException(f"Input validation failed: {e}")
        
    def _feature_input(self, df: pd.DataFrame):
        """
        Creates features based on logic defined.
        """
        try:
            for feature_name, logic in FEATURE_LOGIC.items():
                df[feature_name] = logic(df)
            return df
        except Exception as e:
            raise ChurnPipelineException(f"Feature creation failed: {e}")

    def predict(
        self,
        input_data: Union[Dict, List[Dict], pd.DataFrame]
    ) -> List[Dict]:
        """
        Performs prediction and returns structured output.
        Supports:
        - single dict
        - list of dicts
        - DataFrame
        """

        try:
            # Normalize input
            if isinstance(input_data, dict):
                df = pd.DataFrame([input_data])

            elif isinstance(input_data, list):
                df = pd.DataFrame(input_data)

            elif isinstance(input_data, pd.DataFrame):
                df = input_data.copy()

            else:
                raise ValueError(
                    "Invalid input type. Must be dict, list of dicts, or DataFrame."
                )

            df = self._validate_raw_input(df)
            df = self._feature_input(df)

            #neforce strict column order for model input
            df = df[ALL_FEATURES]
            
            # Predict
            predictions = self.model.predict(df)

            # Extract probability if available
            try:
                probabilities = self.model.predict_proba(df)[:, 1]
            except Exception:
                probabilities = [None] * len(predictions)

            results = []

            for pred, prob in zip(predictions, probabilities):
                results.append({
                    "prediction": int(pred),
                    "label": "Churn" if int(pred) == 1 else "No Churn",
                    "probability": float(prob) if prob is not None else None
                })

            return results

        except Exception as e:
            raise ChurnPipelineException(f"Prediction failed: {e}")
