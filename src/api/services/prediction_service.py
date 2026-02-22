import pandas as pd
import numpy as np
from typing import List

from src.api.core.config import settings
from src.api.core.logger import get_logger
from src.api.services.model_service import ModelService
from src.api.core.exceptions import (
    ValidationException,
    PredictionException,
)

logger = get_logger(__name__)


class PredictionService:
    def __init__(self, model_service: ModelService):
        self.model_service = model_service

        # Load expected feature order
        self.feature_columns = model_service.get_feature_columns()

        if not self.feature_columns:
            raise ValidationException(
                "Feature columns not defined in model signature"
            )

    def _to_dataframe(self, data: dict) -> pd.DataFrame:
        try:
            return pd.DataFrame([data])
        except Exception:
            logger.exception("Failed to convert input to DataFrame")
            raise ValidationException("Invalid input format")

    def _apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df["high_support_calls"] = (
                df["support_calls"] > settings.HIGH_SUPPORT_CALLS_THRESHOLD
            ).astype(int)

            df["payment_delay_flag"] = (
                df["payment_delay"] > settings.PAYMENT_DELAY_THRESHOLD
            ).astype(int)

            df["spend_per_month"] = np.where(
                df["tenure"] > 0,
                df["total_spend"] / df["tenure"],
                0.0
            )

            return df

        except Exception:
            logger.exception("Feature engineering failed")
            raise PredictionException("Feature engineering failed")

    def _enforce_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            for col in settings.INTEGER_FEATURES:
                if col in df.columns:
                    df[col] = (
                        pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
                    )

            for col in settings.FLOAT_FEATURES:
                if col in df.columns:
                    df[col] = (
                        pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float)
                    )

            return df

        except Exception:
            logger.exception("Failed to enforce dtypes")
            raise ValidationException("Data type enforcement failed")

    def _align_features(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            missing_cols = set(self.feature_columns) - set(df.columns)
            if missing_cols:
                raise ValidationException(
                    f"Missing required features: {missing_cols}"
                )

            return df[self.feature_columns]

        except ValidationException:
            raise
        except Exception:
            logger.exception("Feature alignment failed")
            raise PredictionException("Feature alignment failed")

    def _validate_columns(self, df: pd.DataFrame):
        expected_cols = set(self.feature_columns)
        missing = expected_cols - set(df.columns)

        if missing:
            raise ValidationException(f"Missing columns: {missing}")

    def predict(self, request_data: dict) -> dict:
        try:
            df = self._to_dataframe(request_data)
            df = self._apply_feature_engineering(df)
            df = self._enforce_dtypes(df)
            df = self._align_features(df)

            probs = self.model_service.predict(df)

            preds = [
                1 if probability >= settings.PREDICTION_THRESHOLD else 0
                for probability in probs
            ]

            prediction = int(preds[0])
            probability = float(probs[0])
            churn = probability >= settings.PREDICTION_THRESHOLD

            return {
                "prediction": prediction,
                "probability": probability,
                "churn": churn,
            }

        except ValidationException:
            raise
        except Exception:
            logger.exception("Prediction failed")
            raise PredictionException("Prediction pipeline failed")

    def predict_batch(self, df: pd.DataFrame) -> List[dict]:
        try:
            if df.empty:
                raise ValidationException("Input dataframe is empty")

            df = self._apply_feature_engineering(df)
            self._validate_columns(df)
            df = self._enforce_dtypes(df)
            df = self._align_features(df)

            probs = self.model_service.predict(df)

            preds = [
                1 if probability >= settings.PREDICTION_THRESHOLD else 0
                for probability in probs
            ]

            results = []

            for pred, prob in zip(preds, probs):
                probability = float(prob)
                churn = probability >= settings.PREDICTION_THRESHOLD

                results.append(
                    {
                        "prediction": int(pred),
                        "probability": probability,
                        "churn": churn,
                    }
                )

            return results

        except ValidationException:
            raise
        except Exception:
            logger.exception("Batch prediction failed")
            raise PredictionException("Batch prediction failed")