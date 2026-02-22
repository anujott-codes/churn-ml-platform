import pandas as pd
import numpy as np
from typing import List

from src.api.core.config import settings
from src.api.core.logger import get_logger
from src.api.services.model_service import ModelService

logger = get_logger(__name__)


class PredictionService:
    def __init__(self, model_service: ModelService):
        self.model_service = model_service

        # Load expected feature order 
        self.feature_columns = model_service.get_feature_columns()

        if not self.feature_columns:
            raise ValueError("FEATURE_COLUMNS not defined in model signature")

    def _to_dataframe(self, data: dict) -> pd.DataFrame:
        """Convert request dict to pandas DataFrame."""
        try:
            df = pd.DataFrame([data])
            return df
        except Exception as e:
            logger.exception("Failed to convert input to DataFrame")
            raise ValueError("Invalid input format") from e

    def _apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering (must match training expectations)."""
        try:
            # --- Feature Columns ---
            df["high_support_calls"] = (df["support_calls"] > settings.HIGH_SUPPORT_CALLS_THRESHOLD).astype(int)
            df["payment_delay_flag"] = (df["payment_delay"] > settings.PAYMENT_DELAY_THRESHOLD).astype(int)

            df["spend_per_month"] = np.where(
                df["tenure"] > 0,
                df["total_spend"] / df["tenure"],
                0.0
            )

            return df

        except Exception as e:
            logger.exception("Feature engineering failed")
            raise RuntimeError("Feature engineering error") from e
        
    def _enforce_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure correct dtypes for model input."""
        try:
            for col in settings.INTEGER_FEATURES:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

            for col in settings.FLOAT_FEATURES:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype(float)

            return df

        except Exception as e:
            logger.exception("Failed to enforce dtypes")
            raise RuntimeError("Data type enforcement error") from e

    def _align_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure correct column order and presence."""
        try:
            missing_cols = set(self.feature_columns) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Missing required features: {missing_cols}")

            df = df[self.feature_columns]
            return df

        except Exception:
            logger.exception("Feature alignment failed")
            raise

    def predict(self, request_data: dict) -> dict:
        """Run full prediction pipeline for a single record."""
        try:
            # Step 1: Convert to DataFrame
            df = self._to_dataframe(request_data)

            # Step 2: Feature Engineering
            df = self._apply_feature_engineering(df)

            # Step 3: Enforce dtypes
            df = self._enforce_dtypes(df)

            # Step 4: Align features
            df = self._align_features(df)

            # Step 5: Model inference
            probs = self.model_service.predict(df)
            preds = [1 if probability >= settings.PREDICTION_THRESHOLD else 0 for probability in probs]

            # Step 6: Extract values
            prediction = int(preds[0])
            probability = float(probs[0])  

            churn = probability >= settings.PREDICTION_THRESHOLD

            result = {
                "prediction": prediction,
                "probability": probability,
                "churn": churn
            }

            return result

        except Exception as e:
            logger.exception("Prediction failed")
            raise RuntimeError("Prediction pipeline failed") from e
        
    def _validate_columns(self, df: pd.DataFrame):
        expected_cols = set(self.feature_columns)

        missing = expected_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

    def predict_batch(self, df: pd.DataFrame) -> List[dict]:
        """Run batch prediction."""
        try:
            if df.empty:
                raise ValueError("Input dataframe is empty")
            # Step 1: Feature Engineering
            df = self._apply_feature_engineering(df)
            
            # Step 2: Validate columns
            self._validate_columns(df)


            # Step 3: Enforce dtypes
            df = self._enforce_dtypes(df)

            # Step 4: Align features
            df = self._align_features(df)

            # Step 5: Model inference
            probs = self.model_service.predict(df)
            preds = [1 if probability >= settings.PREDICTION_THRESHOLD else 0 for probability in probs]

            results = []

            for pred, prob in zip(preds, probs):
                probability = float(prob)
                churn = probability >= settings.PREDICTION_THRESHOLD

                results.append({
                    "prediction": int(pred),
                    "probability": probability,
                    "churn": churn
                })

            return results

        except Exception as e:
            logger.exception("Batch prediction failed")
            raise RuntimeError("Batch prediction failed") from e