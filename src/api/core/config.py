from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import computed_field
from typing import List


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # ----- App Settings -----
    APP_NAME: str = "Customer Churn System API"
    APP_VERSION: str = "1.0.0"
    ENV: str = "production"
    DEBUG: bool = False

    # ----- Server Settings -----
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1
    ALLOWED_ORIGINS: List[str] = ["*"]

    # ----- MLflow Settings -----
    MLFLOW_TRACKING_URI: str 
    MLFLOW_MODEL_NAME: str = "CustomerChurnModel"
    MLFLOW_MODEL_ALIAS: str = "champion"

    @computed_field
    @property
    def MLFLOW_MODEL_URI(self) -> str:
        return f"models:/{self.MLFLOW_MODEL_NAME}@{self.MLFLOW_MODEL_ALIAS}"

    # ----- Prediction Settings -----
    PREDICTION_THRESHOLD: float = 0.5

    # ----- Thresholds for feature engineering -----
    HIGH_SUPPORT_CALLS_THRESHOLD: int = 4
    PAYMENT_DELAY_THRESHOLD: int = 0

    # ----- Batch prediction settings -----
    MAX_BATCH_ROWS: int = 10000
    INTEGER_FEATURES: List[str] = ["age", "tenure", "usage_frequency","support_calls","payment_delay", "last_interaction", "high_support_calls","payment_delay_flag"]
    FLOAT_FEATURES: List[str] = ["total_spend", "spend_per_month"]


settings = Settings()