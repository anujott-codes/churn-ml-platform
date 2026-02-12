import json
import pickle
from pathlib import Path
from datetime import datetime,timezone

import pandas as pd
import lightgbm as lgb

from src.config.feature_config import TARGET_COLUMN
from src.config.basic_config import (
    TRANSFORMED_DATA_DIR,
    MODEL_DIR,
    MODEL_REPORTS_DIR
)
from src.config.tuner_config import (
    TRAIN_DATA_FILENAME,
    BEST_PARAMS_FILENAME
)

from src.config.trainer_config import (
        MODEL_FILENAME,
        MODEL_TYPE,
        MODEL_METADATA_FILENAME,
        FEATURE_SCHEMA_FILENAME,
    )
from src.logging.logger import logger
from src.exception import ChurnPipelineException

class ModelTrainer:
    def __init__(self,
        best_params_path: Path = MODEL_REPORTS_DIR / BEST_PARAMS_FILENAME,
        train_data_path: Path = TRANSFORMED_DATA_DIR / TRAIN_DATA_FILENAME, model_dir: Path = MODEL_DIR,
        model_filename: str = MODEL_FILENAME,
        model_type: str = MODEL_TYPE
    ):

        self.best_params_path = best_params_path
        self.train_data_path = train_data_path
        self.model_dir = model_dir
        self.model_path = self.model_dir / model_filename
        self.metadata_path = model_dir / MODEL_METADATA_FILENAME
        self.schema_path = model_dir / FEATURE_SCHEMA_FILENAME
        self.model_type = model_type    


    def load_data(self):
        try:
            data = pd.read_csv(self.train_data_path)
            if TARGET_COLUMN not in data.columns:
                raise ChurnPipelineException(
                    f"{TARGET_COLUMN} not found in dataset."
                )
            X = data.drop(TARGET_COLUMN, axis=1)
            y = data[TARGET_COLUMN]
            return X, y
        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            raise ChurnPipelineException(e)
        
    def load_best_params(self):
        try:
            if not self.best_params_path.exists():
                raise ChurnPipelineException(
                    f"Best params file not found at {self.best_params_path}"
                )

            with open(self.best_params_path, "r") as f:
                best_params = json.load(f)

            if not isinstance(best_params, dict):
                raise ChurnPipelineException("best_params must be a dictionary.")
            
            required_keys = ["n_estimators", "objective", "random_state"]
            for key in required_keys:
                if key not in best_params:
                    raise ChurnPipelineException(f"{key} missing in best_params.")

            return best_params

        except Exception as e:
            logger.error(f"Error loading best parameters: {e}")
            raise ChurnPipelineException(e)
        
    def build_model(self, params):
        model_registry = {   
            "lightgbm": lgb.LGBMClassifier,
        }

        if self.model_type not in model_registry:
            raise ChurnPipelineException(f"Unsupported model type: {self.model_type}")
        return model_registry[self.model_type](**params)

    def save_model(self, model):

        self.model_dir.mkdir(parents=True, exist_ok=True)

        with open(self.model_path, "wb") as f:
            pickle.dump(model, f)

    def save_metadata(self, metadata: dict):
        with open(self.metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

    def save_schema(self, feature_columns):
        with open(self.schema_path, "w") as f:
            json.dump(list(feature_columns), f, indent=4)

    def train_model(self):
        try:
            logger.info("Starting model training...")
            X_train, y_train = self.load_data()

            # Recalculate imbalance for full dataset
            pos = y_train.sum()
            neg = len(y_train) - pos
            scale_pos_weight = neg / pos if pos > 0 else 1.0

            best_params = self.load_best_params().copy()
            best_params["scale_pos_weight"] = scale_pos_weight


            model = self.build_model(best_params)
            model.fit(X_train, y_train)

            self.save_model(model)
            self.save_schema(X_train.columns)

            metadata = {
                "model_type": self.model_type,
                "train_rows": len(X_train),
                "n_features": X_train.shape[1],
                "training_timestamp": datetime.now(timezone.utc).isoformat(),
                "best_params": best_params
            }

            self.save_metadata(metadata)

            logger.info(f"Model saved at: {self.model_path}")
            logger.info("Model training completed successfully.")

            return self.model_path

        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise ChurnPipelineException(e)