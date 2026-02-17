from pathlib import Path
import pickle

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidator
from src.components.feature_engineering import FeatureEngineer
from src.components.model_tuner import ModelTuner
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluator import ModelEvaluator

from src.config.basic_config import (
    RAW_DATA_DIR,
    STAGED_DATA_DIR,
    PROCESSED_DATA_DIR,
    VALIDATION_REPORTS_DIR,
    FEATURE_REPORTS_DIR,
    MODEL_REPORTS_DIR
)

from src.config.data_source_config import (
    TRAIN_FILENAME,
    TEST_FILENAME
)

from src.config.tuner_config import (
    TRAIN_DATA_FILENAME,
    BEST_PARAMS_FILENAME
)

import mlflow
import mlflow.sklearn
from src.config.mlflow_config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME
)

from src.exception import ChurnPipelineException
from src.logging.logger import logger

from dotenv import load_dotenv
load_dotenv()


class TrainingPipeline:

    def __init__(self):
        self.raw_train_path = RAW_DATA_DIR / TRAIN_FILENAME
        self.raw_test_path = RAW_DATA_DIR / TEST_FILENAME

        self.staged_dir = STAGED_DATA_DIR
        self.processed_dir = PROCESSED_DATA_DIR

        self.validation_reports_dir = VALIDATION_REPORTS_DIR
        self.feature_reports_dir = FEATURE_REPORTS_DIR

        self.train_data_path = PROCESSED_DATA_DIR / TRAIN_DATA_FILENAME
        self.best_params_path = MODEL_REPORTS_DIR / BEST_PARAMS_FILENAME


    # STAGE 1: INGESTION
    def run_data_ingestion(self):
        logger.info("STAGE 1: DATA INGESTION")
        data_ingestion = DataIngestion()
        data_ingestion.initiate_data_ingestion()
        logger.info("Data ingestion completed")

    # STAGE 2: VALIDATION
    def run_data_validation(self):
        logger.info("STAGE 2: DATA VALIDATION")

        train_validator = DataValidator(
            raw_data_path=self.raw_train_path,
            staged_dir=self.staged_dir,
            reports_dir=self.validation_reports_dir
        )
        staged_train_path = train_validator.validate()

        test_validator = DataValidator(
            raw_data_path=self.raw_test_path,
            staged_dir=self.staged_dir
        )
        staged_test_path = test_validator.validate()

        logger.info("Data validation completed")
        return staged_train_path, staged_test_path


    # STAGE 3: FEATURE ENGINEERING
    def run_feature_engineering(self, staged_train_path, staged_test_path):
        logger.info("STAGE 3: FEATURE ENGINEERING")

        train_engineer = FeatureEngineer(
            input_path=staged_train_path,
            output_filename="train_processed.csv",
            processed_dir=self.processed_dir,
            reports_dir=self.feature_reports_dir
        )
        processed_train_path = train_engineer.run()

        test_engineer = FeatureEngineer(
            input_path=staged_test_path,
            output_filename="test_processed.csv",
            processed_dir=self.processed_dir
        )
        processed_test_path = test_engineer.run()

        logger.info("Feature engineering completed")
        return processed_train_path, processed_test_path


    # STAGE 4: MODEL TUNING
    def run_model_tuning(self):
        logger.info("STAGE 4: MODEL TUNING")
        model_tuner = ModelTuner()
        best_params = model_tuner.tune()
        logger.info("Model tuning completed")
        return best_params


    # STAGE 5: MODEL TRAINING
    def run_model_training(self):
        logger.info("STAGE 5: MODEL TRAINING")

        model_trainer = ModelTrainer(
            train_data_path=self.train_data_path,
            best_params_path=self.best_params_path
        )

        pipeline_path = model_trainer.train_model()
        logger.info("Model training completed")
        return pipeline_path


    # STAGE 6: MODEL EVALUATION
    def run_model_evaluation(self):
        logger.info("STAGE 6: MODEL EVALUATION")

        model_evaluator = ModelEvaluator()
        report, cm = model_evaluator.evaluate()

        logger.info("Model evaluation completed")
        return report, cm


    # FULL PIPELINE EXECUTION
    def run(self):

        try:
            logger.info("--------------- TRAINING PIPELINE STARTED ---------------")

            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

            with mlflow.start_run():

                # 1. Ingestion
                self.run_data_ingestion()

                # 2. Validation
                staged_train_path, staged_test_path = self.run_data_validation()

                # 3. Feature Engineering
                self.run_feature_engineering(staged_train_path, staged_test_path)

                # 4. Tuning
                best_params = self.run_model_tuning()
                mlflow.log_params(best_params)

                # 5. Training 
                pipeline_path = self.run_model_training()

                # 6. Evaluation
                evaluation_report, cm = self.run_model_evaluation()

                # MLflow Logging
                mlflow.log_metrics({
                    "roc_auc": evaluation_report["roc_auc"],
                    "pr_auc": evaluation_report["pr_auc"],
                    "precision": evaluation_report["precision"],
                    "recall": evaluation_report["recall"],
                    "f1_score": evaluation_report["f1_score"],
                    "accuracy": evaluation_report["accuracy"],
                    "precision_at_k": evaluation_report["precision_at_k"],
                })

                # Log full sklearn pipeline
                with open(pipeline_path, "rb") as f:
                    pipeline = pickle.load(f)

                mlflow.sklearn.log_model(
                    pipeline,
                    artifact_path="model",
                    registered_model_name="CustomerChurnModel"
                )

                mlflow.log_artifact(str(self.best_params_path), artifact_path="params")
                mlflow.log_artifact(str(self.train_data_path), artifact_path="data")

            logger.info("--------------- TRAINING PIPELINE COMPLETED SUCCESSFULLY ---------------")

        except Exception as e:
            logger.error("--------------- TRAINING PIPELINE FAILED ---------------")
            raise ChurnPipelineException(e)


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run()
