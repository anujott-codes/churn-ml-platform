from pathlib import Path

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidator
from src.components.feature_engineering import FeatureEngineer
from src.components.data_transformation import DataTransformation
from src.components.model_tuner import ModelTuner

from src.config.basic_config import (
    RAW_DATA_DIR,
    STAGED_DATA_DIR,
    PROCESSED_DATA_DIR,
    VALIDATION_REPORTS_DIR,
    FEATURE_REPORTS_DIR,
    TRANSFORMED_DATA_DIR,
    PREPROCESSING_DIR
)

from src.config.data_source_config import (
    TRAIN_FILENAME,
    TEST_FILENAME
)

from src.exception import ChurnPipelineException
from src.logging.logger import logger


class TrainingPipeline:
    """
    End-to-end training pipeline for customer churn prediction.
    
    Pipeline stages:
    1. Data Ingestion - Download and extract data from Kaggle
    2. Data Validation - Validate train and test datasets
    3. Feature Engineering - Create features for train and test
    4. Model Training - Train ML model (to be implemented)
    5. Model Evaluation - Evaluate model performance (to be implemented)
    """
    
    def __init__(self):
        self.raw_train_path = RAW_DATA_DIR / TRAIN_FILENAME
        self.raw_test_path = RAW_DATA_DIR / TEST_FILENAME
        self.staged_dir = STAGED_DATA_DIR
        self.processed_dir = PROCESSED_DATA_DIR
        self.validation_reports_dir = VALIDATION_REPORTS_DIR
        self.feature_reports_dir = FEATURE_REPORTS_DIR
        self.transformed_data_dir = TRANSFORMED_DATA_DIR
        self.preprocessing_dir = PREPROCESSING_DIR

    def run_data_ingestion(self) -> None:
        
        logger.info("STAGE 1: DATA INGESTION")
        
        try:
            data_ingestion = DataIngestion()
            data_ingestion.initiate_data_ingestion()
            logger.info("Data ingestion completed")
        except Exception as e:
            logger.error("Data ingestion failed")
            raise ChurnPipelineException(e)

    def run_data_validation(self) -> tuple[Path, Path]:
        logger.info("STAGE 2: DATA VALIDATION")
        
        try:
            logger.info("Validating TRAIN dataset")
            train_validator = DataValidator(
                raw_data_path=self.raw_train_path,
                staged_dir=self.staged_dir,
                reports_dir=self.validation_reports_dir
            )
            staged_train_path = train_validator.validate()
            
            logger.info("Validating TEST dataset")
            test_validator = DataValidator(
                raw_data_path=self.raw_test_path,
                staged_dir=self.staged_dir
            )
            staged_test_path = test_validator.validate()
            
            logger.info("Data validation completed for both datasets")
            return staged_train_path, staged_test_path
            
        except Exception as e:
            logger.error("Data validation failed")
            raise ChurnPipelineException(e)

    def run_feature_engineering(
        self, 
        staged_train_path: Path, 
        staged_test_path: Path
    ) -> tuple[Path, Path]:
        logger.info("STAGE 3: FEATURE ENGINEERING")
        
        try:
            logger.info("Processing TRAIN dataset")
            train_engineer = FeatureEngineer(
                input_path=staged_train_path,
                output_filename="train_processed.csv",
                processed_dir=self.processed_dir,
                reports_dir=self.feature_reports_dir
            )
            processed_train_path = train_engineer.run()
            
            logger.info("Processing TEST dataset")

            test_engineer = FeatureEngineer(
                input_path=staged_test_path,
                output_filename="test_processed.csv"
            )
            processed_test_path = test_engineer.run()
            
            logger.info("Feature engineering completed for both datasets")
            return processed_train_path, processed_test_path
            
        except Exception as e:
            logger.error("Feature engineering failed")
            raise ChurnPipelineException(e)
        
    def run_data_transformation(self, processed_train_path: Path, processed_test_path: Path) -> dict:
            logger.info("STAGE 4: DATA TRANSFORMATION") 
            try:
                data_transformer = DataTransformation(
                    processed_train_path=processed_train_path,
                    processed_test_path=processed_test_path,
                    transformed_dir=self.transformed_data_dir,
                    preprocessing_dir=self.preprocessing_dir
                )
                transformation_artifacts = data_transformer.initiate_data_transformation()
                logger.info("Data transformation completed")
                return transformation_artifacts
            except Exception as e:
                logger.error("Data transformation failed")
                raise ChurnPipelineException(e)
            
    def run_model_tuning(self) -> dict:
        logger.info("STAGE 5: MODEL TUNING")
        try:
            model_tuner = ModelTuner()
            best_params = model_tuner.tune()
            logger.info("Model tuning completed")
            return best_params
        except Exception as e:
            logger.error("Model tuning failed")
            raise ChurnPipelineException(e)
            

    def run(self) -> None:
        try:
            
            logger.info("---------------TRAINING PIPELINE STARTED---------------")
            self.run_data_ingestion()

            staged_train_path, staged_test_path = self.run_data_validation()

            processed_train_path, processed_test_path = self.run_feature_engineering(
                staged_train_path, 
                staged_test_path
            )

            transformation_artifacts = self.run_data_transformation(
                processed_train_path, 
                processed_test_path
            )

            train_path = transformation_artifacts["train_path"]
            test_path = transformation_artifacts["test_path"]
            preprocessor_path = transformation_artifacts["preprocessor_path"]

            best_params = self.run_model_tuning()

            logger.info("---------------TRAINING PIPELINE COMPLETED SUCCESSFULLY---------------")

        except Exception as e:
            logger.error("---------------TRAINING PIPELINE FAILED---------------")
            raise ChurnPipelineException(e)


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run()