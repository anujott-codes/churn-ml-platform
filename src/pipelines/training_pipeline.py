from pathlib import Path

from src.components.data_ingestion import extract_raw_data
from src.components.data_validation import RawDataValidator
from src.components.feature_engineering import FeatureEngineer

from src.config.data_source_config import (
    RAW_DATA_DIR,
    TRAIN_FILENAME,
    TEST_FILENAME
)
from src.config.feature_config import (
    STAGED_DATA_DIR,
    PROCESSED_DATA_DIR
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

    def run_data_ingestion(self) -> None:
        
        logger.info("STAGE 1: DATA INGESTION")
        
        try:
            extract_raw_data()
            logger.info("Data ingestion completed")
        except Exception as e:
            logger.error("Data ingestion failed")
            raise ChurnPipelineException(e)

    def run_data_validation(self) -> tuple[Path, Path]:
        logger.info("STAGE 2: DATA VALIDATION")
        
        try:
            logger.info("Validating TRAIN dataset")
            train_validator = RawDataValidator(
                raw_data_path=self.raw_train_path,
                staged_dir=self.staged_dir
            )
            staged_train_path = train_validator.validate()
            
            logger.info("Validating TEST dataset")
            test_validator = RawDataValidator(
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
                output_filename="train_processed.csv"
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

    def run(self) -> None:
        try:
            
            logger.info("---------------TRAINING PIPELINE STARTED---------------")
            self.run_data_ingestion()

            staged_train_path, staged_test_path = self.run_data_validation()

            processed_train_path, processed_test_path = self.run_feature_engineering(
                staged_train_path, 
                staged_test_path
            )

            logger.info("---------------TRAINING PIPELINE COMPLETED SUCCESSFULLY---------------")

        except Exception as e:
            logger.error("---------------TRAINING PIPELINE FAILED---------------")
            raise ChurnPipelineException(e)


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run()