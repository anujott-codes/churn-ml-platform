import zipfile
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from kaggle.api.kaggle_api_extended import KaggleApi

from src.config.basic_config import RAW_DATA_DIR

from src.config.data_source_config import (
    KAGGLE_DATASET,
    KAGGLE_TRAIN_FILENAME,  
    KAGGLE_TEST_FILENAME,   
    TRAIN_FILENAME,         
    TEST_FILENAME,     
)

from src.exception import ChurnPipelineException
from src.logging.logger import logger


class DataIngestion:
    def __init__(self, dataset: str = KAGGLE_DATASET, output_dir: Path = RAW_DATA_DIR):
        self.dataset = dataset
        self.output_dir = output_dir
    
    def download_and_extract_kaggle_dataset(self) -> None:
        try:
            logger.info("Starting Kaggle dataset download and extraction")
            
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Output directory created/verified: {self.output_dir}")

            # Authenticate Kaggle API
            api = KaggleApi()
            api.authenticate()
            logger.info("Kaggle API authenticated successfully")

            logger.info(f"Downloading dataset: {self.dataset}")
            api.dataset_download_files(dataset=self.dataset, path=self.output_dir, unzip=False)
            logger.info("Dataset downloaded successfully")

            # Find the downloaded zip 
            zip_files = list(self.output_dir.glob("*.zip"))
            if not zip_files:
                raise FileNotFoundError("Kaggle dataset zip not found")
            
            zip_path = zip_files[0]
            logger.info(f"Found zip file: {zip_path}")

            # Open and extract with renaming
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                members = zip_ref.namelist()

                if KAGGLE_TRAIN_FILENAME not in members or KAGGLE_TEST_FILENAME not in members:
                    raise ValueError("Expected train/test CSVs not found in Kaggle dataset")

                # Extract train file
                logger.info(f"Extracting {KAGGLE_TRAIN_FILENAME}")
                with zip_ref.open(KAGGLE_TRAIN_FILENAME) as source:
                    with open(self.output_dir / KAGGLE_TRAIN_FILENAME, 'wb') as target:
                        target.write(source.read())
                
                # Extract test file
                logger.info(f"Extracting {KAGGLE_TEST_FILENAME}")
                with zip_ref.open(KAGGLE_TEST_FILENAME) as source:
                    with open(self.output_dir / KAGGLE_TEST_FILENAME, 'wb') as target:
                        target.write(source.read())

            # Merge and re-split the data
            logger.info("Merging train and test datasets")
            train_df = pd.read_csv(self.output_dir / KAGGLE_TRAIN_FILENAME)
            test_df = pd.read_csv(self.output_dir / KAGGLE_TEST_FILENAME)
            merged_df = pd.concat([train_df, test_df], ignore_index=True)
            logger.info(f"Merged dataset shape: {merged_df.shape}")

            # Perform stratified train-test split (80/20)
            logger.info("Performing stratified train-test split (80/20)")
            
            train_data, test_data = train_test_split(
                merged_df,
                test_size=0.2,
                random_state=42,
            )
            logger.info(f"Train set shape: {train_data.shape}, Test set shape: {test_data.shape}")

            # Save the new train and test files
            logger.info(f"Saving new train file as {TRAIN_FILENAME}")
            train_data.to_csv(self.output_dir / TRAIN_FILENAME, index=False)
            
            logger.info(f"Saving new test file as {TEST_FILENAME}")
            test_data.to_csv(self.output_dir / TEST_FILENAME, index=False)

            # Clean up: remove original extracted files and zip
            (self.output_dir / KAGGLE_TRAIN_FILENAME).unlink()
            (self.output_dir / KAGGLE_TEST_FILENAME).unlink()
            zip_path.unlink()
            logger.info(f"Removed temporary files")
            logger.info("Data extraction and re-splitting completed successfully")
            
        except Exception as e:
            logger.error("Error occurred during data ingestion")
            raise ChurnPipelineException(e)

    def initiate_data_ingestion(self) -> None:
        try:
            logger.info("Data ingestion initiated")
            self.download_and_extract_kaggle_dataset()
            logger.info("Data ingestion completed successfully")
            
        except Exception as e:
            logger.error("Error occurred during data ingestion initiation")
            raise ChurnPipelineException(e)