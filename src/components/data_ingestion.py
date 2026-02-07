import zipfile
from pathlib import Path
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

                # Extract and rename train file
                logger.info(f"Extracting and renaming {KAGGLE_TRAIN_FILENAME} to {TRAIN_FILENAME}")
                with zip_ref.open(KAGGLE_TRAIN_FILENAME) as source:
                    with open(self.output_dir / TRAIN_FILENAME, 'wb') as target:
                        target.write(source.read())
                
                # Extract and rename test file
                logger.info(f"Extracting and renaming {KAGGLE_TEST_FILENAME} to {TEST_FILENAME}")
                with zip_ref.open(KAGGLE_TEST_FILENAME) as source:
                    with open(self.output_dir / TEST_FILENAME, 'wb') as target:
                        target.write(source.read())

            zip_path.unlink()
            logger.info(f"Removed zip file: {zip_path}")
            logger.info("Data extraction completed successfully")
            
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