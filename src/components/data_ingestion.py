import zipfile
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

from src.config.data_source_config import (
    RAW_DATA_DIR,
    KAGGLE_DATASET,
    KAGGLE_TRAIN_FILENAME,  
    KAGGLE_TEST_FILENAME,   
    TRAIN_FILENAME,         
    TEST_FILENAME,          
)

def download_and_extract_kaggle_dataset(dataset: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Authenticate Kaggle API
    api = KaggleApi()
    api.authenticate()

    api.dataset_download_files(dataset=dataset, path=output_dir, unzip=False)

    # Find the downloaded zip 
    zip_files = list(output_dir.glob("*.zip"))
    if not zip_files:
        raise FileNotFoundError("Kaggle dataset zip not found")
    
    zip_path = zip_files[0]

    # Open and extract with renaming
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        members = zip_ref.namelist()

        if KAGGLE_TRAIN_FILENAME not in members or KAGGLE_TEST_FILENAME not in members:
            raise ValueError("Expected train/test CSVs not found in Kaggle dataset")

        # Extract and rename train file
        with zip_ref.open(KAGGLE_TRAIN_FILENAME) as source:
            with open(output_dir / TRAIN_FILENAME, 'wb') as target:
                target.write(source.read())
        
        # Extract and rename test file
        with zip_ref.open(KAGGLE_TEST_FILENAME) as source:
            with open(output_dir / TEST_FILENAME, 'wb') as target:
                target.write(source.read())

    zip_path.unlink()


def extract_raw_data() -> None:
    download_and_extract_kaggle_dataset(KAGGLE_DATASET, RAW_DATA_DIR)