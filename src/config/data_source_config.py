from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Kaggle dataset identifier
KAGGLE_DATASET = "muhammadshahidazeem/customer-churn-dataset"

# Original Kaggle filenames 
KAGGLE_TRAIN_FILENAME = "customer_churn_dataset-training-master.csv"
KAGGLE_TEST_FILENAME = "customer_churn_dataset-testing-master.csv"

# Standardized filenames 
TRAIN_FILENAME = "train.csv"
TEST_FILENAME = "test.csv"