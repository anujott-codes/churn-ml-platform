from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RAW_DATA_DIR = DATA_DIR / "raw"
STAGED_DATA_DIR = DATA_DIR / "staged"

PROCESSED_TRAIN_FILENAME = "processed_train_data.csv"
PROCESSED_TEST_FILENAME = "processed_test_data.csv"