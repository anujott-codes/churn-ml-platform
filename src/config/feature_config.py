from pathlib import Path

TARGET_COLUMN = "churn"

DROP_COLUMNS = [
    "customerid"  
]

FEATURE_THRESHOLDS = {
    "high_support_calls": 4,        
    "payment_delay_flag": 0,                               
}


FEATURES_TO_CREATE = [
    "high_support_calls",           
    "payment_delay_flag",                        
    "spend_per_month",              
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

# Data directories
RAW_DATA_DIR = DATA_DIR / "raw"
STAGED_DATA_DIR = DATA_DIR / "staged"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Reports directory
REPORTS_DIR = PROJECT_ROOT / "reports"
FEATURE_REPORTS_DIR = REPORTS_DIR / "feature_engineering"