from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
STAGED_DATA_DIR = DATA_DIR / "staged"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Artifact directories
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_DIR = ARTIFACTS_DIR / "model"
METRICS_DIR = ARTIFACTS_DIR / "metrics"

# Report directories
REPORTS_DIR = PROJECT_ROOT / "reports"
VALIDATION_REPORTS_DIR = REPORTS_DIR / "validation" 
EVALUATION_REPORTS_DIR = REPORTS_DIR / "evaluation"
FEATURE_REPORTS_DIR = REPORTS_DIR / "feature_engineering"
MODEL_REPORTS_DIR = REPORTS_DIR / "model"
EVALUATION_REPORTS_DIR = REPORTS_DIR / "evaluation"