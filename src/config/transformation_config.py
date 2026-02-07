from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRANSFORMED_DATA_DIR = DATA_DIR / "transformed"


ARTIFACTS_DIR = BASE_DIR / "artifacts"
PREPROCESSING_DIR = ARTIFACTS_DIR / "preprocessing"

TRAIN_FILENAME = "train_processed.csv"
TEST_FILENAME = "test_processed.csv"

TRANSFORMED_TRAIN_FILENAME = "transformed_train.csv"
TRANSFORMED_TEST_FILENAME = "transformed_test.csv"

# Target Column
TARGET_COLUMN = "churn"   

# Numerical Features
NUMERICAL_FEATURES = ['age', 'tenure', 'usage_frequency', 'support_calls', 'payment_delay', 'total_spend', 'last_interaction', 'high_support_calls', 'payment_delay_flag', 'spend_per_month']

# Nominal Categorical Features
NOMINAL_CATEGORICAL_FEATURES = ['gender', 'subscription_type']

# Ordinal Categorical Features
ORDINAL_CATEGORICAL_FEATURES = [
    "contract_length",
]

ORDINAL_CATEGORIES = ["Monthly","Quaterly","Yearly"]
