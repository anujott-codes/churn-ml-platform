from pathlib import Path

# Schema definition
RAW_DATA_SCHEMA = {
    "CustomerID": "float64",
    "Age": "float64",
    "Gender": "object",
    "Tenure": "float64",
    "Usage Frequency": "float64",
    "Support Calls": "float64",
    "Payment Delay": "float64",
    "Subscription Type": "object",
    "Contract Length": "object",
    "Total Spend": "float64",
    "Last Interaction": "float64",
    "Churn": "float64"
}

TARGET_COLUMN = "Churn"

# Validation thresholds
VALIDATION_CONFIG = {
    "min_rows": 100,                    
    "max_missing_percentage": 50,       
    "min_minority_class_ratio": 0.05,   
}


 