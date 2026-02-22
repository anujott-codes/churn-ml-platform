
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

INTEGER_FEATURES = [
    "age",
    "tenure",
    "usage_frequency",
    "support_calls",
    "payment_delay",
    "last_interaction"
]