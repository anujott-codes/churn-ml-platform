MODEL_NAME = "CustomerChurnModel"
MODEL_ALIAS = "champion" 

RAW_FEATURES = [ "age", "gender","tenure", "usage_frequency", "support_calls", "payment_delay", "subscription_type", "contract_length", "total_spend", "last_interaction" ]

FEATURE_LOGIC = {
    "high_support_calls": lambda df: (df["support_calls"] > 4).astype(int),
    "payment_delay_flag": lambda df: (df["payment_delay"] > 0).astype(int),
    "spend_per_month": lambda df: df["total_spend"] / (df["tenure"].replace({0:1}))
}

ALL_FEATURES = [ 
    "age",
    "gender",
    "tenure",
    "usage_frequency",
    "support_calls",
    "payment_delay",
    "subscription_type",
    "contract_length",
    "total_spend",
    "last_interaction",
    "high_support_calls",
    "payment_delay_flag",
    "spend_per_month"
]