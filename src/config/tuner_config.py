TRAIN_DATA_FILENAME = "train_processed.csv"

N_TRIALS = 50
TEST_SIZE = 0.2
RANDOM_STATE = 42

BEST_PARAMS_FILENAME = "best_params.json"
TUNING_TRIALS_FILENAME = "tuning_trials.csv"
TUNING_SUMMARY_FILENAME = "tuning_summary.json"

PARAMS = {
    "num_leaves": (4, 150),
    "max_depth": (3, 15),
    "learning_rate": (0.01, 0.3),
    "n_estimators": (200, 1000),
    "min_child_samples": (5, 100),
    "subsample": (0.6, 1.0),
    "colsample_bytree": (0.6, 1.0),
    "reg_alpha": (0.0, 10.0),
    "reg_lambda": (0.0, 10.0),
}

EARLY_STOPPING_ROUNDS = 50