import json
from pathlib import Path

import optuna
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from src.config.feature_config import TARGET_COLUMN
from src.config.basic_config import TRANSFORMED_DATA_DIR, MODEL_REPORTS_DIR
from src.config.tuner_config import (
    TRAIN_DATA_FILENAME,
    N_TRIALS,
    TEST_SIZE,
    RANDOM_STATE,
    BEST_PARAMS_FILENAME,
    TUNING_TRIALS_FILENAME,
    TUNING_SUMMARY_FILENAME,
    PARAMS,
    EARLY_STOPPING_ROUNDS
)
from src.logging.logger import logger
from src.exception import ChurnPipelineException


class ModelTuner:

    def __init__(
        self,
        train_data_path: Path = TRANSFORMED_DATA_DIR / TRAIN_DATA_FILENAME,
        report_dir: Path = MODEL_REPORTS_DIR,
        n_trials: int = N_TRIALS,
        test_size: float = TEST_SIZE,
        random_state: int = RANDOM_STATE
    ):
        self.train_data_path = train_data_path
        self.report_dir = report_dir
        self.n_trials = n_trials
        self.test_size = test_size
        self.random_state = random_state

        self.best_params_path = self.report_dir / BEST_PARAMS_FILENAME
        self.trials_path = self.report_dir / TUNING_TRIALS_FILENAME
        self.summary_path = self.report_dir / TUNING_SUMMARY_FILENAME

        self.params_dict = PARAMS
        self.early_stopping_rounds = EARLY_STOPPING_ROUNDS

    def load_data(self):
        try:
            df = pd.read_csv(self.train_data_path)

            if TARGET_COLUMN not in df.columns:
                raise ChurnPipelineException(
                    f"{TARGET_COLUMN} not found in dataset."
                )

            X = df.drop(columns=[TARGET_COLUMN])
            y = df[TARGET_COLUMN]

            X_train, X_valid, y_train, y_valid = train_test_split(
                X,
                y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y
            )

            return X_train, X_valid, y_train, y_valid

        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            raise ChurnPipelineException(e)

    def objective(self, trial, X_train, y_train, X_valid, y_valid):

        # Handle imbalance
        pos = y_train.sum()
        neg = len(y_train) - pos
        scale_pos_weight = neg / pos if pos > 0 else 1.0

        # Suggest depth first
        max_depth = trial.suggest_int(
            "max_depth",
            self.params_dict["max_depth"][0],
            self.params_dict["max_depth"][1]
        )

        # Enforce valid num_leaves constraint
        max_leaves_limit = 2 ** max_depth
        num_leaves = trial.suggest_int(
            "num_leaves",
            self.params_dict["num_leaves"][0],
            min(self.params_dict["num_leaves"][1], max_leaves_limit)
        )

        params = {
            "objective": "binary",
            "boosting_type": "gbdt",
            "verbosity": -1,
            "random_state": self.random_state,
            "scale_pos_weight": scale_pos_weight,

            "max_depth": max_depth,
            "num_leaves": num_leaves,

            "learning_rate": trial.suggest_float(
                "learning_rate",
                self.params_dict["learning_rate"][0],
                self.params_dict["learning_rate"][1],
                log=True
            ),

            "n_estimators": trial.suggest_int(
                "n_estimators",
                self.params_dict["n_estimators"][0],
                self.params_dict["n_estimators"][1]
            ),

            "min_child_samples": trial.suggest_int(
                "min_child_samples",
                self.params_dict["min_child_samples"][0],
                self.params_dict["min_child_samples"][1]
            ),

            "subsample": trial.suggest_float(
                "subsample",
                self.params_dict["subsample"][0],
                self.params_dict["subsample"][1]
            ),

            "colsample_bytree": trial.suggest_float(
                "colsample_bytree",
                self.params_dict["colsample_bytree"][0],
                self.params_dict["colsample_bytree"][1]
            ),

            "reg_alpha": trial.suggest_float(
                "reg_alpha",
                self.params_dict["reg_alpha"][0],
                self.params_dict["reg_alpha"][1]
            ),

            "reg_lambda": trial.suggest_float(
                "reg_lambda",
                self.params_dict["reg_lambda"][0],
                self.params_dict["reg_lambda"][1]
            ),
        }

        model = lgb.LGBMClassifier(**params)

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="auc",
            callbacks=[
                lgb.early_stopping(self.early_stopping_rounds),
                lgb.log_evaluation(0)
            ]
        )

        preds = model.predict_proba(X_valid)[:, 1]
        auc = roc_auc_score(y_valid, preds)

        
        trial.set_user_attr("best_iteration", model.best_iteration_)
        return auc

    def save_reports(self, study, best_params):

        self.report_dir.mkdir(parents=True, exist_ok=True)

        with open(self.best_params_path, "w") as f:
            json.dump(best_params, f, indent=4)

        trials_df = study.trials_dataframe()
        trials_df.to_csv(self.trials_path, index=False)

        summary = {
            "best_auc": study.best_value,
            "n_trials": len(study.trials),
        }

        with open(self.summary_path, "w") as f:
            json.dump(summary, f, indent=4)

    def tune(self):

        X_train, X_valid, y_train, y_valid = self.load_data()

        sampler = optuna.samplers.TPESampler(
            seed=self.random_state
        )

        study = optuna.create_study(
            direction="maximize",
            sampler=sampler
        )

        study.optimize(
            lambda trial: self.objective(
                trial,
                X_train,
                y_train,
                X_valid,
                y_valid
            ),
            n_trials=self.n_trials
        )

        best_trial = study.best_trial
        best_params = best_trial.params

        # Inject optimal tree count
        best_params["n_estimators"] = best_trial.user_attrs["best_iteration"]

        # Add fixed params
        best_params.update({
            "objective": "binary",
            "boosting_type": "gbdt",
            "random_state": self.random_state
        })

        logger.info(f"Best AUC: {study.best_value}")
        logger.info(f"Best Parameters: {best_params}")

        self.save_reports(study, best_params)
        return best_params
