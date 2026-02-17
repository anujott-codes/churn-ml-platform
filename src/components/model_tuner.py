import json
from pathlib import Path

import optuna
import pandas as pd
import numpy as np
import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from src.components.data_transformation import DataTransformation
from src.config.feature_config import TARGET_COLUMN
from src.config.basic_config import PROCESSED_DATA_DIR, MODEL_REPORTS_DIR
from src.config.tuner_config import (
    TRAIN_DATA_FILENAME,
    N_TRIALS,
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
        train_data_path: Path = PROCESSED_DATA_DIR / TRAIN_DATA_FILENAME,
        report_dir: Path = MODEL_REPORTS_DIR,
        n_trials: int = N_TRIALS,
        random_state: int = RANDOM_STATE
    ):
        self.train_data_path = train_data_path
        self.report_dir = report_dir
        self.n_trials = n_trials
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

            return X, y

        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            raise ChurnPipelineException(e)


    def objective(self, trial, X, y):

        max_depth = trial.suggest_int(
            "max_depth",
            self.params_dict["max_depth"][0],
            self.params_dict["max_depth"][1]
        )

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

        skf = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=self.random_state
        )

        auc_scores = []

        for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(X, y)):

            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

            pos = (y_train == 1).sum()
            neg = (y_train == 0).sum()
            fold_params = params.copy()
            fold_params["scale_pos_weight"] = neg / max(pos, 1)

            preprocessor = DataTransformation().get_preprocessor()
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_valid_transformed = preprocessor.transform(X_valid)

            model = lgb.LGBMClassifier(**fold_params)

            model.fit(
                X_train_transformed,
                y_train,
                eval_set=[(X_valid_transformed, y_valid)],
                eval_metric="auc",
                callbacks=[
                    lgb.early_stopping(self.early_stopping_rounds, verbose=False),
                    lgb.log_evaluation(0)
                ]
            )

            preds = model.predict_proba(X_valid_transformed)[:, 1]
            auc = roc_auc_score(y_valid, preds)
            auc_scores.append(auc)

            trial.report(np.mean(auc_scores), step=fold_idx)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        mean_auc = np.mean(auc_scores)
        logger.info(
            f"Trial {trial.number} — mean AUC: {mean_auc:.4f}, std: {np.std(auc_scores):.4f}"
        )

        return mean_auc



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

        X, y = self.load_data()

        sampler = optuna.samplers.TPESampler(
            seed=self.random_state
        )

        study = optuna.create_study(
            direction="maximize",
            sampler=sampler
        )

        study.optimize(
            lambda trial: self.objective(trial, X, y),
            n_trials=self.n_trials
        )

        best_trial = study.best_trial
        best_params = best_trial.params

        best_params.update({
            "objective": "binary",
            "boosting_type": "gbdt",
            "random_state": self.random_state
        })

        logger.info(f"Best AUC: {study.best_value}")
        logger.info(f"Best Parameters: {best_params}")

        self.save_reports(study, best_params)

        return best_params
