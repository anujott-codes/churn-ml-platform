import json
import pickle
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)

from src.config.feature_config import TARGET_COLUMN
from src.config.basic_config import PROCESSED_DATA_DIR, MODEL_DIR, EVALUATION_REPORTS_DIR, METRICS_DIR
from src.config.trainer_config import MODEL_FILENAME
from src.config.evaluation_config import (
    TEST_DATA_FILENAME,
    EVALUATION_REPORT_FILENAME,
    CONFUSION_MATRIX_FILENAME,
    ROC_CURVE_FILENAME,
    PR_CURVE_FILENAME,
    DEFAULT_THRESHOLD,
    TOP_K_PERCENT,
)

from src.exception import ChurnPipelineException
from src.logging.logger import logger


class ModelEvaluator:
    def __init__(
        self,
        model_path: Path = MODEL_DIR / MODEL_FILENAME,
        test_data_path: Path = PROCESSED_DATA_DIR / TEST_DATA_FILENAME,
        reports_dir: Path = EVALUATION_REPORTS_DIR,
        threshold: float = DEFAULT_THRESHOLD,
        metrics_dir: Path = METRICS_DIR,
    ):
        self.model_path = model_path
        self.test_data_path = test_data_path
        self.reports_dir = reports_dir
        self.threshold = threshold
        self.metrics_dir = metrics_dir

        if not 0 <= self.threshold <= 1:
            raise ChurnPipelineException("Threshold must be between 0 and 1.")

        self.report_path = self.reports_dir / EVALUATION_REPORT_FILENAME
        self.conf_matrix_path = self.metrics_dir / CONFUSION_MATRIX_FILENAME
        self.roc_curve_path = self.metrics_dir / ROC_CURVE_FILENAME
        self.pr_curve_path = self.metrics_dir / PR_CURVE_FILENAME

        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

    def load_pipeline(self):
        try:
            with open(self.model_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            raise ChurnPipelineException(f"Error loading pipeline: {e}")

    def load_test_data(self):
        try:
            data = pd.read_csv(self.test_data_path)

            if TARGET_COLUMN not in data.columns:
                raise ChurnPipelineException(
                    f"{TARGET_COLUMN} not found in test dataset."
                )

            X = data.drop(TARGET_COLUMN, axis=1)
            y = data[TARGET_COLUMN]

            return X, y

        except Exception as e:
            raise ChurnPipelineException(f"Error loading test data: {e}")

    def compute_precision_at_k(self, y_true, y_proba):
        k = max(1, int(len(y_proba) * TOP_K_PERCENT))
        indices = np.argsort(y_proba)[::-1][:k]
        return y_true.iloc[indices].mean()

    def evaluate(self):
        try:
            logger.info("Starting model evaluation...")

            pipeline = self.load_pipeline()
            X_test, y_test = self.load_test_data()

            y_proba = pipeline.predict_proba(X_test)[:, 1]
            y_pred = (y_proba >= self.threshold).astype(int)

            # Core metrics
            roc_auc = roc_auc_score(y_test, y_proba)
            pr_auc = average_precision_score(y_test, y_proba)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            accuracy = accuracy_score(y_test, y_pred)
            precision_at_k = self.compute_precision_at_k(y_test, y_proba)

            cm = confusion_matrix(y_test, y_pred)

            self._save_confusion_matrix(cm)
            self._save_roc_curve(y_test, y_proba)
            self._save_pr_curve(y_test, y_proba)

            report = {
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "accuracy": accuracy,
                "precision_at_k": precision_at_k,
                "threshold": self.threshold,
                "test_rows": len(X_test),
                "class_distribution": {
                    "positive": int(y_test.sum()),
                    "negative": int(len(y_test) - y_test.sum()),
                },
                "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
            }

            with open(self.report_path, "w") as f:
                json.dump(report, f, indent=4)

            logger.info("Model evaluation completed successfully.")
            return report, cm

        except Exception as e:
            logger.error(f"Error during model evaluation: {e}")
            raise ChurnPipelineException(e)

    def _save_confusion_matrix(self, cm):
        plt.figure()
        plt.imshow(cm, interpolation="nearest")
        plt.title("Confusion Matrix")
        plt.colorbar()
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig(self.conf_matrix_path)
        plt.close()

    def _save_roc_curve(self, y_test, y_proba):
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure()
        plt.plot(fpr, tpr)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.savefig(self.roc_curve_path)
        plt.close()

    def _save_pr_curve(self, y_test, y_proba):
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        plt.figure()
        plt.plot(recall, precision)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.savefig(self.pr_curve_path)
        plt.close()
