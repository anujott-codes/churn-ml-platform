from pathlib import Path
import json
import pandas as pd

from src.config.feature_config import (
    DROP_COLUMNS,
    TARGET_COLUMN,
    FEATURE_THRESHOLDS,
    FEATURES_TO_CREATE,
    PROCESSED_DATA_DIR,
    FEATURE_REPORTS_DIR
)
from src.exception import ChurnPipelineException
from src.logging.logger import logger


class FeatureEngineer:
    
    def __init__(self, input_path: Path, output_filename: str):
       
        self.input_path = input_path
        self.output_path = PROCESSED_DATA_DIR / output_filename
        self.reports_dir = FEATURE_REPORTS_DIR
        self.initial_shape = None
        self.final_shape = None
        self.features_created = []

    def _load_data(self) -> pd.DataFrame:
        """Load staged data from CSV file."""
        try:
            if not self.input_path.exists():
                raise FileNotFoundError(f"Input file not found: {self.input_path}")
            
            df = pd.read_csv(self.input_path)
            self.initial_shape = df.shape
            logger.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise ChurnPipelineException(e)

    def _rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
       
        logger.info("Standardizing column names")

        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
        )
        
        logger.info(f"Columns standardized: {len(df.columns)} columns")
        return df

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        
        logger.info("Cleaning data (removing nulls and duplicates)")

        initial_rows = len(df)
        
        # Count and drop null values
        null_counts = df.isnull().sum().sum()
        if null_counts > 0:
            logger.info(f"  Dropping {null_counts} null values")
            df = df.dropna()
        
        # Count and drop duplicates
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            logger.info(f"  Dropping {dup_count} duplicate rows")
            df = df.drop_duplicates()
        
        final_rows = len(df)
        rows_dropped = initial_rows - final_rows
        
        if rows_dropped > 0:
            logger.info(f"Data cleaned: {initial_rows} → {final_rows} rows ({rows_dropped} removed)")
        else:
            logger.info(f"Data cleaned: No rows removed")
        
        return df

    def _create_high_support_calls(self, df: pd.DataFrame) -> pd.DataFrame:
        
        threshold = FEATURE_THRESHOLDS["high_support_calls"]
        feature_name = "high_support_calls"
        
        df[feature_name] = (df["support_calls"] >= threshold).astype(int)
        
        count = df[feature_name].sum()
        pct = (count / len(df)) * 100
        logger.info(f"  Created '{feature_name}' (>= {threshold} calls): {count} customers ({pct:.1f}%)")
        
        self.features_created.append(feature_name)
        return df

    def _create_payment_delay_flag(self, df: pd.DataFrame) -> pd.DataFrame:
        
        threshold = FEATURE_THRESHOLDS["payment_delay_flag"]
        feature_name = "payment_delay_flag"
        
        df[feature_name] = (df["payment_delay"] > threshold).astype(int)
        
        count = df[feature_name].sum()
        pct = (count / len(df)) * 100
        logger.info(f"  Created '{feature_name}' (> {threshold} days): {count} customers ({pct:.1f}%)")
        
        self.features_created.append(feature_name)
        return df

    def _create_long_tenure_flag(self, df: pd.DataFrame) -> pd.DataFrame:
       
        threshold = FEATURE_THRESHOLDS["long_tenure_months"]
        feature_name = "long_tenure_flag"
        
        df[feature_name] = (df["tenure"] >= threshold).astype(int)
        
        count = df[feature_name].sum()
        pct = (count / len(df)) * 100
        logger.info(f"  Created '{feature_name}' (>= {threshold} months): {count} customers ({pct:.1f}%)")
        
        self.features_created.append(feature_name)
        return df

    def _create_high_usage_flag(self, df: pd.DataFrame) -> pd.DataFrame:
        
        threshold = FEATURE_THRESHOLDS["high_usage_frequency"]
        feature_name = "high_usage_flag"
        
        df[feature_name] = (df["usage_frequency"] >= threshold).astype(int)
        
        count = df[feature_name].sum()
        pct = (count / len(df)) * 100
        logger.info(f"  Created '{feature_name}' (>= {threshold}): {count} customers ({pct:.1f}%)")
        
        self.features_created.append(feature_name)
        return df

    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        
        features_added = []
        
        # Tenure × Usage interaction
        if "tenure_usage_interaction" in FEATURES_TO_CREATE:
            df["tenure_usage_interaction"] = df["tenure"] * df["usage_frequency"]
            features_added.append("tenure_usage_interaction")
            self.features_created.append("tenure_usage_interaction")
        
        # Spend per month (avoid division by zero)
        if "spend_per_month" in FEATURES_TO_CREATE:
            df["spend_per_month"] = df["total_spend"] / (df["tenure"] + 1)
            features_added.append("spend_per_month")
            self.features_created.append("spend_per_month")
        
        if features_added:
            logger.info(f"  Created interaction features: {', '.join(features_added)}")
        
        return df

    def _business_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        
        logger.info("Creating business-relevant churn features")
        
        # Create flag features
        if "high_support_calls" in FEATURES_TO_CREATE:
            df = self._create_high_support_calls(df)
        
        if "payment_delay_flag" in FEATURES_TO_CREATE:
            df = self._create_payment_delay_flag(df)
        
        if "long_tenure_flag" in FEATURES_TO_CREATE:
            df = self._create_long_tenure_flag(df)
        
        if "high_usage_flag" in FEATURES_TO_CREATE:
            df = self._create_high_usage_flag(df)
        
        # Create interaction/derived features
        if any(f in FEATURES_TO_CREATE for f in ["tenure_usage_interaction", "spend_per_month"]):
            df = self._create_interaction_features(df)
        
        logger.info(f"✓ Created {len(self.features_created)} new features")
        
        return df

    def _drop_useless_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Dropping non-predictive columns")
        
        cols_to_drop = [col for col in DROP_COLUMNS if col in df.columns]
        
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            logger.info(f"Dropped {len(cols_to_drop)} columns: {cols_to_drop}")
        else:
            logger.info("No columns to drop (already removed or not present)")
        
        return df

    def _generate_feature_report(self, df: pd.DataFrame) -> dict:
        
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        self.final_shape = df.shape
        
        # Calculate changes
        rows_removed = self.initial_shape[0] - self.final_shape[0]
        cols_added = len(self.features_created)
        cols_removed = len(DROP_COLUMNS)
        
        # Build report
        report = {
            "input_file": str(self.input_path.name),
            "output_file": str(self.output_path.name),
            "timestamp": pd.Timestamp.now().isoformat(),
            "data_shape_change": {
                "initial": {
                    "rows": self.initial_shape[0],
                    "columns": self.initial_shape[1]
                },
                "final": {
                    "rows": self.final_shape[0],
                    "columns": self.final_shape[1]
                },
                "rows_removed": rows_removed,
                "columns_added": cols_added,
                "columns_removed": cols_removed
            },
            "features_created": self.features_created,
            "features_dropped": DROP_COLUMNS,
            "feature_thresholds_used": FEATURE_THRESHOLDS,
            "final_columns": df.columns.tolist(),
            "target_column": TARGET_COLUMN,
            "target_in_data": TARGET_COLUMN in df.columns,
            "summary": {
                "total_features": len(df.columns),
                "new_features_count": len(self.features_created),
                "original_features_retained": self.initial_shape[1] - len(DROP_COLUMNS)
            }
        }
        
        # Save report as JSON
        report_filename = f"{self.input_path.stem}_feature_engineering_report.json"
        report_path = self.reports_dir / report_filename
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Feature engineering report saved: {report_path}")
        return report

    def _save_data(self, df: pd.DataFrame) -> None:
        """Save processed data to CSV file."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.output_path, index=False)
        logger.info(f"Processed data saved: {self.output_path}")

    def run(self) -> Path:
        
        try:
        
            logger.info(f"Starting feature engineering: {self.input_path.name}")
           

            # Execute pipeline steps
            df = self._load_data()
            df = self._rename_columns(df)
            df = self._clean_data(df)
            df = self._business_feature_engineering(df)
            df = self._drop_useless_columns(df)
            
            # Generate audit trail
            self._generate_feature_report(df)
            
            # Save final output
            self._save_data(df)

            
            logger.info("Feature engineering completed successfully")
            logger.info(f"Final shape: {self.final_shape[0]} rows × {self.final_shape[1]} columns")
            logger.info(f"Features created: {len(self.features_created)}")
            logger.info(f"Output: {self.output_path}")
            
            
            return self.output_path

        except Exception as e:
            
            logger.error(f"Feature engineering failed: {str(e)}")
            raise ChurnPipelineException(e)
