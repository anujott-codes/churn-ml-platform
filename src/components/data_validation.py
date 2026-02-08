from pathlib import Path
from typing import Dict
from shutil import copy2
import json

import pandas as pd

from src.config.basic_config import (
    STAGED_DATA_DIR,
    VALIDATION_REPORTS_DIR
)

from src.config.schema import (
    RAW_DATA_SCHEMA,
    TARGET_COLUMN,
    VALIDATION_CONFIG,
)
from src.exception import ChurnPipelineException
from src.logging.logger import logger


class DataValidator:    
    def __init__(
        self, 
        raw_data_path: Path, 
        staged_dir: Path = STAGED_DATA_DIR,
        reports_dir: Path = VALIDATION_REPORTS_DIR,
        min_rows: int = None,
        max_missing_pct: float = None,
        min_minority_ratio: float = None
    ):
        self.raw_data_path = raw_data_path
        self.staged_dir = staged_dir
        self.schema: Dict[str, str] = RAW_DATA_SCHEMA
        self.reports_dir = reports_dir
        
        # Use config defaults if not provided
        self.min_rows = min_rows or VALIDATION_CONFIG["min_rows"]
        self.max_missing_pct = max_missing_pct or VALIDATION_CONFIG["max_missing_percentage"]
        self.min_minority_ratio = min_minority_ratio or VALIDATION_CONFIG["min_minority_class_ratio"]

    def _validate_file_exists(self) -> None:
        """Check if raw data file exists."""
        if not self.raw_data_path.exists():
            raise FileNotFoundError(
                f"Raw data file not found: {self.raw_data_path}"
            )
        logger.info(f"File exists: {self.raw_data_path.name}")

    def _load_data(self) -> pd.DataFrame:
        """Load data from CSV file."""
        try:
            df = pd.read_csv(self.raw_data_path)
            logger.info(f"Data loaded: {len(df)} rows, {len(df.columns)} columns")
            return df
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise ChurnPipelineException(e)

    def _validate_empty(self, df: pd.DataFrame) -> None:
        """Check if dataset is empty."""
        if df.empty:
            raise ValueError("Raw dataset is empty")
        logger.info("Dataset is not empty")

    def _validate_columns(self, df: pd.DataFrame) -> None:
        """Validate presence of required columns."""
        expected_cols = set(self.schema.keys())
        actual_cols = set(df.columns)

        missing_cols = expected_cols - actual_cols
        extra_cols = actual_cols - expected_cols

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        if extra_cols:
            logger.warning(f"Extra columns detected: {extra_cols}")
        
        logger.info(f"All {len(expected_cols)} required columns present")

    def _validate_dtypes(self, df: pd.DataFrame) -> None:
        """Validate column data types match schema."""
        mismatches = []
        
        for col, expected_dtype in self.schema.items():
            if col not in df.columns:
                continue
                
            actual_dtype = str(df[col].dtype)
            
            # Handle pandas dtype variations
            dtype_match = False
            if expected_dtype == "object" and actual_dtype == "object":
                dtype_match = True
            elif expected_dtype == "float64" and actual_dtype in ["float64", "int64", "float32", "int32"]:
                dtype_match = True
            elif expected_dtype in actual_dtype:
                dtype_match = True
            
            if not dtype_match:
                mismatches.append(f"{col}: expected {expected_dtype}, got {actual_dtype}")
        
        if mismatches:
            logger.warning(f"Data type mismatches detected:\n  " + "\n  ".join(mismatches))
        else:
            logger.info("All column data types match schema")

    def _validate_duplicates(self, df: pd.DataFrame) -> None:
        """Check for duplicate columns and rows."""
        # Check duplicate column names
        if df.columns.duplicated().any():
            dup_cols = df.columns[df.columns.duplicated()].tolist()
            raise ValueError(f"Duplicate column names detected: {dup_cols}")
        
        # Check duplicate rows
        dup_rows = df.duplicated().sum()
        if dup_rows > 0:
            dup_pct = (dup_rows / len(df)) * 100
            logger.warning(f"Found {dup_rows} duplicate rows ({dup_pct:.2f}%)")
        else:
            logger.info("No duplicate rows found")

    def _validate_missing_values(self, df: pd.DataFrame) -> None:
        """Check and report missing values."""
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        
        # Check for columns with high missing percentage
        high_missing = missing_pct[missing_pct > self.max_missing_pct]
        
        if not high_missing.empty:
            logger.warning(
                f"Columns with >{self.max_missing_pct}% missing values:\n" +
                "\n".join([f"  {col}: {pct:.2f}%" for col, pct in high_missing.items()])
            )
        
        # Log all columns with missing values
        cols_with_missing = missing[missing > 0]
        if len(cols_with_missing) > 0:
            logger.info(
                f"Missing values summary ({len(cols_with_missing)} columns):\n" +
                "\n".join([f"  {col}: {count} ({(count/len(df)*100):.2f}%)" 
                          for col, count in cols_with_missing.items()])
            )
        else:
            logger.info("No missing values detected")

    def _validate_target(self, df: pd.DataFrame) -> None:
        """Validate target column."""
        if TARGET_COLUMN not in df.columns:
            raise ValueError(f"Target column '{TARGET_COLUMN}' not found")
        
        # Check for missing values in target
        target_nulls = df[TARGET_COLUMN].isnull().sum()
        if target_nulls > 0:
            null_pct = (target_nulls / len(df)) * 100
            logger.warning(
                f"Target column has {target_nulls} missing values ({null_pct:.2f}%)"
            )
        
        # Validate binary values
        unique_vals = df[TARGET_COLUMN].dropna().unique()
        valid_values = {0, 1, 0.0, 1.0}
        if not set(unique_vals).issubset(valid_values):
            raise ValueError(
                f"Target must be binary (0/1), found: {sorted(unique_vals)}"
            )
        
        logger.info("Target column is binary (0/1)")
        
        # Check class balance
        class_counts = df[TARGET_COLUMN].value_counts()
        class_dist = df[TARGET_COLUMN].value_counts(normalize=True)
        
        logger.info(
            f"Target distribution:\n" +
            "\n".join([f"  Class {int(cls)}: {count} ({dist*100:.2f}%)" 
                      for cls, (count, dist) in enumerate(zip(class_counts, class_dist))])
        )
        
        # Check for severe imbalance
        minority_ratio = class_dist.min()
        if minority_ratio < self.min_minority_ratio:
            logger.warning(
                f"Severe class imbalance detected: "
                f"minority class is {minority_ratio*100:.2f}% "
                f"(threshold: {self.min_minority_ratio*100}%)"
            )
        else:
            logger.info(f"Class balance acceptable (minority: {minority_ratio*100:.2f}%)")

    def _generate_validation_report(self, df: pd.DataFrame) -> Dict:
        """Generate and save validation report."""
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate statistics
        missing_summary = df.isnull().sum().to_dict()
        target_dist = df[TARGET_COLUMN].value_counts().to_dict()
        
        report = {
            "data_file": str(self.raw_data_path.name),
            "validation_timestamp": pd.Timestamp.now().isoformat(),
            "data_shape": {
                "n_rows": len(df),
                "n_columns": len(df.columns)
            },
            "schema_validation": {
                "expected_columns": len(self.schema),
                "actual_columns": len(df.columns),
                "all_required_present": set(self.schema.keys()).issubset(set(df.columns))
            },
            "data_quality": {
                "duplicate_rows": int(df.duplicated().sum()),
                "duplicate_row_percentage": float((df.duplicated().sum() / len(df)) * 100),
                "total_missing_values": int(df.isnull().sum().sum()),
                "columns_with_missing": {k: int(v) for k, v in missing_summary.items() if v > 0}
            },
            "target_analysis": {
                "target_column": TARGET_COLUMN,
                "target_missing": int(df[TARGET_COLUMN].isnull().sum()),
                "class_distribution": {str(int(k)): int(v) for k, v in target_dist.items()},
                "class_balance_ratio": float(df[TARGET_COLUMN].value_counts(normalize=True).min())
            },
            "validation_thresholds": {
                "min_rows": self.min_rows,
                "max_missing_percentage": self.max_missing_pct,
                "min_minority_class_ratio": self.min_minority_ratio
            },
            "validation_status": "PASSED"
        }
        
        # Save report
        report_filename = f"{self.raw_data_path.stem}_validation_report.json"
        report_path = self.reports_dir / report_filename
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Validation report saved: {report_path}")
        return report

    def _stage_file(self) -> Path:
        """Copy validated file to staging directory."""
        self.staged_dir.mkdir(parents=True, exist_ok=True)

        staged_path = self.staged_dir / f"staged_{self.raw_data_path.name}"
        copy2(self.raw_data_path, staged_path)

        logger.info(f"Validated data staged at: {staged_path}")
        return staged_path

    def validate(self) -> Path:
        try:
            logger.info(f"Starting data validation: {self.raw_data_path.name}")

            # Run all validation checks
            self._validate_file_exists()
            df = self._load_data()

            self._validate_empty(df)
            self._validate_columns(df)
            self._validate_dtypes(df)
            self._validate_duplicates(df)
            self._validate_missing_values(df)
            self._validate_target(df)

            # Generate validation report
            self._generate_validation_report(df)

            # Stage validated file
            staged_path = self._stage_file()

            logger.info("Data validation completed successfully")
            
            return staged_path

        except Exception as e:
            logger.error(f"Data validation failed: {str(e)}")
            raise ChurnPipelineException(e)