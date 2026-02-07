import pandas as pd
import joblib
from pathlib import Path
import os

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer

from src.config.transformation_config import (
    PROCESSED_DATA_DIR,
    TRANSFORMED_DATA_DIR,
    NUMERICAL_FEATURES,
    NOMINAL_CATEGORICAL_FEATURES,
    ORDINAL_CATEGORICAL_FEATURES,
    TARGET_COLUMN,
    ORDINAL_CATEGORIES,
    PREPROCESSING_DIR,
    TRANSFORMED_TRAIN_FILENAME,
    TRANSFORMED_TEST_FILENAME
)
from src.exception import ChurnPipelineException
from src.logging.logger import logger


class DataTransformation:
    """
    Handles scaling and encoding.
    Reads from data/processed
    Writes to data/transformed
    Saves preprocessor to artifacts/preprocessing
    """

    def __init__(self, processed_train_path: Path, processed_test_path: Path):
        self.processed_dir = PROCESSED_DATA_DIR
        self.transformed_dir = TRANSFORMED_DATA_DIR
        self.preprocessing_dir = PREPROCESSING_DIR
        self.processed_train_path = processed_train_path
        self.processed_test_path = processed_test_path

        self.transformed_dir.mkdir(parents=True, exist_ok=True)
        self.preprocessing_dir.mkdir(parents=True, exist_ok=True)

    def _build_preprocessor(self) -> ColumnTransformer:
        try:
            numerical_transformer = Pipeline(steps=[
                ('scaler', StandardScaler())
            ])

            nominal_transformer = Pipeline(steps=[
                ('onehot', OneHotEncoder(
                    drop='first',
                    sparse_output=False,
                    handle_unknown='ignore'
                ))
            ])

            ordinal_transformer = Pipeline(steps=[
                ('ordinal', OrdinalEncoder(
                    categories=[ORDINAL_CATEGORIES],
                    handle_unknown='use_encoded_value',
                    unknown_value=-1
                ))
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, NUMERICAL_FEATURES),
                    ('nom', nominal_transformer, NOMINAL_CATEGORICAL_FEATURES),
                    ('ord', ordinal_transformer, ORDINAL_CATEGORICAL_FEATURES)
                ],
                remainder='passthrough'
            )

            return preprocessor

        except Exception as e:
            raise ChurnPipelineException(e)

    def initiate_data_transformation(self) -> dict:
        try:
    
            logger.info("DATA TRANSFORMATION STARTED")
    
            train_df = pd.read_csv(self.processed_train_path)
            test_df = pd.read_csv(self.processed_test_path)

            logger.info(f"Train shape before transform: {train_df.shape}")
            logger.info(f"Test shape before transform: {test_df.shape}")

            X_train = train_df.drop(columns=[TARGET_COLUMN])
            y_train = train_df[TARGET_COLUMN]

            X_test = test_df.drop(columns=[TARGET_COLUMN])
            y_test = test_df[TARGET_COLUMN]

            preprocessor = self._build_preprocessor()

            # Fit only on training data
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            # Preserve feature names
            feature_names = preprocessor.get_feature_names_out()

            train_out = pd.DataFrame(
                X_train_transformed,
                columns=feature_names
            )
            train_out[TARGET_COLUMN] = y_train.values

            test_out = pd.DataFrame(
                X_test_transformed,
                columns=feature_names
            )
            test_out[TARGET_COLUMN] = y_test.values

            # Save transformed datasets
           
            transformed_train_path = self.transformed_dir / TRANSFORMED_TRAIN_FILENAME
            transformed_test_path = self.transformed_dir / TRANSFORMED_TEST_FILENAME

            train_out.to_csv(transformed_train_path, index=False)
            test_out.to_csv(transformed_test_path, index=False)

            # Save preprocessor
            preprocessor_path = self.preprocessing_dir / "preprocessor.joblib"
            joblib.dump(preprocessor, preprocessor_path)

            logger.info(f"Transformed train saved to: {transformed_train_path}")
            logger.info(f"Transformed test saved to: {transformed_test_path}")
            logger.info(f"Preprocessor saved to: {preprocessor_path}")
            logger.info(f"Total features after transformation: {len(feature_names)}")

 
            logger.info("DATA TRANSFORMATION COMPLETED")


            return {
                "train_path": transformed_train_path,
                "test_path": transformed_test_path,
                "preprocessor_path": preprocessor_path,
                "num_features": len(feature_names),
            }

        except Exception as e:
            logger.error("Error during data transformation")
            raise ChurnPipelineException(e)
