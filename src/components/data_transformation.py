from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer

from src.config.transformation_config import (
    NUMERICAL_FEATURES,
    NOMINAL_CATEGORICAL_FEATURES,
    ORDINAL_CATEGORICAL_FEATURES,
    ORDINAL_CATEGORIES
)

from src.exception import ChurnPipelineException
from src.logging.logger import logger


class DataTransformation:
    """
    Builds and returns an UNFITTED preprocessing object.
    """

    def __init__(self):
        pass

    def get_preprocessor(self) -> ColumnTransformer:
        try:
            logger.info("Building preprocessing pipeline")

            numerical_transformer = Pipeline(steps=[
                ("scaler", StandardScaler())
            ])

            nominal_transformer = Pipeline(steps=[
                ("onehot", OneHotEncoder(
                    drop="first",
                    sparse_output=False,
                    handle_unknown="ignore"
                ))
            ])

            ordinal_transformer = Pipeline(steps=[
                ("ordinal", OrdinalEncoder(
                    categories=[ORDINAL_CATEGORIES],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1
                ))
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numerical_transformer, NUMERICAL_FEATURES),
                    ("nom", nominal_transformer, NOMINAL_CATEGORICAL_FEATURES),
                    ("ord", ordinal_transformer, ORDINAL_CATEGORICAL_FEATURES),
                ],
                remainder="passthrough"
            )

            logger.info("Preprocessing pipeline built successfully")

            return preprocessor

        except Exception as e:
            logger.error("Error while building preprocessing pipeline")
            raise ChurnPipelineException(e)
