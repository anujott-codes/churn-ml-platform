import mlflow.pyfunc
import pandas as pd
import numpy as np

class ChurnModelWrapper(mlflow.pyfunc.PythonModel):

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def predict(self, context, model_input: pd.DataFrame) -> np.ndarray:
        probs = self.pipeline.predict_proba(model_input)
        return probs[:, 1]