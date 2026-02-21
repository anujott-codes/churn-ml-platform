from pydantic import BaseModel
from typing import List, Literal

class PredictionResponse(BaseModel):
    prediction: Literal[0,1]
    probability: float
    churn: bool

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]

class HealthResponse(BaseModel):
    status: Literal["OK", "degraded"]

class ModelInfoResponse(BaseModel):
    model_name: str
    version: str
    alias: str