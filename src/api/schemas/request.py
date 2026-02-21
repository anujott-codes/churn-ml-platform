from pydantic import BaseModel, Field, computed_field
from typing import Literal

from src.api.core.config import settings

class CustomerPredictionRequest(BaseModel):
    age: int = Field(..., description="Age of the customer", ge=18, le=100)
    gender: Literal['Male', 'Female'] = Field(..., description="Gender of the customer")
    tenure: int = Field(..., description="Duration in months for which a customer has been using the company's products or services",gt=0)
    usage_frequency: int = Field(..., description="Number of times that the customer has used the company’s services in the last month",ge=0)
    support_calls: int = Field(..., description="Number of calls that the customer has made to the customer support in the last month",ge=0)
    payment_delay: int = Field(..., description="Number of days that the customer has delayed their payment in the last month",ge=0)
    subscription_type: Literal['Basic', 'Standard', 'Premium'] = Field(..., description="Type of subscription the customer has choosen")
    contract_length: Literal['Annual', 'Monthly', 'Quarterly'] = Field(..., description="Duration of the contract that the customer has signed with the company")
    total_spend: float = Field(..., description="Total amount of money the customer has spent on the company's products or services",ge=0.0)
    last_interaction: int = Field(..., description="Number of days since the last interaction that the customer had with the company",ge=0)
    
    model_config = {
        "extra": "forbid",
        "json_schema_extra": {
            "example": {
                "age": 35,  
                "gender": "Male",
                "tenure": 24,
                "usage_frequency": 10,
                "support_calls": 5,
                "payment_delay": 2,
                "subscription_type": "Standard",
                "contract_length": "Annual",
                "total_spend": 1200.0,
                "last_interaction": 15
            }
        }
    }


class BatchPredictionRequest(BaseModel):
    records: list[CustomerPredictionRequest] = Field(..., description="List of customer records for batch prediction")