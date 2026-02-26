from fastapi import APIRouter

router = APIRouter()


@router.get("/", tags=["Root"])
async def home():
    return {
        "service": "Customer Churn ML Platform API",
        "status": "running"
    }