from fastapi import APIRouter
from app.schemas.health import HealthResponse
from src.config import TICKER

router = APIRouter(
    prefix="/api/v1",
    tags=["health"],
)

@router.get("/health", response_model=HealthResponse)
def health_check():
    return {
        "status": "healthy",
        "service": f"LSTM {TICKER} Forecaster",
    }