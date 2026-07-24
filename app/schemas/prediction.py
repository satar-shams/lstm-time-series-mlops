from pydantic import BaseModel, Field
from app.example import PREDICT_EXAMPLE
from src.config import WINDOW_SIZE

class PredictRequest(BaseModel):
    data: list[float]=Field(
        ...,
        min_length=WINDOW_SIZE,
        max_length=WINDOW_SIZE,
        description="The most recent closing prices used to predict the next value.",
        examples=[PREDICT_EXAMPLE],
    )

class PredictResponse(BaseModel):
    prediction: float = Field(
        ...,
        description="The predicted next closing price.",
    )
