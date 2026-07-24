from pydantic_settings import BaseSettings, SettingsConfigDict
from src.config import TICKER

class APISettings(BaseSettings):
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    APP_VERSION: str = "1.3.0"
    title: str = f"LSTM {TICKER} Forecaster"

    model_config = SettingsConfigDict(
        env_file=".env"
    )
    
settings = APISettings()