from pydantic_settings import BaseSettings, SettingsConfigDict

from src.config import TICKER


class APISettings(BaseSettings):

    # ============================================================
    # API Configuration
    # ============================================================

    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    APP_VERSION: str = "1.3.0"

    title: str = f"LSTM {TICKER} Forecaster"


    # ============================================================
    # Model Loading Configuration
    # ============================================================

    MODEL_SOURCE_LOCAL: str = "local"
    MODEL_SOURCE_MLFLOW: str = "mlflow"

    MODEL_SOURCE: str

    MODEL_PATH: str = "models/production_model.keras"
    SCALER_PATH: str = "models/scaler.bin"


    # ============================================================
    # MLflow Configuration
    # ============================================================

    MLFLOW_ALIAS: str = "production"


    model_config = SettingsConfigDict(
        env_file=".env"
    )


settings = APISettings()