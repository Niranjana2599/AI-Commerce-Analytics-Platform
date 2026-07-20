"""Environment-aware application settings."""

from dataclasses import dataclass
from pathlib import Path
import os


PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Settings:
    app_name: str = "AI Commerce Analytics API"
    api_prefix: str = "/api/v1"
    environment: str = os.getenv("APP_ENV", "development")
    allowed_origins: tuple[str, ...] = tuple(os.getenv("CORS_ORIGINS", "*").split(","))
    data_path: Path = Path(os.getenv("MASTER_DATA_PATH", PROJECT_ROOT / "data/processed/master_df.parquet"))
    models_dir: Path = Path(os.getenv("MODELS_DIR", PROJECT_ROOT / "models"))


settings = Settings()
