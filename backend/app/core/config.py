"""Environment-aware application settings."""

from dataclasses import dataclass
from pathlib import Path
import os

# ------------------------------------------------------------
# Project root
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ------------------------------------------------------------
# Application settings
# ------------------------------------------------------------

class Settings:

    # Application
    app_name: str = "AI Commerce Analytics API"
    api_prefix: str = "/api/v1"

    # Environment
    environment: str = os.getenv(
        "APP_ENV",
        "development"
    )

    # CORS
    allowed_origins: tuple[str, ...] = tuple(
        os.getenv(
            "CORS_ORIGINS",
            "*"
        ).split(",")
    )

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    data_path: Path = Path(
        os.getenv(
            "MASTER_DATA_PATH",
            PROJECT_ROOT / "data/processed/master_df.parquet"
        )
    )

    # --------------------------------------------------------
    # ML models
    # --------------------------------------------------------

    models_dir: Path = Path(
        os.getenv(
            "MODELS_DIR",
            PROJECT_ROOT / "models"
        )
    )

    # --------------------------------------------------------
    # RAG / LLMOps
    # --------------------------------------------------------

    rag_ops_dir: Path = Path(
        os.getenv(
            "RAG_OPS_DIR",
            PROJECT_ROOT / "rag_ops"
        )
    )

    # --------------------------------------------------------
    # Ollama
    #
    # Backend runs in Docker.
    # Ollama runs on the Windows host.
    # --------------------------------------------------------

    ollama_base_url: str = os.getenv(
        "OLLAMA_BASE_URL",
        "http://host.docker.internal:11434"
    )

    ollama_model: str = os.getenv(
        "OLLAMA_MODEL",
        "llama3.2"
    )

    # Enable LLM-based RAG answer generation
    rag_llm_enabled: bool = (
        os.getenv(
            "RAG_LLM_ENABLED",
            "true"
        ).lower()
        == "true"
    )

    # --------------------------------------------------------
    # LangSmith
    # --------------------------------------------------------

    langsmith_tracing_enabled: bool = (
        os.getenv(
            "LANGSMITH_TRACING",
            os.getenv(
                "LANGCHAIN_TRACING_V2",
                "false"
            )
        ).lower()
        == "true"
    )

    langsmith_project: str = os.getenv(
        "LANGCHAIN_PROJECT",
        "AI-Commerce-Analytics-Platform"
    )

    langsmith_endpoint: str = os.getenv(
        "LANGCHAIN_ENDPOINT",
        "https://api.smith.langchain.com"
    )


# ------------------------------------------------------------
# Create settings instance
# ------------------------------------------------------------

settings = Settings()