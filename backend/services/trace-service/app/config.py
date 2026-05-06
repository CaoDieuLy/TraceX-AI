"""Configuration for trace-service."""

import os
from pathlib import Path
from urllib.parse import quote_plus

from pydantic_settings import BaseSettings, SettingsConfigDict


def _build_default_database_url() -> str:
    """Build PostgreSQL connection URL for Docker/Coolify deployment."""
    direct_url = os.getenv("DATABASE_URL", "").strip()
    if direct_url and direct_url != "":
        return direct_url

    db_user = os.getenv("POSTGRES_USER", "mcpt_user")
    db_password = quote_plus(os.getenv("POSTGRES_PASSWORD", "Mcpt2026Secure"))
    db_host = os.getenv("POSTGRES_HOST", "postgres")
    db_port = os.getenv("POSTGRES_PORT", "5432")
    db_name = os.getenv("POSTGRES_DATABASE", os.getenv("POSTGRES_DB", "mcpt"))

    return f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"


class Settings(BaseSettings):
    app_name: str = "trace-service"
    api_prefix: str = "/api/v1"
    database_url: str = _build_default_database_url()
    storage_base_url: str = os.getenv("STORAGE_BASE_URL", "https://storage.example.com")
    video_storage_path: str = "/workspace/storage/videos"
    trace_output_path: str = "/workspace/storage/traces"
    max_trace_window_hours: int = 168  # 7 days max
    default_trace_window_hours: int = 24
    video_merge_enabled: bool = True
    max_segments_per_trace: int = 50

    model_config = SettingsConfigDict(extra="ignore")


settings = Settings()

# Normalize paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
settings.video_storage_path = str(Path(settings.video_storage_path).expanduser().resolve())
settings.trace_output_path = str(Path(settings.trace_output_path).expanduser().resolve())
