"""Configuration settings for query-service."""

from __future__ import annotations

import os
from pathlib import Path


class Settings:
    """Settings for query-service."""

    # Database
    database_url: str = os.getenv("DATABASE_URL", "")

    # Translation model path
    translation_model_path: Path = Path(os.getenv("TRANSLATION_MODEL_PATH", "/workspace/models/translation"))
    translation_vi_en_model: str = os.getenv("TRANSLATION_VI_EN_MODEL", "Helsinki-NLP/opus-mt-vi-en")
    translation_en_vi_model: str = os.getenv("TRANSLATION_EN_VI_MODEL", "Helsinki-NLP/opus-mt-en-vi")

    # Tracking service (for remote ranking)
    tracking_service_url: str = os.getenv("TRACKING_SERVICE_URL", "")
    lightning_api_token: str = os.getenv("LIGHTNING_API_TOKEN", "")
    lightning_api_auth_header: str = os.getenv("LIGHTNING_API_AUTH_HEADER", "Authorization")
    lightning_api_auth_prefix: str = os.getenv("LIGHTNING_API_AUTH_PREFIX", "Bearer ")
    tracking_request_timeout_seconds: int = int(os.getenv("TRACKING_REQUEST_TIMEOUT_SECONDS", "120"))
    tracking_startup_max_wait_seconds: int = int(os.getenv("TRACKING_STARTUP_MAX_WAIT_SECONDS", "60"))
    tracking_startup_poll_interval_seconds: int = int(os.getenv("TRACKING_STARTUP_POLL_INTERVAL_SECONDS", "5"))
    tracking_health_timeout_seconds: int = int(os.getenv("TRACKING_HEALTH_TIMEOUT_SECONDS", "10"))
    tracking_startup_retry_attempts: int = int(os.getenv("TRACKING_STARTUP_RETRY_ATTEMPTS", "3"))

    # Storage
    preview_root: Path = Path(os.getenv("PREVIEW_ROOT", "/workspace/storage/candidate-previews"))

    # Public API
    public_api_base_url: str = os.getenv("PUBLIC_API_BASE_URL", "")


settings = Settings()
