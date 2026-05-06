"""Shared configuration settings."""

from __future__ import annotations

import os
from pathlib import Path


class Settings:
    """Shared settings for all services."""

    # Database
    database_url: str = os.getenv("DATABASE_URL", "")

    # Paths
    project_root: Path = Path(__file__).parent.parent.parent.parent
    a20_root: Path | None = Path(os.getenv("A20_ROOT", "")) if os.getenv("A20_ROOT") else None

    # Storage
    video_storage_root: Path = Path(os.getenv("VIDEO_STORAGE_ROOT", "/workspace/storage/videos"))
    tracking_output_root: Path = Path(os.getenv("TRACKING_OUTPUT_ROOT", "/workspace/storage/tracking-output"))
    queue_local_root: Path = Path(os.getenv("QUEUE_LOCAL_ROOT", "/workspace/storage/queue"))

    # Services
    tracking_service_url: str = os.getenv("TRACKING_SERVICE_URL", "")
    lightning_api_token: str = os.getenv("LIGHTNING_API_TOKEN", "")
    lightning_api_auth_header: str = os.getenv("LIGHTNING_API_AUTH_HEADER", "Authorization")
    lightning_api_auth_prefix: str = os.getenv("LIGHTNING_API_AUTH_PREFIX", "Bearer ")

    # Public API
    public_api_base_url: str = os.getenv("PUBLIC_API_BASE_URL", "")


settings = Settings()
