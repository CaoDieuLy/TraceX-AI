import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


def _detect_storage_root() -> Path:
    candidates = [
        Path(os.getenv("A20_STORAGE_ROOT", "")),
        Path("/storage"),
        Path("/workspace/storage"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return Path("/storage")


class Settings(BaseSettings):
    app_name: str = "ai-service"
    host: str = "0.0.0.0"
    port: int = 8000

    storage_root: Path = _detect_storage_root()
    model_cache_dir: Path = Path("/storage/model_cache")
    video_cache_dir: Path = Path("/storage/video_cache")

    detection_model: str = "yolov8n"
    tracking_model: str = "ocsort"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    detection_confidence_threshold: float = 0.5
    detection_iou_threshold: float = 0.45

    max_frames_per_video: int = 300
    sample_fps: int = 5

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
settings.model_cache_dir.mkdir(parents=True, exist_ok=True)
settings.video_cache_dir.mkdir(parents=True, exist_ok=True)
