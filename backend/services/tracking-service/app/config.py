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


def _default_workers() -> int:
    configured = os.getenv("MCPT_METADATA_WORKERS", "").strip()
    if configured.isdigit():
        return max(1, int(configured))
    return max(1, min((os.cpu_count() or 1), 8))


class Settings(BaseSettings):
    app_name: str = "tracking-service"
    host: str = "0.0.0.0"
    port: int = 8000

    storage_root: Path = _detect_storage_root()
    model_cache_dir: Path = Path("/storage/model_cache")
    video_cache_dir: Path = Path("/storage/video_cache")

    workers: int = _default_workers()

    rf_detr_weights: str = os.getenv("MCPT_RF_DETR_WEIGHTS", "")
    detector_batch_size: int = int(os.getenv("MCPT_DETECTOR_BATCH_SIZE", "8"))
    stream_batch_size: int = int(os.getenv("MCPT_STREAM_BATCH_SIZE", "150"))

    sample_fps: int = 4
    max_frames_per_video: int = 300

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
settings.model_cache_dir.mkdir(parents=True, exist_ok=True)
settings.video_cache_dir.mkdir(parents=True, exist_ok=True)
