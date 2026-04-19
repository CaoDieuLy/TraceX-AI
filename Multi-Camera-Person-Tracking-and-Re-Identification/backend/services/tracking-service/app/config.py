from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv
import os

# Load .env from tracking-service root (2 levels up from this file)
_here = Path(__file__).parent
_env_file = _here.parent / ".env"
if _env_file.exists():
    load_dotenv(_env_file, override=True)
    print(f"[config] Loaded .env from {_env_file}")
else:
    print(f"[config] WARNING: .env not found at {_env_file}")

# Compute PROJECT_ROOT from env or file location
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", _here.parent.parent.parent.parent))


class Settings(BaseSettings):
    app_name: str = "mcpt-tracking-service"

    # Paths - auto-detect from PROJECT_ROOT
    legacy_root: str = str(PROJECT_ROOT / "backend" / "legacy-engine")
    ingestion_work_root: str = str(PROJECT_ROOT / "storage" / "tracking-ingestion")
    video_conversion_output_dir: str = str(PROJECT_ROOT / "storage" / "video-conversion")
    video_download_output_dir: str = str(PROJECT_ROOT / "storage" / "tracking-outputs")

    # Camera calibration for 3D world projection
    camera_calibration_path: str = str(PROJECT_ROOT / "backend" / "config" / "camera_calibration.json")

    # PostgreSQL
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_database: str = "video_tracking"
    postgres_user: str = "mcpt_user"
    postgres_password: str = "Mcpt@2026!Secure"

    google_drive_enabled: bool = False
    google_drive_credentials_file: Path = Path("")
    google_drive_make_public: bool = True
    google_drive_root_folder_id: str = ""
    google_drive_vinuni_folder_id: str = ""
    google_drive_vinuni_folder_name: str = "VinUni"
    google_drive_queue_folder_name: str = "Queue"
    google_drive_import_folder_name: str = "Import_New"
    google_drive_h265_folder_name: str = ".h265"
    google_drive_metadata_folder_name: str = "Metadata"

    tracking_runtime_mode: str = "production_ready"
    pipeline_profile: str = "accuracy_first"
    tracking_hyperparameter_overrides_json: str = ""
    gpu_hardware_profile: str = "auto"
    gpu_hardware_overrides_json: str = ""
    gpu_count: int = 1
    host_cpu_count: int = 16
    host_ram_gb: int = 64
    uvicorn_workers: int = 1

    ffmpeg_crf: int = 28
    ffmpeg_preset: str = "medium"
    ffmpeg_audio_codec: str = "aac"
    ffmpeg_overwrite_output: bool = False

    lightning_api_base_url: str = ""
    lightning_api_endpoint: str = "/api/v1/ai/worker"
    lightning_api_token: str = ""
    lightning_api_auth_header: str = "Authorization"
    lightning_api_auth_prefix: str = "Bearer "
    lightning_timeout_seconds: int = 180

    enable_trackeval: bool = True
    enable_geometry_gating: bool = True
    enable_corrective_cascade: bool = True

    # Compute .env path relative to this config file
    _here = Path(__file__).parent
    model_config = SettingsConfigDict(
        env_file=str(_here / ".env"),
        extra="ignore"
    )


settings = Settings()
