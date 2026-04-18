from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "mcpt-tracking-service"
    legacy_root: str = "/workspace/backend/legacy-engine"
    ingestion_work_root: str = "/workspace/storage/tracking-ingestion"
    video_conversion_output_dir: str = "/workspace/storage/video-conversion"
    video_download_output_dir: str = "/workspace/storage/tracking-outputs"

    # Google Drive settings
    google_drive_enabled: bool = False
    google_drive_credentials_file: str = ""
    google_drive_make_public: bool = True

    # PostgreSQL Database
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_database: str = "video_tracking"
    postgres_user: str = "postgres"
    postgres_password: str = ""

    # Queue settings
    max_queue_size: int = 100
    queue_fifo_enabled: bool = True

    # Tracking settings
    tracking_use_mock: bool = True
    tracking_runtime_mode: str = "design_ready"
    pipeline_profile: str = "accuracy_first"
    tracking_hyperparameter_overrides_json: str = ""
    gpu_hardware_profile: str = "l4"
    gpu_hardware_overrides_json: str = ""
    gpu_count: int = 1
    host_cpu_count: int = 16
    host_ram_gb: int = 64
    uvicorn_workers: int = 1

    # Video conversion settings
    ffmpeg_crf: int = 28
    ffmpeg_preset: str = "medium"
    ffmpeg_audio_codec: str = "aac"
    ffmpeg_overwrite_output: bool = False

    # Google Drive folder structure
    gdrive_base_folder: str = "VinUni"
    gdrive_queue_folder: str = "Queue"
    gdrive_import_new_folder: str = "Import_New"
    gdrive_queue_h265_subfolder: str = ".h265"
    gdrive_queue_metadata_subfolder: str = "Metadata"

    # Lightning AI settings
    lightning_api_base_url: str = ""
    lightning_api_endpoint: str = "/predict"
    lightning_api_token: str = ""
    lightning_api_auth_header: str = "Authorization"
    lightning_api_auth_prefix: str = "Bearer "
    lightning_timeout_seconds: int = 180

    # Feature flags
    enable_trackeval: bool = True
    enable_geometry_gating: bool = True
    enable_corrective_cascade: bool = True

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
