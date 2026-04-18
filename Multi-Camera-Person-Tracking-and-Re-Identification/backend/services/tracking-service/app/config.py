from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "mcpt-tracking-service"
    legacy_root: str = "/workspace/backend/legacy-engine"
    ingestion_work_root: str = "/workspace/storage/tracking-ingestion"
    video_conversion_output_dir: str = "/workspace/storage/video-conversion"
    video_download_output_dir: str = "/workspace/storage/tracking-outputs"

    google_drive_enabled: bool = False
    google_drive_credentials_file: str = ""
    google_drive_make_public: bool = True

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

    ffmpeg_crf: int = 28
    ffmpeg_preset: str = "medium"
    ffmpeg_audio_codec: str = "aac"
    ffmpeg_overwrite_output: bool = False

    lightning_api_base_url: str = ""
    lightning_api_endpoint: str = "/predict"
    lightning_api_token: str = ""
    lightning_api_auth_header: str = "Authorization"
    lightning_api_auth_prefix: str = "Bearer "
    lightning_timeout_seconds: int = 180

    enable_trackeval: bool = True
    enable_geometry_gating: bool = True
    enable_corrective_cascade: bool = True

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
