from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "mcpt-metadata-service"
    api_prefix: str = "/api/v1"
    database_url: str = "postgresql+psycopg://mcpt_user:mcpt_password@postgres:5432/mcpt"
    legacy_metadata_dir: str = "/workspace/backend/legacy-engine/data/metadata"
    tracking_service_url: str = "http://tracking-service:8000"
    jwt_secret_key: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24
    video_storage_root: str = "/workspace/storage/videos"
    default_storage_backend: str = "local_volume"
    queue_local_root: str = "/workspace/storage/queue"
    queue_max_size: int = 32
    queue_poll_interval_seconds: int = 30
    google_drive_enabled: bool = False
    google_drive_credentials_file: str = ""
    google_drive_root_folder_id: str = ""
    google_drive_vinuni_folder_id: str = ""
    google_drive_vinuni_folder_name: str = "VinUni"
    google_drive_queue_folder_name: str = "Queue"
    google_drive_import_folder_name: str = "Import_New"
    google_drive_h265_folder_name: str = ".h265"
    google_drive_metadata_folder_name: str = "Metadata"
    google_drive_make_public: bool = True

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
