from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "mcpt-metadata-service"
    api_prefix: str = "/api/v1"
    database_url: str = "postgresql+psycopg://mcpt_user:mcpt_password@postgres:5432/mcpt"
    legacy_metadata_dir: str = "/workspace/backend/legacy-engine/data/metadata"
    jwt_secret_key: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24
    video_storage_root: str = "/workspace/storage/videos"
    default_storage_backend: str = "local_volume"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
