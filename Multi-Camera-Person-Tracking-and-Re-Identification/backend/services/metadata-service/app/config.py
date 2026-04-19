import os
import sys
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_HERE = Path(__file__).resolve()
A20_ROOT = _HERE.parents[5]
if str(A20_ROOT) not in sys.path:
    sys.path.insert(0, str(A20_ROOT))

loaded_envs: list[Path] = []


def _resolve_drive_credentials_path(raw_value: str) -> Path:
    candidate = Path(raw_value or "/workspace/project/secrets/google-drive/drive-sa.json").expanduser()
    if candidate.is_absolute():
        return candidate
    return Path(os.getenv("MCPT_SECRETS_ROOT", "/workspace/project/secrets")) / candidate


default_drive_credentials_file = _resolve_drive_credentials_path(os.getenv("GOOGLE_DRIVE_CREDENTIALS_FILE", ""))

try:
    from shared_secret_runtime import load_runtime_env, resolve_google_drive_service_account_path  # noqa: E402

    loaded_envs = load_runtime_env(include_tracking_service_env=True, override=True)
    default_drive_credentials_file = resolve_google_drive_service_account_path()
except ImportError:
    pass


def _build_default_database_url() -> str:
    direct_url = os.getenv("DATABASE_URL", "").strip()
    if direct_url:
        return direct_url
    db_user = os.getenv("POSTGRES_USER", "mcpt_user")
    db_password = os.getenv("POSTGRES_PASSWORD", "")
    db_host = os.getenv("POSTGRES_HOST", "postgres")
    db_port = os.getenv("POSTGRES_PORT", "5432")
    db_name = os.getenv("POSTGRES_DATABASE", os.getenv("POSTGRES_DB", "mcpt"))
    return f"postgresql+psycopg://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"


class Settings(BaseSettings):
    app_name: str = "mcpt-metadata-service"
    api_prefix: str = "/api/v1"
    database_url: str = _build_default_database_url()
    legacy_metadata_dir: str = "/workspace/backend/legacy-engine/data/metadata"
    tracking_service_url: str = "http://tracking-service:8000"
    jwt_secret_key: str = os.getenv("JWT_SECRET_KEY", "")
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24
    video_storage_root: str = "/workspace/storage/videos"
    default_storage_backend: str = "local_volume"
    queue_local_root: str = "/workspace/storage/queue"
    queue_max_size: int = 32
    queue_poll_interval_seconds: int = 30
    queue_parallel_jobs: int = 4
    queue_download_workers: int = 4
    google_drive_enabled: bool = False
    google_drive_credentials_file: str = str(default_drive_credentials_file)
    google_drive_root_folder_id: str = ""
    google_drive_vinuni_folder_id: str = ""
    google_drive_vinuni_folder_name: str = "VinUni"
    google_drive_queue_folder_name: str = "Queue"
    google_drive_import_folder_name: str = "Import_New"
    google_drive_h265_folder_name: str = ".h265"
    google_drive_metadata_folder_name: str = "Metadata"
    google_drive_make_public: bool = True

    model_config = SettingsConfigDict(extra="ignore")


settings = Settings()
