import os
import sys
from pathlib import Path
from urllib.parse import quote_plus

from pydantic_settings import BaseSettings, SettingsConfigDict

_HERE = Path(__file__).resolve()


def _detect_a20_root() -> Path | None:
    env_root = os.getenv("A20_ROOT", "").strip()
    candidates = []
    if env_root:
        candidates.append(Path(env_root).expanduser())
    candidates.extend([Path("/workspace/a20-root"), _HERE.parent, *_HERE.parents])
    for candidate in candidates:
        if (candidate / "shared_secret_runtime.py").exists():
            return candidate
    return None


A20_ROOT = _detect_a20_root()
if A20_ROOT and str(A20_ROOT) not in sys.path:
    sys.path.insert(0, str(A20_ROOT))

def _resolve_project_root() -> Path:
    repo_root = _HERE.parent.parent.parent.parent.parent.resolve()
    raw_value = os.getenv("PROJECT_ROOT", "").strip()
    if raw_value:
        candidate = Path(raw_value).expanduser()
        resolved = candidate.resolve() if candidate.is_absolute() else (repo_root / candidate).resolve()
        if resolved.exists():
            return resolved
    return repo_root


PROJECT_ROOT = _resolve_project_root()

loaded_envs: list[Path] = []


def _resolve_oauth_secret_path(raw_value: str, default_relative_path: str) -> Path:
    candidate = Path(raw_value or default_relative_path).expanduser()
    if candidate.is_absolute():
        return candidate
    return Path(os.getenv("MCPT_SECRETS_ROOT", "/workspace/a20-root/secrets")) / candidate


default_oauth_credentials_file = _resolve_oauth_secret_path(
    os.getenv("MCPT_OAUTH2_CREDENTIALS_FILE", ""),
    "oauth/oauth2_credentials.json",
)
default_oauth_token_file = _resolve_oauth_secret_path(
    os.getenv("MCPT_OAUTH2_TOKEN_FILE", ""),
    "oauth/oauth2_token.pickle",
)

try:
    from shared_secret_runtime import (  # noqa: E402
        load_runtime_env,
        resolve_oauth2_credentials_path,
        resolve_oauth2_token_path,
    )

    loaded_envs = load_runtime_env(include_tracking_service_env=True, override=False)
    default_oauth_credentials_file = resolve_oauth2_credentials_path()
    default_oauth_token_file = resolve_oauth2_token_path()
except ImportError:
    pass


def _build_default_database_url() -> str:
    direct_url = os.getenv("DATABASE_URL", "").strip()
    if direct_url:
        return direct_url
    db_user = os.getenv("POSTGRES_USER", "mcpt_user")
    db_password = quote_plus(os.getenv("POSTGRES_PASSWORD", ""))
    db_host = os.getenv("POSTGRES_HOST", "postgres")
    db_port = os.getenv("POSTGRES_PORT", "5432")
    db_name = os.getenv("POSTGRES_DATABASE", os.getenv("POSTGRES_DB", "mcpt"))
    return f"postgresql+psycopg://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"


class Settings(BaseSettings):
    app_name: str = "mcpt-metadata-service"
    api_prefix: str = "/api/v1"
    database_url: str = _build_default_database_url()
    tracking_service_url: str = "http://tracking-service:8000"
    public_api_base_url: str = os.getenv("NEXT_PUBLIC_API_GATEWAY_URL", "").strip()
    lightning_api_token: str = ""
    lightning_api_auth_header: str = "Authorization"
    lightning_api_auth_prefix: str = "Bearer "
    tracking_request_timeout_seconds: int = 1800
    jwt_secret_key: str = os.getenv("JWT_SECRET_KEY", "")
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 1440
    bootstrap_admin_email: str = os.getenv("BOOTSTRAP_ADMIN_EMAIL", "").strip().lower()
    bootstrap_admin_password: str = os.getenv("BOOTSTRAP_ADMIN_PASSWORD", "").strip()
    bootstrap_admin_full_name: str = os.getenv("BOOTSTRAP_ADMIN_FULL_NAME", "Administrator").strip()
    video_storage_root: str = str(PROJECT_ROOT / "storage" / "videos")
    tracking_output_root: str = str(PROJECT_ROOT / "storage" / "tracking-output")
    default_storage_backend: str = "local_volume"
    queue_local_root: str = str(PROJECT_ROOT / "storage" / "queue")
    queue_video_folder_name: str = "Videos"
    queue_max_size: int = 32
    queue_poll_interval_seconds: int = 30
    queue_parallel_jobs: int = 4
    queue_download_workers: int = 4
    storage_ingest_enabled: bool = True
    storage_ingest_root: str = str(PROJECT_ROOT / "storage")
    storage_ingest_batch_size: int = 50
    storage_ingest_min_file_age_seconds: int = 2
    storage_ingest_processed_dir_name: str = "ProcessedStorage"
    google_drive_enabled: bool = False
    google_drive_oauth_credentials_file: str = str(default_oauth_credentials_file)
    google_drive_oauth_token_file: str = str(default_oauth_token_file)
    google_drive_root_folder_id: str = "1gxKBTQ9BlqUmeashklclv429FDjr6Xbp"
    google_drive_vinuni_folder_id: str = "1gxKBTQ9BlqUmeashklclv429FDjr6Xbp"
    google_drive_vinuni_folder_name: str = "VinUni"
    google_drive_queue_folder_name: str = "Storage"
    google_drive_metadata_folder_name: str = "Metadata"
    google_drive_make_public: bool = True

    model_config = SettingsConfigDict(extra="ignore")


settings = Settings()


def _normalize_runtime_path(raw_value: str, fallback_relative_path: str) -> str:
    raw_text = str(raw_value or "").strip()
    candidate = Path(raw_text).expanduser()
    if not str(candidate):
        return str((PROJECT_ROOT / fallback_relative_path).resolve())
    if candidate.is_absolute():
        if candidate.exists():
            return str(candidate.resolve())
        if os.name != "nt":
            return str(candidate)
        resolved = candidate.resolve()
        return str((PROJECT_ROOT / fallback_relative_path).resolve()) if raw_text.startswith(("/workspace", "\\workspace")) else str(resolved)
    return str((PROJECT_ROOT / candidate).resolve())


settings.video_storage_root = _normalize_runtime_path(settings.video_storage_root, "storage/videos")
settings.tracking_output_root = _normalize_runtime_path(settings.tracking_output_root, "storage/tracking-output")
settings.queue_local_root = _normalize_runtime_path(settings.queue_local_root, "storage/queue")
settings.storage_ingest_root = _normalize_runtime_path(settings.storage_ingest_root, "storage")
