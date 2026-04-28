from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
import os
import sys

_here = Path(__file__).parent
REPO_ROOT = _here.parent.parent.parent.parent.resolve()
A20_ROOT = REPO_ROOT
if str(A20_ROOT) not in sys.path:
    sys.path.insert(0, str(A20_ROOT))

loaded_envs: list[Path] = []


def _resolve_oauth_secret_path(raw_value: str, default_relative_path: str) -> Path:
    candidate = Path(raw_value or default_relative_path).expanduser()
    if candidate.is_absolute():
        return candidate
    return Path(os.getenv("MCPT_SECRETS_ROOT", "/workspace/a20-root/secrets")) / candidate


resolved_oauth_credentials_path = _resolve_oauth_secret_path(
    os.getenv("MCPT_OAUTH2_CREDENTIALS_FILE", ""),
    "oauth/oauth2_credentials.json",
)
resolved_oauth_token_path = _resolve_oauth_secret_path(
    os.getenv("MCPT_OAUTH2_TOKEN_FILE", ""),
    "oauth/oauth2_token.pickle",
)

try:
    from shared_secret_runtime import (  # noqa: E402
        load_runtime_env,
        resolve_oauth2_credentials_path,
        resolve_oauth2_token_path,
        shared_env_file,
    )

    loaded_envs = load_runtime_env(include_tracking_service_env=True, override=False)
    resolved_oauth_credentials_path = resolve_oauth2_credentials_path()
    resolved_oauth_token_path = resolve_oauth2_token_path()
    for env_file in loaded_envs:
        print(f"[config] Loaded env from {env_file}")
    if not loaded_envs:
        print(f"[config] WARNING: no env file found; expected shared env at {shared_env_file()}")
except ImportError:
    pass

def _resolve_project_root() -> Path:
    raw_value = os.getenv("PROJECT_ROOT", "").strip()
    if raw_value:
        candidate = Path(raw_value).expanduser()
        resolved = candidate.resolve() if candidate.is_absolute() else (REPO_ROOT / candidate).resolve()
        if resolved.exists():
            return resolved
    return REPO_ROOT


PROJECT_ROOT = _resolve_project_root()


class Settings(BaseSettings):
    app_name: str = "mcpt-tracking-service"

    # Paths - auto-detect from PROJECT_ROOT
    legacy_root: str = str(PROJECT_ROOT / "backend" / "legacy-engine")
    ingestion_work_root: str = str(PROJECT_ROOT / "storage" / "tracking-ingestion")
    video_conversion_output_dir: str = str(PROJECT_ROOT / "storage" / "video-conversion")
    video_download_output_dir: str = str(PROJECT_ROOT / "storage" / "tracking-outputs")
    tracking_artifact_root: str = str(PROJECT_ROOT / "storage" / "tracking-artifacts")

    # Camera calibration for 3D world projection
    camera_calibration_path: str = str(PROJECT_ROOT / "backend" / "config" / "camera_calibration.json")
    world_projection_calibration_path: str = str(PROJECT_ROOT / "backend" / "config" / "world_projection_calibration.json")

    # PostgreSQL
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_database: str = "video_tracking"
    postgres_user: str = "mcpt_user"
    postgres_password: str = os.getenv("POSTGRES_PASSWORD", "")

    google_drive_enabled: bool = False
    google_drive_oauth_credentials_file: Path = resolved_oauth_credentials_path
    google_drive_oauth_token_file: Path = resolved_oauth_token_path
    google_drive_make_public: bool = True
    google_drive_root_folder_id: str = "1gxKBTQ9BlqUmeashklclv429FDjr6Xbp"
    google_drive_vinuni_folder_id: str = "1gxKBTQ9BlqUmeashklclv429FDjr6Xbp"
    google_drive_vinuni_folder_name: str = "VinUni"
    google_drive_queue_folder_name: str = "Storage"
    google_drive_metadata_folder_name: str = "Metadata"

    tracking_runtime_mode: str = "production_ready"
    tracking_hyperparameter_overrides_json: str = ""  # disabled in strict mode
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
    download_remote_outputs: bool = False
    cleanup_remote_query_inputs: bool = True

    enable_trackeval: bool = True
    enable_geometry_gating: bool = True
    enable_corrective_cascade: bool = True

    # Compute .env path relative to this config file
    model_config = SettingsConfigDict(
        extra="ignore"
    )


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


settings.legacy_root = _normalize_runtime_path(settings.legacy_root, "backend/legacy-engine")
settings.ingestion_work_root = _normalize_runtime_path(settings.ingestion_work_root, "storage/tracking-ingestion")
settings.video_conversion_output_dir = _normalize_runtime_path(settings.video_conversion_output_dir, "storage/video-conversion")
settings.video_download_output_dir = _normalize_runtime_path(settings.video_download_output_dir, "storage/tracking-outputs")
settings.tracking_artifact_root = _normalize_runtime_path(settings.tracking_artifact_root, "storage/tracking-artifacts")
settings.camera_calibration_path = _normalize_runtime_path(settings.camera_calibration_path, "backend/config/camera_calibration.json")
settings.world_projection_calibration_path = _normalize_runtime_path(
    settings.world_projection_calibration_path,
    "backend/config/world_projection_calibration.json",
)
