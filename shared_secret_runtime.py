from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parent
APP_ROOT = REPO_ROOT / "Multi-Camera-Person-Tracking-and-Re-Identification"


def canonical_secrets_root() -> Path:
    configured = os.getenv("MCPT_SECRETS_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser()
    return REPO_ROOT / "secrets"


def shared_env_file() -> Path:
    configured = os.getenv("MCPT_SHARED_ENV_FILE", "").strip()
    if configured:
        path = Path(configured).expanduser()
        if path.is_absolute():
            return path
        return canonical_secrets_root() / path
    return canonical_secrets_root() / "shared.env"


def tracking_service_env_file() -> Path:
    return APP_ROOT / "backend" / "services" / "tracking-service" / ".env"


def _resolve_secret_path(configured: str, default_relative_path: str) -> Path:
    if configured:
        candidate = Path(configured).expanduser()
        if candidate.is_absolute():
            return candidate
        return canonical_secrets_root() / candidate
    return canonical_secrets_root() / default_relative_path


def oauth2_credentials_candidates() -> list[Path]:
    configured = os.getenv("MCPT_OAUTH2_CREDENTIALS_FILE", "").strip()
    candidates = []
    if configured:
        candidates.append(_resolve_secret_path(configured, "oauth/oauth2_credentials.json"))
    candidates.append(canonical_secrets_root() / "oauth" / "oauth2_credentials.json")
    candidates.append(REPO_ROOT / "secret" / "oauth" / "oauth2_credentials.json")
    return candidates


def oauth2_token_candidates() -> list[Path]:
    configured = os.getenv("MCPT_OAUTH2_TOKEN_FILE", "").strip()
    candidates = []
    if configured:
        candidates.append(_resolve_secret_path(configured, "oauth/oauth2_token.pickle"))
    candidates.append(canonical_secrets_root() / "oauth" / "oauth2_token.pickle")
    candidates.append(REPO_ROOT / "secret" / "oauth" / "oauth2_token.pickle")
    return candidates


def _first_existing(candidates: Iterable[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def resolve_oauth2_credentials_path() -> Path:
    return _first_existing(oauth2_credentials_candidates()) or oauth2_credentials_candidates()[0]


def resolve_oauth2_token_path() -> Path:
    return _first_existing(oauth2_token_candidates()) or oauth2_token_candidates()[0]


def ensure_canonical_secret_dirs() -> dict[str, Path]:
    secrets_root = canonical_secrets_root()
    env_dir = secrets_root / "env"
    db_dir = secrets_root / "db"
    api_dir = secrets_root / "api"
    deploy_dir = secrets_root / "deploy"
    docker_dir = secrets_root / "docker"
    gpu_dir = secrets_root / "gpu"
    oauth_dir = secrets_root / "oauth"
    drive_dir = secrets_root / "google-drive"
    secrets_root.mkdir(parents=True, exist_ok=True)
    env_dir.mkdir(parents=True, exist_ok=True)
    db_dir.mkdir(parents=True, exist_ok=True)
    api_dir.mkdir(parents=True, exist_ok=True)
    deploy_dir.mkdir(parents=True, exist_ok=True)
    docker_dir.mkdir(parents=True, exist_ok=True)
    gpu_dir.mkdir(parents=True, exist_ok=True)
    oauth_dir.mkdir(parents=True, exist_ok=True)
    drive_dir.mkdir(parents=True, exist_ok=True)
    return {
        "secrets_root": secrets_root,
        "env_dir": env_dir,
        "db_dir": db_dir,
        "api_dir": api_dir,
        "deploy_dir": deploy_dir,
        "docker_dir": docker_dir,
        "gpu_dir": gpu_dir,
        "oauth_dir": oauth_dir,
        "drive_dir": drive_dir,
        "shared_env": shared_env_file(),
    }


def modular_env_files() -> list[Path]:
    env_dir = canonical_secrets_root() / "env"
    ordered_paths = [
        shared_env_file(),
        "env/shared.env",
        "db/postgres.env",
        "api/providers.env",
        "oauth/oauth.env",
        "google-drive/google-drive.env",
        "deploy/runtime.env",
        "docker/compose.env",
        "gpu/runtime.env",
    ]
    files: list[Path] = []
    seen: set[Path] = set()
    for item in ordered_paths:
        path = item if isinstance(item, Path) else canonical_secrets_root() / item
        if path not in seen:
            files.append(path)
            seen.add(path)
    if env_dir.exists():
        for path in sorted(env_dir.glob("*.env")):
            if path not in seen:
                files.append(path)
                seen.add(path)
    return files


def load_runtime_env(*, include_tracking_service_env: bool = True, override: bool = True) -> list[Path]:
    loaded: list[Path] = []
    candidate_files: list[Path] = []

    root_env = REPO_ROOT / ".env"
    if root_env.exists():
        candidate_files.append(root_env)

    candidate_files.extend(modular_env_files())

    for env_file in candidate_files:
        if env_file.exists():
            load_dotenv(env_file, override=override)
            loaded.append(env_file)

    secrets_root = canonical_secrets_root()
    shared_env = shared_env_file()
    oauth2_credentials = resolve_oauth2_credentials_path()
    oauth2_token = resolve_oauth2_token_path()

    os.environ["MCPT_SECRETS_ROOT"] = str(secrets_root)
    os.environ["MCPT_SHARED_ENV_FILE"] = str(shared_env)
    os.environ["MCPT_OAUTH2_CREDENTIALS_FILE"] = str(oauth2_credentials)
    os.environ["MCPT_OAUTH2_TOKEN_FILE"] = str(oauth2_token)
    return loaded


def build_google_drive_oauth_credentials():
    import google.auth.transport.requests

    token_path = resolve_oauth2_token_path()
    if not token_path.exists():
        raise FileNotFoundError(
            f"OAuth2 token not found: {token_path}. "
            "Authenticate first so Google Drive runtime can reuse the stored token."
        )
    with token_path.open("rb") as handle:
        credentials = pickle.load(handle)
    if getattr(credentials, "expired", False):
        credentials.refresh(google.auth.transport.requests.Request())
        try:
            with token_path.open("wb") as handle:
                pickle.dump(credentials, handle)
        except OSError:
            # Secrets volume may be read-only (container mount :ro).
            # Token is refreshed in memory and valid for this session.
            pass
    return credentials


def build_google_drive_oauth_service():
    from googleapiclient.discovery import build

    credentials = build_google_drive_oauth_credentials()
    return build("drive", "v3", credentials=credentials, cache_discovery=False)
