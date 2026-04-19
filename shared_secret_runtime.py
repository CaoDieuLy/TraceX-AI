from __future__ import annotations

import os
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
        return Path(configured).expanduser()
    return canonical_secrets_root() / "shared.env"


def tracking_service_env_file() -> Path:
    return APP_ROOT / "backend" / "services" / "tracking-service" / ".env"


def oauth2_credentials_candidates() -> list[Path]:
    configured = os.getenv("MCPT_OAUTH2_CREDENTIALS_FILE", "").strip()
    candidates = []
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend(
        [
            canonical_secrets_root() / "oauth" / "oauth2_credentials.json",
            REPO_ROOT / "oauth2_credentials.json",
        ]
    )
    return candidates


def oauth2_token_candidates() -> list[Path]:
    configured = os.getenv("MCPT_OAUTH2_TOKEN_FILE", "").strip()
    candidates = []
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend(
        [
            canonical_secrets_root() / "oauth" / "oauth2_token.pickle",
            REPO_ROOT / "oauth2_token.pickle",
        ]
    )
    return candidates


def google_drive_service_account_candidates() -> list[Path]:
    configured = os.getenv("GOOGLE_DRIVE_CREDENTIALS_FILE", "").strip()
    candidates = []
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend(
        [
            canonical_secrets_root() / "google-drive" / "drive-sa.json",
            APP_ROOT / "backend" / "services" / "tracking-service" / "credentials" / "mcpt-tracker-sa.json",
        ]
    )
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


def resolve_google_drive_service_account_path() -> Path:
    return _first_existing(google_drive_service_account_candidates()) or google_drive_service_account_candidates()[0]


def ensure_canonical_secret_dirs() -> dict[str, Path]:
    secrets_root = canonical_secrets_root()
    oauth_dir = secrets_root / "oauth"
    drive_dir = secrets_root / "google-drive"
    secrets_root.mkdir(parents=True, exist_ok=True)
    oauth_dir.mkdir(parents=True, exist_ok=True)
    drive_dir.mkdir(parents=True, exist_ok=True)
    return {
        "secrets_root": secrets_root,
        "oauth_dir": oauth_dir,
        "drive_dir": drive_dir,
        "shared_env": shared_env_file(),
    }


def load_runtime_env(*, include_tracking_service_env: bool = True, override: bool = True) -> list[Path]:
    loaded: list[Path] = []
    candidate_files: list[Path] = []

    root_env = REPO_ROOT / ".env"
    if root_env.exists():
        candidate_files.append(root_env)

    if include_tracking_service_env:
        candidate_files.append(tracking_service_env_file())

    candidate_files.append(shared_env_file())

    for env_file in candidate_files:
        if env_file.exists():
            load_dotenv(env_file, override=override)
            loaded.append(env_file)

    service_account_path = resolve_google_drive_service_account_path()
    if service_account_path.exists():
        os.environ.setdefault("GOOGLE_DRIVE_CREDENTIALS_FILE", str(service_account_path))

    os.environ.setdefault("MCPT_SECRETS_ROOT", str(canonical_secrets_root()))
    os.environ.setdefault("MCPT_SHARED_ENV_FILE", str(shared_env_file()))
    os.environ.setdefault("MCPT_OAUTH2_CREDENTIALS_FILE", str(resolve_oauth2_credentials_path()))
    os.environ.setdefault("MCPT_OAUTH2_TOKEN_FILE", str(resolve_oauth2_token_path()))
    return loaded
