"""
shared_secret_runtime.py — Shared secret/env loader cho cả tracking-service và metadata-service.

Tại sao ở root repo:
  - Tracking service (LightningAI): sys.path include REPO_ROOT
  - Metadata service (Docker VPS): Dockerfile copy sang /workspace/a20-root/,
    PYTHONPATH=/workspace/a20-root/ → cả hai đều import được

Chức năng chính:
  - load_runtime_env()          → load secrets/shared.env vào os.environ
  - build_google_drive_oauth_service() → khởi tạo Google Drive API client
  - resolve_oauth2_credentials_path() → tìm file credentials.json
  - resolve_oauth2_token_path()       → tìm file token.pickle
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parent


def canonical_secrets_root() -> Path:
    configured = os.getenv("MCPT_SECRETS_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser()
    return REPO_ROOT / "secrets"


def shared_env_file() -> Path:
    configured = os.getenv("MCPT_SHARED_ENV_FILE", "").strip()
    if configured:
        path = Path(configured).expanduser()
        return path if path.is_absolute() else canonical_secrets_root() / path
    return canonical_secrets_root() / "shared.env"


def _resolve_secret_path(configured: str, default_relative_path: str) -> Path:
    if configured:
        candidate = Path(configured).expanduser()
        if candidate.is_absolute():
            return candidate
        return canonical_secrets_root() / candidate
    return canonical_secrets_root() / default_relative_path


def oauth2_credentials_candidates() -> list[Path]:
    configured = os.getenv("MCPT_OAUTH2_CREDENTIALS_FILE", "").strip()
    candidates: list[Path] = []
    if configured:
        candidates.append(_resolve_secret_path(configured, "oauth/oauth2_credentials.json"))
    candidates.append(canonical_secrets_root() / "oauth" / "oauth2_credentials.json")
    return candidates


def oauth2_token_candidates() -> list[Path]:
    configured = os.getenv("MCPT_OAUTH2_TOKEN_FILE", "").strip()
    candidates: list[Path] = []
    if configured:
        candidates.append(_resolve_secret_path(configured, "oauth/oauth2_token.pickle"))
    candidates.append(canonical_secrets_root() / "oauth" / "oauth2_token.pickle")
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


def load_runtime_env(*, include_tracking_service_env: bool = True, override: bool = True) -> list[Path]:
    """Load secrets/shared.env vào os.environ. Gọi ở đầu config.py của mỗi service."""
    loaded: list[Path] = []
    env_file = shared_env_file()
    if env_file.exists():
        load_dotenv(env_file, override=override)
        loaded.append(env_file)

    secrets_root = canonical_secrets_root()
    os.environ["MCPT_SECRETS_ROOT"] = str(secrets_root)
    os.environ["MCPT_SHARED_ENV_FILE"] = str(env_file)
    os.environ["MCPT_OAUTH2_CREDENTIALS_FILE"] = str(resolve_oauth2_credentials_path())
    os.environ["MCPT_OAUTH2_TOKEN_FILE"] = str(resolve_oauth2_token_path())
    return loaded


def build_google_drive_oauth_credentials():
    import google.auth.transport.requests

    token_path = resolve_oauth2_token_path()
    if not token_path.exists():
        raise FileNotFoundError(
            f"OAuth2 token not found: {token_path}. "
            "Run OAuth flow locally to generate token, then copy secrets/ folder here."
        )
    with token_path.open("rb") as handle:
        credentials = pickle.load(handle)
    if getattr(credentials, "expired", False):
        credentials.refresh(google.auth.transport.requests.Request())
        try:
            with token_path.open("wb") as handle:
                pickle.dump(credentials, handle)
        except OSError:
            pass  # Read-only volume (Docker :ro mount) — token refreshed in memory only
    return credentials


def build_google_drive_oauth_service():
    from googleapiclient.discovery import build

    credentials = build_google_drive_oauth_credentials()
    return build("drive", "v3", credentials=credentials, cache_discovery=False)
