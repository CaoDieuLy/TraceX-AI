"""Cấu hình ai_service (tracking URL + token dùng khi proxy)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_HERE = Path(__file__).resolve()


def _detect_a20_root() -> Path | None:
    env_root = os.getenv("A20_ROOT", "").strip()
    candidates: list[Path] = []
    if env_root:
        candidates.append(Path(env_root).expanduser())
    candidates.extend(
        [
            Path("/workspace/a20-root"),
            _HERE.parent,
            *_HERE.parents,
        ]
    )
    for candidate in candidates:
        if (candidate / "shared_secret_runtime.py").exists():
            return candidate
    return None


A20_ROOT = _detect_a20_root()
if A20_ROOT and str(A20_ROOT) not in sys.path:
    sys.path.insert(0, str(A20_ROOT))

try:
    from shared_secret_runtime import load_runtime_env  # noqa: E402

    load_runtime_env(include_tracking_service_env=True, override=False)
except ImportError:
    pass


class Settings(BaseSettings):
    tracking_service_url: str = "http://tracking-service:8000"
    lightning_api_token: str = ""
    lightning_api_auth_header: str = "Authorization"
    lightning_api_auth_prefix: str = "Bearer "
    tracking_proxy_timeout_seconds: float = 180.0
    tracking_proxy_connect_timeout_seconds: float = 10.0
    tracking_proxy_pool_timeout_seconds: float = 10.0
    tracking_proxy_max_connections: int = 100
    tracking_proxy_max_keepalive_connections: int = 20
    tracking_proxy_keepalive_expiry_seconds: float = 30.0

    model_config = SettingsConfigDict(extra="ignore")


settings = Settings()
