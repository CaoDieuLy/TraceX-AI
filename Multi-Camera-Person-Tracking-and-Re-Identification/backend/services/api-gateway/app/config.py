import os
import sys
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_HERE = Path(__file__).resolve()


def _detect_a20_root() -> Path | None:
    env_root = os.getenv("A20_ROOT", "").strip()
    candidates = []
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
    app_name: str = "mcpt-api-gateway"
    metadata_service_url: str = "http://metadata-service:8000"
    tracking_service_url: str = "http://tracking-service:8000"
    cors_allowed_origins: str = "http://localhost:3000"

    model_config = SettingsConfigDict(extra="ignore")

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_allowed_origins.split(",") if origin.strip()]


settings = Settings()
