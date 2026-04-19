import sys
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_HERE = Path(__file__).resolve()
A20_ROOT = _HERE.parents[5]
if str(A20_ROOT) not in sys.path:
    sys.path.insert(0, str(A20_ROOT))

try:
    from shared_secret_runtime import load_runtime_env  # noqa: E402

    load_runtime_env(include_tracking_service_env=True, override=True)
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
