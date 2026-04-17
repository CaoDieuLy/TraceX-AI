from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "mcpt-tracking-service"
    legacy_root: str = "/workspace/backend/legacy-engine"
    tracking_use_mock: bool = True
    lightning_api_base_url: str = ""
    lightning_api_endpoint: str = "/predict"
    lightning_api_token: str = ""
    lightning_api_auth_header: str = "Authorization"
    lightning_api_auth_prefix: str = "Bearer "
    lightning_timeout_seconds: int = 180

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
