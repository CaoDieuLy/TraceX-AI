from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "mcpt-tracking-service"
    legacy_root: str = "/workspace/backend/legacy-engine"
    tracking_use_mock: bool = True

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
