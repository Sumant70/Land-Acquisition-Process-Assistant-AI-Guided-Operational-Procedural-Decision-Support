from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Land Acquisition Delay Decision Support System"
    secret_key: str = "change-this-to-a-long-random-string"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 480
    database_url: str = "sqlite:///./land_acquisition.db"
    default_admin_username: str = "admin"
    default_admin_password: str = "admin123"
    default_officer_username: str = "officer"
    default_officer_password: str = "officer123"
    api_host: str = "127.0.0.1"
    api_port: int = 8000


settings = Settings()
PROJECT_ROOT = Path(__file__).resolve().parent.parent
