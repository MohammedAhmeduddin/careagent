"""
careagent.config
~~~~~~~~~~~~~~~~
Centralised application settings loaded from environment variables.
"""

from functools import lru_cache
from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        # Ignore variables this version doesn't use (e.g. MLFLOW_TRACKING_URI in an
        # older .env or Cloud Run config) instead of refusing to start.
        extra="ignore",
    )

    # App
    app_env: str = Field(default="development")
    log_level: str = Field(default="INFO")

    # Database
    postgres_user: str = Field(default="careagent")
    postgres_password: str = Field(default="careagent_dev")
    postgres_db: str = Field(default="careagent_db")
    postgres_host: str = Field(default="localhost")
    postgres_port: int = Field(default=5432)
    # "require" for Neon; CI's local Postgres has no TLS, so it sets "disable".
    postgres_sslmode: str = Field(default="require")
    # A full connection string (e.g. from Secret Manager) overrides POSTGRES_*.
    database_url_env: str = Field(default="", validation_alias="DATABASE_URL")

    @computed_field
    @property
    def database_url(self) -> str:
        if self.database_url_env:
            url = self.database_url_env.strip()
            for prefix in ("postgres://", "postgresql://"):
                if url.startswith(prefix):
                    return "postgresql+psycopg2://" + url[len(prefix):]
            return url
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
            f"?sslmode={self.postgres_sslmode}"
        )

    # OpenAI
    openai_api_key: str = Field(default="")
    # gpt-4o-mini shuts down on 2027-01-14; OpenAI recommends gpt-6-luna.
    openai_model: str = Field(default="gpt-6-luna")
    openai_temperature: float = Field(default=0.1)
    # Reasoning models count reasoning tokens against this limit.
    openai_max_completion_tokens: int = Field(default=4000)
    openai_timeout_seconds: float = Field(default=60.0)
    openai_max_retries: int = Field(default=1)

    # LangSmith
    langchain_tracing_v2: bool = Field(default=False)
    langchain_api_key: str = Field(default="")
    langchain_project: str = Field(default="careagent")

    # Data
    cms_data_path: str = Field(default="data/raw/cms_providers_NJ_2024.csv")

    # Data validation: below this completeness the provider is not compared
    min_data_completeness: float = Field(default=1.0)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
