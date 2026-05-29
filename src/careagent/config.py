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

    @computed_field
    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # OpenAI
    openai_api_key: str = Field(default="")
    openai_model: str = Field(default="gpt-4o-mini")

    # LangSmith
    langchain_tracing_v2: bool = Field(default=False)
    langchain_api_key: str = Field(default="")
    langchain_project: str = Field(default="careagent")

    # MLflow
    mlflow_tracking_uri: str = Field(default="http://localhost:5001")

    # Data
    cms_data_path: str = Field(default="data/cms_provider_2022.csv")

    # Agent thresholds
    data_quality_threshold: float = Field(default=0.15)
    anomaly_contamination: float = Field(default=0.03)
    quality_score_review_threshold: float = Field(default=65.0)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
