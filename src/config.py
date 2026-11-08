"""Application settings loaded from environment / .env (M4).

All configuration lives in `.env` (never hardcoded). Copy `.env.example`
to `.env` and adjust the values.
"""

from functools import lru_cache
from pydantic import model_validator

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration read from `.env` or environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_provider: str = "ollama"
    llm_model: str = "llama3.1:8b"
    ollama_base_url: str = "http://localhost:11434"
    # 0.0 = deterministic.
    llm_temperature: float = 0.0
    llm_max_tokens: int | None = None

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "postgres"
    postgres_password: str = ""
    postgres_db: str = "data_detective"

    # SQLite file used when DATABASE_URL is unset.
    database_path: str = "./data_detective.db"

    # Built from the postgres_* fields above when not set directly.
    database_url: str | None = None

    @model_validator(mode="after")
    def _build_database_url(self) -> "Settings":
        """Build DATABASE_URL from individual postgres_* fields when not given.

        When ``DATABASE_URL`` is unset we fall back to a local SQLite file at
        ``database_path`` so the project runs without PostgreSQL.
        """
        if self.database_url:
            return self
        if self.database_path:
            self.database_url = f"sqlite:///{self.database_path}"
        else:
            self.database_url = (
                f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
                f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
            )
        return self

    @property
    def backend_name(self) -> str:
        """Return ``"postgres"`` or ``"sqlite"`` based on the resolved URL."""
        return "sqlite" if self.database_url.startswith("sqlite") else "postgres"

    @property
    def is_sqlite(self) -> bool:
        return self.backend_name == "sqlite"

    @property
    def is_postgres(self) -> bool:
        return self.backend_name == "postgres"

    @property
    def postgres_conn_string(self) -> str:
        """Return a psycopg-native connection string (no SQLAlchemy dialect).

        PostgresSaver expects ``postgresql://``, not the SQLAlchemy form.
        """
        if self.database_url:
            return self.database_url.replace("postgresql+psycopg://", "postgresql://")
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    """Return cached settings (read once from environment / .env)."""
    return Settings()


__all__ = ["Settings", "get_settings"]
