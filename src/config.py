"""Application settings loaded from environment / .env (M4).

All model and LLM configuration lives in `.env`, never hardcoded in code
(AGENTS.md rule). Copy `.env.example` to `.env` and adjust the values.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration read from `.env` or environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM provider:
    llm_provider: str = "ollama"
    # Model name served by the provider 
    llm_model: str = "llama3.1:8b"
    # Base URL of the Ollama server.
    ollama_base_url: str = "http://localhost:11434"
    # Sampling temperature (0.0 = deterministic).
    llm_temperature: float = 0.0
    # Optional max tokens for the model response.
    llm_max_tokens: int | None = None


@lru_cache
def get_settings() -> Settings:
    """Return cached settings (read once from environment / .env)."""
    return Settings()


__all__ = ["Settings", "get_settings"]
