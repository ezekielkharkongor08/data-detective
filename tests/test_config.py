"""Tests for .env-based configuration (M4)."""

from src.config import Settings


def test_settings_defaults(monkeypatch):
    """Defaults are used when no env vars are set."""
    for key in [
        "LLM_PROVIDER",
        "LLM_MODEL",
        "OLLAMA_BASE_URL",
        "LLM_TEMPERATURE",
        "LLM_MAX_TOKENS",
    ]:
        monkeypatch.delenv(key, raising=False)

    settings = Settings(_env_file=None)

    assert settings.llm_provider == "ollama"
    assert settings.llm_model == "llama3.1:8b"
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.llm_temperature == 0.0
    assert settings.llm_max_tokens is None


def test_settings_env_override(monkeypatch):
    """Env vars override defaults (this is how .env values arrive)."""
    monkeypatch.setenv("LLM_MODEL", "qwen2.5-coder:7b")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.3")

    settings = Settings(_env_file=None)

    assert settings.llm_model == "qwen2.5-coder:7b"
    assert settings.llm_temperature == 0.3
