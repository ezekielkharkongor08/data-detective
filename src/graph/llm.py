"""LLM factory.

Builds the chat model from settings in `.env`.
"""

from functools import lru_cache

from langchain_core.language_models import BaseChatModel

from src.config import get_settings


def get_llm() -> BaseChatModel:
    """
    Build the chat model configured in `.env`.
    """
    settings = get_settings()

    if settings.llm_provider == "ollama":
        from langchain_ollama import ChatOllama

        kwargs: dict[str, Any] = {
            "model": settings.llm_model,
            "base_url": settings.ollama_base_url,
            "temperature": settings.llm_temperature,
        }
        if settings.llm_max_tokens is not None:
            kwargs["num_predict"] = settings.llm_max_tokens
        return ChatOllama(**kwargs)

    raise ValueError(
        f"Unsupported LLM provider: {settings.llm_provider!r}. "
        "Set LLM_PROVIDER=ollama in .env."
    )


@lru_cache
def get_cached_llm() -> BaseChatModel:
    """Cached LLM instance (one client per process)."""
    return get_llm()


__all__ = ["get_llm", "get_cached_llm"]
