"""LLM provider factory."""

from llm.base import BaseLLM


def _resolve_config(config: dict) -> dict:
    """Merge user config with defaults from LLMProviderConfig (single source of truth)."""
    from config.settings import LLMProviderConfig

    # Defaults from Pydantic model — the only place defaults live
    defaults = LLMProviderConfig().model_dump()
    # User config overrides; skip None so explicit empty strings still work
    merged = {**defaults, **{k: v for k, v in config.items() if v is not None}}
    return merged


def create_llm(provider: str, config: dict) -> BaseLLM:
    """Create an LLM instance by provider name.

    Args:
        provider: Provider identifier ('openai', 'anthropic')
        config: Provider-specific configuration dict or Pydantic model

    Returns:
        Configured LLM instance

    Raises:
        ValueError: If provider is not supported
    """
    # Handle both dict and Pydantic models
    if hasattr(config, "model_dump"):
        config = config.model_dump()

    kwargs = _resolve_config(config)

    if provider == "openai":
        from llm.openai_provider import OpenAILLM

        return OpenAILLM(**kwargs)
    elif provider == "anthropic":
        from llm.anthropic_provider import AnthropicLLM

        return AnthropicLLM(**kwargs)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
