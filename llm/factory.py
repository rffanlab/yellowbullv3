"""LLM provider factory."""

from llm.base import BaseLLM


def create_llm(provider: str, config: dict) -> BaseLLM:
    """Create an LLM instance by provider name.

    Args:
        provider: Provider identifier ('openai', 'anthropic')
        config: Provider-specific configuration dict

    Returns:
        Configured LLM instance

    Raises:
        ValueError: If provider is not supported
    """
    kwargs = {
        "api_key": config.get("api_key", ""),
        "base_url": config.get("base_url"),
        "model": config.get("model", "gpt-4o-mini"),
        "temperature": config.get("temperature", 0.7),
        "max_tokens": config.get("max_tokens", 4096),
        "timeout": config.get("timeout", 60.0),
    }

    if provider == "openai":
        from llm.openai_provider import OpenAILLM

        return OpenAILLM(**kwargs)
    elif provider == "anthropic":
        from llm.anthropic_provider import AnthropicLLM

        return AnthropicLLM(**kwargs)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
