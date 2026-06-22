"""Configuration management with YAML loading, Pydantic validation, and env var resolution."""

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict


def _resolve_env_vars(value: Any) -> Any:
    """Recursively resolve ${ENV_VAR} patterns in config values."""
    if isinstance(value, str):
        import re

        def _replace(match: re.Match[str]) -> str:
            env_key = match.group(1)
            return os.environ.get(env_key, match.group(0))

        return re.sub(r"\$\{(\w+(?:_\w+)*)\}", _replace, value)
    if isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env_vars(item) for item in value]
    return value


class LLMProviderConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    api_key: str = ""
    base_url: str | None = None
    model: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: float = 60.0


class LLMConfig(BaseModel):
    active: str = "openai"
    providers: dict[str, LLMProviderConfig] = {}


class AgentConfig(BaseModel):
    system_prompt: str = (
        "You are a helpful AI assistant. You can use tools to help answer questions. "
        "Think step by step and provide clear, accurate responses."
    )
    max_chain_depth: int = 5
    context_window: int = 48
    tool_retry_limit: int = 2
    max_tool_calls_per_turn: int = 5


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False


class Settings(BaseModel):
    llm: LLMConfig = LLMConfig()
    agent: AgentConfig = AgentConfig()
    server: ServerConfig = ServerConfig()


def load_settings(config_path: str | None = None) -> Settings:
    """Load settings from YAML config file, with env var resolution."""
    if config_path is None:
        config_path = os.environ.get("AGENT_CONFIG", "config/default.yaml")

    path = Path(config_path)
    raw_config: dict[str, Any] = {}

    if path.exists():
        with open(path) as f:
            raw_config = yaml.safe_load(f) or {}

    # Resolve environment variables
    raw_config = _resolve_env_vars(raw_config)

    return Settings(**raw_config)
