"""Configuration management with YAML loading, Pydantic validation, and env var resolution."""

import os
from collections.abc import Callable
from pathlib import Path
from threading import Event, Thread
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

    # ── Schema validation ────────────────────────────────────────────

    def validate(self, schema: dict[str, Any]) -> list[str]:
        """Validate settings against a JSON-like schema. Returns error list."""
        errors: list[str] = []
        props = schema.get("properties", {})
        reqs = schema.get("required", [])
        self._validate_node("", self.model_dump(), props, reqs, errors)
        return errors

    def _validate_node(
        self,
        prefix: str,
        data: dict[str, Any],
        properties: dict[str, Any],
        required: list[str],
        errors: list[str],
    ):
        """Recursive schema validation helper."""
        for req in required:
            if req not in data:
                errors.append(f"Missing required field: {prefix}{req}")

        for key, spec in properties.items():
            value = data.get(key)
            if value is None and key not in data:
                continue
            expected = spec.get("type")
            if expected == "object" and isinstance(value, dict):
                self._validate_node(
                    f"{prefix}{key}.",
                    value,
                    spec.get("properties", {}),
                    spec.get("required", []),
                    errors,
                )
            elif expected == "string" and not isinstance(value, str):
                errors.append(f"Invalid type for {prefix}{key}: expected string")
            elif expected == "number" and not isinstance(value, (int, float)):
                errors.append(f"Invalid type for {prefix}{key}: expected number")

    # ── Hot reload support ───────────────────────────────────────────

    def watch(
        self,
        config_path: str = "config/default.yaml",
        callback: Callable[[str], None] | None = None,
    ) -> "WatchHandle":
        """Start watching config file for changes. Returns a handle to stop."""
        handle = WatchHandle(callback=callback)
        watcher = _ConfigWatcher(config_path, self.__class__, callback, handle)
        watcher.start()
        return handle


class WatchHandle:
    """Handle to stop config file watching."""

    def __init__(self, callback: Callable[[str], None] | None = None):
        self._stop_event: Event | None = None
        self.callback = callback

    def stop(self):
        if self._stop_event is not None:
            self._stop_event.set()


class _ConfigWatcher(Thread):
    """Background thread that polls config file for changes."""

    def __init__(
        self,
        path: str,
        settings_cls: type,
        callback: Callable[[str], None] | None,
        handle: WatchHandle,
    ):
        super().__init__(daemon=True)
        self.path = path
        self.settings_cls = settings_cls
        self.callback = callback
        self._hash: str = ""

        if os.path.exists(path):
            import hashlib

            self._hash = hashlib.md5(Path(path).read_bytes()).hexdigest()

        handle._stop_event = Event()

    def run(self):
        import hashlib
        import time

        while not self._stop_event.is_set():
            time.sleep(2)
            if not os.path.exists(self.path):
                continue
            new_hash = hashlib.md5(Path(self.path).read_bytes()).hexdigest()
            if new_hash != self._hash:
                self._hash = new_hash
                try:
                    load_settings(self.path)
                    if self.callback:
                        self.callback(self.path)
                except Exception:
                    pass


# ── Module-level cached settings for hot reload ───────────────────────

_cached_settings: Settings | None = None
_cached_path: str = ""


def load_settings(config_path: str | None = None) -> Settings:
    """Load settings from YAML config file, with env var resolution."""
    global _cached_settings, _cached_path

    if config_path is None:
        config_path = os.environ.get("AGENT_CONFIG", "config/default.yaml")

    path = Path(config_path)
    raw_config: dict[str, Any] = {}

    try:
        if path.exists():
            with open(path, encoding="utf-8") as f:
                raw_config = yaml.safe_load(f) or {}
    except (FileNotFoundError, UnicodeDecodeError):
        pass

    # Resolve environment variables
    raw_config = _resolve_env_vars(raw_config)

    settings = Settings(**raw_config)
    _cached_settings = settings
    _cached_path = config_path
    return settings


def reload_settings(config_path: str | None = None) -> bool:
    """Public API to manually trigger config reload."""
    if config_path is None:
        config_path = _cached_path or os.environ.get("AGENT_CONFIG", "config/default.yaml")
    try:
        load_settings(config_path)
        return True
    except Exception:
        return False
