"""Tests for configuration system."""

import os
from unittest.mock import patch

import yaml


def test_load_default_settings():
    from config.settings import load_settings, Settings

    settings = load_settings("config/default.yaml")
    assert isinstance(settings, Settings)
    assert settings.llm.active == "openai"
    assert settings.agent.max_chain_depth == 5
    assert settings.server.port == 8000


def test_env_var_resolution():
    from config.settings import _resolve_env_vars

    with patch.dict(os.environ, {"MY_KEY": "secret123"}):
        result = _resolve_env_vars("${MY_KEY}")
        assert result == "secret123"

    # Nested resolution
    data = {"api_key": "${TEST_VAR}", "nested": {"val": "${ANOTHER}"}}
    with patch.dict(os.environ, {"TEST_VAR": "a", "ANOTHER": "b"}):
        resolved = _resolve_env_vars(data)
        assert resolved["api_key"] == "a"
        assert resolved["nested"]["val"] == "b"


def test_missing_config_returns_defaults():
    from config.settings import Settings, load_settings

    settings = load_settings("nonexistent.yaml")
    assert isinstance(settings, Settings)
    # Should have default values
    assert settings.server.host == "0.0.0.0"


def test_custom_system_prompt():
    """Test loading custom system prompt from YAML."""
    config_content = {
        "agent": {"system_prompt": "Custom prompt here"},
    }
    import tempfile

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as f:
        yaml.dump(config_content, f)
        path = f.name

    from config.settings import load_settings

    settings = load_settings(path)
    assert settings.agent.system_prompt == "Custom prompt here"
