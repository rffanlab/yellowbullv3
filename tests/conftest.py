"""Pytest configuration and shared fixtures."""

import sys

import pytest


@pytest.fixture(autouse=True)
def init_tools():
    """Clear registry then re-register built-in tools before each test."""
    from tools.registry import ToolRegistry
    ToolRegistry.clear()

    # Force re-import of ALL tools modules to trigger @register_tool decorators
    for name in list(sys.modules):
        if name.startswith("tools."):
            del sys.modules[name]

    # Import builtins package - this triggers all submodule imports and registration
    import tools.builtins  # noqa: F401
    yield
