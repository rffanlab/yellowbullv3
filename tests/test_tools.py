"""Tests for tool system."""

import pytest


async def test_current_time_tool():
    """Test current time tool returns valid datetime string."""
    # Import to trigger registration
    from tools.builtins.current_time import CurrentTimeTool  # noqa: F401

    from tools.registry import ToolRegistry

    tool = ToolRegistry.get("current_time")
    assert tool is not None

    result = await tool.execute()
    assert result.success
    assert len(result.content) > 0


async def test_current_time_with_timezone():
    from tools.builtins.current_time import CurrentTimeTool  # noqa: F401
    from tools.registry import ToolRegistry

    tool = ToolRegistry.get("current_time")
    result = await tool.execute(timezone="Asia/Shanghai")
    assert result.success


async def test_calculator_basic():
    """Test calculator with basic arithmetic."""
    from tools.builtins.calculator import CalculatorTool  # noqa: F401
    from tools.registry import ToolRegistry

    tool = ToolRegistry.get("calculator")
    assert tool is not None

    result = await tool.execute(expression="2 + 3 * 4")
    assert result.success
    assert "14" in result.content


async def test_calculator_functions():
    """Test calculator with math functions."""
    from tools.builtins.calculator import CalculatorTool  # noqa: F401
    from tools.registry import ToolRegistry

    tool = ToolRegistry.get("calculator")
    result = await tool.execute(expression="sqrt(16)")
    assert result.success
    assert "4.0" in result.content


async def test_calculator_invalid():
    """Test calculator with invalid expression."""
    from tools.builtins.calculator import CalculatorTool  # noqa: F401
    from tools.registry import ToolRegistry

    tool = ToolRegistry.get("calculator")
    result = await tool.execute(expression="import os")
    assert not result.success


async def test_web_search_tool_registered():
    """Test web search tool is registered."""
    from tools.builtins.web_search import WebSearchTool  # noqa: F401
    from tools.registry import ToolRegistry

    tool = ToolRegistry.get("web_search")
    assert tool is not None
    assert "search" in tool.info.description.lower()


async def test_tool_registry():
    """Test registry operations."""
    from tools.base import BaseTool, ToolInfo, ToolResult
    from tools.registry import ToolRegistry

    class DummyTool(BaseTool):
        @property
        def info(self) -> ToolInfo:
            return ToolInfo(
                name="dummy",
                description="A dummy tool",
                parameters={"type": "object", "properties": {}},
            )

        async def execute(self, **kwargs) -> ToolResult:
            return ToolResult(content="ok")

    t = DummyTool()
    ToolRegistry.register(t)

    assert len(ToolRegistry.list_all()) >= 1
    assert ToolRegistry.get("dummy") is not None

    defs = ToolRegistry.to_function_definitions()
    names = [d["name"] for d in defs]
    assert "dummy" in names


async def test_tool_retry_on_failure():
    """Test that execute_with_retry retries failed tool calls."""
    from tools.base import BaseTool, ToolInfo, ToolResult
    from tools.registry import ToolRegistry

    call_count = 0

    class FlakyTool(BaseTool):
        @property
        def info(self) -> ToolInfo:
            return ToolInfo(
                name="flaky",
                description="Fails then succeeds",
                parameters={"type": "object", "properties": {}},
            )

        async def execute(self, **kwargs) -> ToolResult:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                return ToolResult(content="fail", success=False)
            return ToolResult(content="success", success=True)

    ToolRegistry.register(FlakyTool())
    tool = ToolRegistry.get("flaky")

    result = await tool.execute_with_retry(max_retries=1)
    assert result.success
    assert call_count == 2
