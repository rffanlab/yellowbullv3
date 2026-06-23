"""Tool registry for centralized tool management."""

from typing import Any, TypeVar

from tools.base import BaseTool, ToolInfo, ToolResult

T = TypeVar("T", bound=BaseTool)


class ToolRegistry:
    _tools: dict[str, BaseTool] = {}

    @classmethod
    def register(cls, tool: BaseTool):
        cls._tools[tool.info.name] = tool

    @classmethod
    def unregister(cls, name: str) -> bool:
        return cls._tools.pop(name, None) is not None

    @classmethod
    def get(cls, name: str) -> BaseTool | None:
        return cls._tools.get(name)

    @classmethod
    async def resolve(cls, name: str, **kwargs: Any) -> ToolResult:
        """Resolve and execute a tool by name with validation.

        Validates parameters before execution. Returns error result if not found.
        """
        tool = cls._tools.get(name)
        if tool is None:
            return ToolResult(content=f"Unknown tool: {name}", success=False)
        return await tool.execute_with_retry(**kwargs)

    @classmethod
    def list_all(cls) -> list[BaseTool]:
        """Return all registered tools."""
        return list(cls._tools.values())

    @classmethod
    def to_function_definitions(cls) -> list[dict[str, Any]]:
        """Convert to LLM function calling format."""
        return [
            {
                "name": t.info.name,
                "description": t.info.description,
                "parameters": t.info.parameters,
            }
            for t in cls._tools.values()
        ]

    @classmethod
    def clear(cls):
        """Clear all registered tools (mainly for testing)."""
        cls._tools.clear()


def register_tool(name: str, description: str, parameters: dict[str, Any]):
    """Decorator to auto-register a tool class."""

    def decorator(cls: type[T]) -> type[T]:
        instance = cls()
        instance._info = ToolInfo(name=name, description=description, parameters=parameters)  # type: ignore[attr-defined]
        ToolRegistry.register(instance)
        return cls

    return decorator
