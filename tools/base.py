"""Base tool abstraction."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolInfo:
    """Tool metadata for function calling schema generation."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class ToolResult:
    """Standardized tool execution result."""

    content: str = ""
    success: bool = True


class BaseTool(ABC):
    """Abstract base class for all tools."""

    _info: ToolInfo = field(default=None)  # type: ignore[misc]

    @property
    @abstractmethod
    def info(self) -> ToolInfo:
        """Tool description for generating function calling schema."""
        ...

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute business logic."""
        ...

    def validate(self, params: dict[str, Any]) -> list[str]:
        """Parameter validation. Returns error list (empty means pass)."""
        return []
