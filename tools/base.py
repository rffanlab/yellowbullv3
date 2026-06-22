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
    max_retries: int = 0

    @property
    @abstractmethod
    def info(self) -> ToolInfo:
        """Tool description for generating function calling schema."""
        ...

    async def execute_with_retry(
        self, max_retries: int | None = None, **kwargs: Any
    ) -> ToolResult:
        """Execute with optional retry on failure."""
        limit = max_retries if max_retries is not None else self.max_retries
        result = await self.execute(**kwargs)
        for _ in range(limit):
            if result.success:
                break
            result = await self.execute(**kwargs)
        return result

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute business logic."""
        ...

    def validate(self, params: dict[str, Any]) -> list[str]:
        """Parameter validation. Returns error list (empty means pass)."""
        return []
