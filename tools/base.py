"""Base tool abstraction."""

from __future__ import annotations

import abc
import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar


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
    error: str = ""


# ── Dependency injection support ───────────────────────────────────────


class _UnsetSentinel:
    """Marker for unresolved dependency."""

    pass


_UNSET = _UnsetSentinel()


@dataclass
class ToolContext:
    """Execution context passed to tools via dependency injection.

    Holds runtime dependencies like session, config, etc.
    """

    session_id: str | None = None
    user_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


# ── Base Tool ──────────────────────────────────────────────────────────


class BaseTool(abc.ABC):
    """Abstract base class for all tools."""

    _info: ToolInfo | None = None
    max_retries: int = 0

    @property
    @abc.abstractmethod
    def info(self) -> ToolInfo:
        """Tool description for generating function calling schema."""
        ...

    async def execute_with_retry(
        self, max_retries: int | None = None, **kwargs: Any
    ) -> ToolResult:
        """Execute with optional retry on failure.

        Validates parameters before execution and injects dependencies.
        """
        # Validate parameters first
        errors = self.validate(kwargs)
        if errors:
            return ToolResult(
                content=f"Parameter validation failed: {'; '.join(errors)}",
                success=False,
            )

        limit = max_retries if max_retries is not None else self.max_retries
        result = await self._execute_with_context(**kwargs)
        for _ in range(limit):
            if result.success:
                break
            result = await self._execute_with_context(**kwargs)
        return result

    async def _execute_with_context(self, **kwargs: Any) -> ToolResult:
        """Execute with dependency injection.

        If the execute() method accepts a 'context' parameter, it will be
        populated with a ToolContext instance. Override this method to add
        custom DI logic.
        """
        sig = inspect.signature(self.execute)
        params = list(sig.parameters.keys())

        if "context" in params:
            kwargs["context"] = ToolContext()

        return await self.execute(**kwargs)

    @abc.abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute business logic."""
        ...

    def validate(self, params: dict[str, Any]) -> list[str]:
        """Parameter validation. Returns error list (empty means pass).

        Default implementation validates against JSON Schema 'required' field
        and type hints from the parameters definition. Override for custom logic.
        """
        errors = []
        schema_props = self.info.parameters.get("properties", {}) if self.info else {}
        required = set(self.info.parameters.get("required", []) if self.info else [])

        # Check required fields
        for req in required:
            if req not in params or params[req] is None:
                errors.append(f"Missing required parameter: {req}")

        # Basic type checking against schema
        for name, value in params.items():
            prop_schema = schema_props.get(name)
            if prop_schema and isinstance(prop_schema, dict):
                expected_type = prop_schema.get("type")
                if expected_type and not _check_json_schema_type(value, expected_type):
                    errors.append(
                        f"Parameter '{name}' expected type {expected_type}, "
                        f"got {type(value).__name__}"
                    )

        return errors


def _check_json_schema_type(value: Any, schema_type: str) -> bool:
    """Check if a value matches a JSON Schema type."""
    type_map = {
        "string": str,
        "number": (int, float),
        "integer": int,
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    expected = type_map.get(schema_type)
    if expected is None:
        return True  # Unknown type, skip check
    return isinstance(value, expected)


# ── Dependency injection decorator ─────────────────────────────────────


T = TypeVar("T", bound=BaseTool)


def inject(
    name: str,
    description: str,
    parameters: dict[str, Any],
    deps: dict[str, Callable[[], Any]] | None = None,
) -> Callable[[type[T]], type[T]]:
    """Decorator to register a tool with dependency injection support.

    Args:
        name: Tool name for LLM function calling.
        description: Human-readable description shown to the model.
        parameters: JSON Schema parameters object.
        deps: Optional dict mapping attribute names to factory callables
               that produce the dependency value at registration time.

    Example:
        @inject("search", "Search the web", {...}, deps={"config": lambda: settings})
        class SearchTool(BaseTool): ...
    """

    def decorator(cls: type[T]) -> type[T]:
        instance = cls()
        instance._info = ToolInfo(name=name, description=description, parameters=parameters)  # type: ignore[attr-defined]

        # Inject dependencies
        if deps:
            for attr_name, factory in deps.items():
                setattr(instance, attr_name, factory())

        from tools.registry import ToolRegistry

        ToolRegistry.register(instance)
        return cls

    return decorator
