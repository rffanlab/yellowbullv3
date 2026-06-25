"""Tool execution engine — timeout, retry, parallel scheduling, and caching.

Responsibilities:
  • Per-tool timeout enforcement (asyncio.wait_for)
  • Configurable retry with exponential backoff
  • Parallel / serial batch execution modes
  • Idempotent result caching (simple in-memory dict)
  • Execution audit logging

Design doc: docs/modules/05-tool-executor.md
Acceptance criteria: docs/acceptance-criteria/05-tool-executor.md
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from core.logging_setup import get_logger
from tools.base import ToolResult
from tools.registry import ToolRegistry

logger = get_logger(__name__)


# ── Enums ────────────────────────────────────────────────────────────────


class ExecutionStatus(Enum):
    """Terminal states of a tool execution."""

    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    RETRY_EXHAUSTED = "retry_exhausted"


# ── Data models ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ToolCallRequest:
    """Single tool-call request from the LLM."""

    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    @property
    def cache_key(self) -> str:
        """Generate cache key: tool_name + sorted args (order-independent)."""
        raw = f"{self.tool_name}:{json.dumps(self.arguments, sort_keys=True)}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class ExecutionResult:
    """Single tool-call result with metadata."""

    tool_name: str
    status: ExecutionStatus
    content: str = ""       # Output on success; empty on failure
    error: str = ""         # Error message on failure; empty on success
    duration_ms: float = 0.0   # Execution time in milliseconds
    retry_count: int = 0       # Actual number of retries performed

    @property
    def success(self) -> bool:
        return self.status == ExecutionStatus.SUCCESS


@dataclass(frozen=True)
class BatchExecutionResult:
    """Aggregated result of a batch tool execution."""

    results: list[ExecutionResult] = field(default_factory=list)
    total_duration_ms: float = 0.0

    @property
    def all_success(self) -> bool:
        return all(r.success for r in self.results) if self.results else True

    @property
    def any_failed(self) -> bool:
        return not self.all_success and len(self.results) > 0


# ── Tool Executor ──────────────────────────────────────────────────────


class ToolExecutor:
    """Core tool execution engine.

    Handles timeout enforcement, retry with exponential backoff, parallel/serial
    batch scheduling, and idempotent result caching.

    Usage:
        executor = ToolExecutor(
            registry=tool_registry,
            default_timeout=30.0,
            max_retries=2,
            parallel=True,
        )
        batch_result = await executor.execute_batch([request1, request2])
    """

    def __init__(
        self,
        *,
        registry: ToolRegistry,
        default_timeout: float = 30.0,
        max_retries: int = 2,
        retry_backoff_factor: float = 1.5,
        parallel: bool = True,
        enable_cache: bool = False,
    ):
        self._registry = registry
        self._default_timeout = default_timeout
        self._max_retries = max_retries
        self._backoff_factor = retry_backoff_factor
        self._parallel = parallel
        self._enable_cache = enable_cache

        # Simple in-memory cache: cache_key -> ExecutionResult
        self._cache: dict[str, ExecutionResult] = {}

    # ── Public API ─────────────────────────────────────────────────────

    async def execute_batch(
        self,
        requests: list[ToolCallRequest],
    ) -> BatchExecutionResult:
        """Execute a batch of tool calls.

        Args:
            requests: Tool-call request list (all tool calls from one LLM turn).

        Returns:
            BatchExecutionResult with per-tool results and total duration.
        """
        start_time = time.monotonic()
        results: list[ExecutionResult] = []

        if self._parallel:
            # Parallel execution: all tools start simultaneously, each has
            # independent timeout / retry handling.
            tasks = [self._execute_single(req) for req in requests]
            raw_results = await asyncio.gather(*tasks, return_exceptions=True)

            resolved: list[ExecutionResult] = []
            for i, r in enumerate(raw_results):
                if isinstance(r, ExecutionResult):
                    resolved.append(r)
                else:
                    req = requests[i]
                    resolved.append(ExecutionResult(
                        tool_name=req.tool_name,
                        status=ExecutionStatus.FAILED,
                        error=f"Unexpected error: {r}",
                    ))
            results = resolved
        else:
            # Serial execution: execute one by one (for dependent tools).
            for req in requests:
                result = await self._execute_single(req)
                results.append(result)

        total_ms = (time.monotonic() - start_time) * 1000
        return BatchExecutionResult(results=results, total_duration_ms=total_ms)

    def clear_cache(self) -> None:
        """Clear the execution cache."""
        self._cache.clear()

    # ── Internal helpers ───────────────────────────────────────────────

    async def _execute_single(
        self,
        request: ToolCallRequest,
    ) -> ExecutionResult:
        """Execute a single tool call (with timeout + retry)."""

        # 1. Check cache
        if self._enable_cache and request.cache_key in self._cache:
            cached = self._cache[request.cache_key]
            logger.debug(f"Cache hit for {request.tool_name}({request.arguments})")
            return ExecutionResult(
                tool_name=request.tool_name,
                status=cached.status,
                content=cached.content,
                error=cached.error,
            )

        # 2. Resolve tool from registry
        tool = self._registry.get(request.tool_name)
        if tool is None:
            logger.warning(f"Tool not found: {request.tool_name}")
            return ExecutionResult(
                tool_name=request.tool_name,
                status=ExecutionStatus.FAILED,
                error=f"Unknown tool: {request.tool_name}",
            )

        # 3. Get effective timeout (tool-level > default)
        timeout = getattr(tool, "timeout", None) or self._default_timeout

        # 4. Retry loop
        last_error = ""
        for attempt in range(self._max_retries + 1):
            if attempt > 0:
                wait_time = self._backoff_factor ** attempt
                logger.info(
                    f"Retrying {request.tool_name} (attempt {attempt}/{self._max_retries}) "
                    f"after {wait_time:.1f}s"
                )
                await asyncio.sleep(wait_time)

            start_ms = time.monotonic()
            try:
                tool_result: ToolResult = await asyncio.wait_for(
                    self._invoke_tool(tool, request.arguments),
                    timeout=timeout,
                )
                duration_ms = (time.monotonic() - start_ms) * 1000

                result = ExecutionResult(
                    tool_name=request.tool_name,
                    status=(
                        ExecutionStatus.SUCCESS
                        if tool_result.success
                        else ExecutionStatus.FAILED
                    ),
                    content=tool_result.content if tool_result.success else "",
                    error="" if tool_result.success else (tool_result.error or "Unknown error"),
                    duration_ms=duration_ms,
                    retry_count=attempt,
                )

                # Cache successful results when enabled
                if result.success and self._enable_cache:
                    self._cache[request.cache_key] = result

                return result

            except asyncio.TimeoutError:
                last_error = f"Tool '{request.tool_name}' timed out after {timeout}s"
                logger.warning(last_error)
            except Exception as e:
                last_error = str(e)
                logger.error(
                    f"Tool '{request.tool_name}' error on attempt {attempt}: {e}"
                )

        # 5. All retries exhausted
        return ExecutionResult(
            tool_name=request.tool_name,
            status=(
                ExecutionStatus.RETRY_EXHAUSTED
                if self._max_retries > 0
                else ExecutionStatus.FAILED
            ),
            error=f"Failed after {self._max_retries} retries: {last_error}",
        )

    async def _invoke_tool(
        self,
        tool: Any,
        arguments: dict[str, Any],
    ) -> ToolResult:
        """Invoke the actual tool execution method."""
        if hasattr(tool, "execute"):
            return await tool.execute(**arguments)
        elif callable(tool):
            return await tool(**arguments)
        else:
            raise RuntimeError(f"Tool '{tool}' is not executable")
