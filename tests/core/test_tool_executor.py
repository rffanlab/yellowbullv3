"""Tests for ToolExecutor — TEX-01 through TEX-12."""

import asyncio
import time

from tools.base import BaseTool, ToolInfo, ToolResult
from tools.registry import ToolRegistry


# ── Mock tool factory (uses type() to avoid class-body scoping) ─────


def _create_mock_tool_class(
    name: str,
    *,
    return_result: ToolResult | None = None,
    side_effect: Exception | None = None,
    delay: float = 0.0,
    idempotent: bool = False,
    timeout: float | None = None,
):
    """Factory that returns a BaseTool subclass with configured behavior."""

    async def _execute(self, **kwargs) -> ToolResult:
        if delay > 0:
            await asyncio.sleep(delay)
        if side_effect is not None:
            raise side_effect
        return return_result or ToolResult(content="ok")

    def _info_getter(self) -> ToolInfo:
        return ToolInfo(
            name=name,
            description=f"Mock {name}",
            parameters={"type": "object", "properties": {}},
        )

    cls = type(
        f"_MockTool_{name}",
        (BaseTool,),
        {"execute": _execute, "info": property(_info_getter)},
    )
    cls.idempotent = idempotent
    if timeout is not None:
        cls.timeout = timeout
    return cls


def _register(name: str, **kwargs) -> BaseTool:
    """Create and register a mock tool. Must be called inside test body."""
    cls = _create_mock_tool_class(name, **kwargs)
    tool = cls()
    ToolRegistry.register(tool)
    return tool


# ── TEX-01: Execute registered tool → success result ────────────────


async def test_tex_01_execute_registered_tool():
    """TEX-01: ToolExecutor correctly executes a registered tool."""
    from core.tool_executor import (
        ExecutionStatus,
        ToolCallRequest,
        ToolExecutor,
    )

    _register("mock_ok", return_result=ToolResult(content="hello"))

    executor = ToolExecutor(registry=ToolRegistry())
    batch = await executor.execute_batch([ToolCallRequest(tool_name="mock_ok")])

    assert len(batch.results) == 1
    result = batch.results[0]
    assert result.status == ExecutionStatus.SUCCESS
    assert result.content == "hello"
    assert result.tool_name == "mock_ok"
    assert result.duration_ms >= 0


# ── TEX-02: Timeout tool → timeout error ────────────────────────────


async def test_tex_02_timeout():
    """TEX-02: Slow tool is interrupted after timeout, not blocking others."""
    from core.tool_executor import (
        ExecutionStatus,
        ToolCallRequest,
        ToolExecutor,
    )

    _register("slow_tool", delay=5.0)
    _register("fast_tool", return_result=ToolResult(content="done"))

    executor = ToolExecutor(registry=ToolRegistry(), default_timeout=0.1)
    batch = await executor.execute_batch(
        [
            ToolCallRequest(tool_name="slow_tool"),
            ToolCallRequest(tool_name="fast_tool"),
        ],
    )

    assert len(batch.results) == 2
    slow_result = batch.results[0]
    fast_result = batch.results[1]

    # Slow tool should fail (timeout or retry exhausted)
    assert not slow_result.success
    assert "timed out" in slow_result.error.lower() or "retry_exhausted" in str(slow_result.status.value).lower()

    # Fast tool should succeed and NOT be blocked by the slow one
    assert fast_result.status == ExecutionStatus.SUCCESS
    assert fast_result.content == "done"


# ── TEX-03: Retry on retryable errors with backoff ─────────────────


async def test_tex_03_retry_on_failure():
    """TEX-03: Retry mechanism works for transient exceptions."""
    from core.tool_executor import (
        ExecutionStatus,
        ToolCallRequest,
        ToolExecutor,
    )

    call_count = 0

    class FlakyTool(BaseTool):
        @property
        def info(self) -> ToolInfo:
            return ToolInfo(name="flaky", description="", parameters={})

        async def execute(self, **kwargs) -> ToolResult:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("transient")
            return ToolResult(content="ok")

    ToolRegistry.register(FlakyTool())

    executor = ToolExecutor(
        registry=ToolRegistry(),
        max_retries=2,
        retry_backoff_factor=0.01,
    )
    batch = await executor.execute_batch([ToolCallRequest(tool_name="flaky")])

    result = batch.results[0]
    assert result.status == ExecutionStatus.SUCCESS
    assert call_count == 3


# ── TEX-04: Non-retryable error no retry (P0 simplified) ───────────


async def test_tex_04_non_retryable_error():
    """TEX-04: Errors that always fail exhaust retries.

    Note: P0 simplified — all exceptions are treated as retryable.
    Input validation filtering is a future enhancement.
    """
    from core.tool_executor import (
        ExecutionStatus,
        ToolCallRequest,
        ToolExecutor,
    )

    call_count = 0

    class BadTool(BaseTool):
        @property
        def info(self) -> ToolInfo:
            return ToolInfo(name="bad", description="", parameters={})

        async def execute(self, **kwargs) -> ToolResult:
            nonlocal call_count
            call_count += 1
            raise ValueError("permanent")

    ToolRegistry.register(BadTool())

    executor = ToolExecutor(
        registry=ToolRegistry(),
        max_retries=2,
        retry_backoff_factor=0.01,
    )
    batch = await executor.execute_batch([ToolCallRequest(tool_name="bad")])

    result = batch.results[0]
    assert not result.success
    # P0 simplified: all exceptions are retryable → retries exhausted
    assert result.status == ExecutionStatus.RETRY_EXHAUSTED
    assert call_count == 3  # initial + 2 retries


# ── TEX-05: Parallel batch execution ───────────────────────────────


async def test_tex_05_parallel_batch():
    """TEX-05: Multiple tools execute in parallel."""
    from core.tool_executor import (
        ExecutionStatus,
        ToolCallRequest,
        ToolExecutor,
    )

    _register("p1", return_result=ToolResult(content="a"))
    _register("p2", return_result=ToolResult(content="b"))

    executor = ToolExecutor(registry=ToolRegistry(), parallel=True)
    batch = await executor.execute_batch(
        [
            ToolCallRequest(tool_name="p1"),
            ToolCallRequest(tool_name="p2"),
        ],
    )

    assert len(batch.results) == 2
    contents = {r.content for r in batch.results}
    assert "a" in contents and "b" in contents


# ── TEX-06: Serial batch execution ────────────────────────────────


async def test_tex_06_serial_batch():
    """TEX-06: Tools execute sequentially in serial mode."""
    from core.tool_executor import (
        ExecutionStatus,
        ToolCallRequest,
        ToolExecutor,
    )

    _register("s1", return_result=ToolResult(content="x"))
    _register("s2", return_result=ToolResult(content="y"))

    executor = ToolExecutor(registry=ToolRegistry(), parallel=False)
    batch = await executor.execute_batch(
        [
            ToolCallRequest(tool_name="s1"),
            ToolCallRequest(tool_name="s2"),
        ],
    )

    assert len(batch.results) == 2
    contents = {r.content for r in batch.results}
    assert "x" in contents and "y" in contents


# ── TEX-07: Unknown tool → clear error message ─────────────────────


async def test_tex_07_unknown_tool():
    """TEX-07: Calling unregistered tool returns clear error."""
    from core.tool_executor import (
        ExecutionStatus,
        ToolCallRequest,
        ToolExecutor,
    )

    executor = ToolExecutor(registry=ToolRegistry())
    batch = await executor.execute_batch(
        [ToolCallRequest(tool_name="nonexistent")]
    )

    result = batch.results[0]
    assert not result.success
    assert "Unknown tool" in result.error or "nonexistent" in result.error


# ── TEX-08: Error isolation in parallel mode ───────────────────────


async def test_tex_08_error_isolation():
    """TEX-08: One failing tool does not affect others in parallel."""
    from core.tool_executor import (
        ExecutionStatus,
        ToolCallRequest,
        ToolExecutor,
    )

    _register("good", return_result=ToolResult(content="ok"))

    executor = ToolExecutor(registry=ToolRegistry(), max_retries=0)
    batch = await executor.execute_batch(
        [
            ToolCallRequest(tool_name="nonexistent"),  # will fail
            ToolCallRequest(tool_name="good"),          # should succeed
        ],
    )

    assert len(batch.results) == 2
    assert not batch.results[0].success
    assert batch.results[1].status == ExecutionStatus.SUCCESS


# ── TEX-09: Cache for idempotent tools ─────────────────────────────


async def test_tex_09_cache_hit():
    """TEX-09: Idempotent tool results are cached."""
    from core.tool_executor import (
        ExecutionStatus,
        ToolCallRequest,
        ToolExecutor,
    )

    call_count = 0

    class CachedTool(BaseTool):
        idempotent = True

        @property
        def info(self) -> ToolInfo:
            return ToolInfo(name="cached", description="", parameters={})

        async def execute(self, **kwargs) -> ToolResult:
            nonlocal call_count
            call_count += 1
            return ToolResult(content=f"v{call_count}")

    ToolRegistry.register(CachedTool())

    executor = ToolExecutor(
        registry=ToolRegistry(),
        enable_cache=True,
    )

    batch1 = await executor.execute_batch([ToolCallRequest(tool_name="cached")])
    assert call_count == 1

    batch2 = await executor.execute_batch([ToolCallRequest(tool_name="cached")])
    # Cache hit — tool not re-executed
    assert call_count == 1
    assert batch2.results[0].content == "v1"


# ── TEX-10: Cache clear ───────────────────────────────────────────


async def test_tex_10_cache_clear():
    """TEX-10: clear_cache() invalidates all cached entries."""
    from core.tool_executor import (
        ExecutionStatus,
        ToolCallRequest,
        ToolExecutor,
    )

    call_count = 0

    class TtlTool(BaseTool):
        idempotent = True

        @property
        def info(self) -> ToolInfo:
            return ToolInfo(name="ttl", description="", parameters={})

        async def execute(self, **kwargs) -> ToolResult:
            nonlocal call_count
            call_count += 1
            return ToolResult(content=f"v{call_count}")

    ToolRegistry.register(TtlTool())

    executor = ToolExecutor(
        registry=ToolRegistry(),
        enable_cache=True,
    )

    await executor.execute_batch([ToolCallRequest(tool_name="ttl")])
    assert call_count == 1

    # Clear cache and re-execute
    executor.clear_cache()
    await executor.execute_batch([ToolCallRequest(tool_name="ttl")])
    assert call_count == 2


# ── TEX-11: Audit logging with duration_ms and retry_count ─────────


async def test_tex_11_audit_logging():
    """TEX-11: Execution results include audit metadata."""
    from core.tool_executor import (
        ExecutionStatus,
        ToolCallRequest,
        ToolExecutor,
    )

    call_count = 0

    class FlakyAuditTool(BaseTool):
        @property
        def info(self) -> ToolInfo:
            return ToolInfo(name="audit", description="", parameters={})

        async def execute(self, **kwargs) -> ToolResult:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("transient")
            return ToolResult(content="ok")

    ToolRegistry.register(FlakyAuditTool())

    executor = ToolExecutor(
        registry=ToolRegistry(),
        max_retries=2,
        retry_backoff_factor=0.01,
    )
    batch = await executor.execute_batch([ToolCallRequest(tool_name="audit")])

    result = batch.results[0]
    assert result.status == ExecutionStatus.SUCCESS
    assert result.duration_ms > 0
    assert result.retry_count >= 1


# ── TEX-12: Batch statistics accuracy ─────────────────────────────


async def test_tex_12_batch_stats():
    """TEX-12: BatchExecutionResult has accurate properties."""
    from core.tool_executor import (
        ExecutionStatus,
        ToolCallRequest,
        ToolExecutor,
    )

    _register("ok_tool", return_result=ToolResult(content="good"))

    executor = ToolExecutor(registry=ToolRegistry(), max_retries=0)
    batch = await executor.execute_batch(
        [
            ToolCallRequest(tool_name="ok_tool"),
            ToolCallRequest(tool_name="unknown_fail"),
            ToolCallRequest(tool_name="ok_tool"),
        ],
    )

    assert len(batch.results) == 3
    assert batch.total_duration_ms >= 0
    # Two succeed, one fails
    success_count = sum(1 for r in batch.results if r.success)
    fail_count = sum(1 for r in batch.results if not r.success)
    assert success_count == 2
    assert fail_count == 1
    assert batch.any_failed is True
    assert batch.all_success is False


# ── Additional: ToolCallRequest cache_key ───────────────────────────


async def test_cache_key_deterministic():
    """ToolCallRequest.cache_key is deterministic."""
    from core.tool_executor import ToolCallRequest

    r1 = ToolCallRequest(tool_name="weather", arguments={"city": "Beijing"})
    r2 = ToolCallRequest(tool_name="weather", arguments={"city": "Beijing"})
    assert r1.cache_key == r2.cache_key


async def test_cache_key_arg_order_independent():
    """ToolCallRequest.cache_key ignores argument order."""
    from core.tool_executor import ToolCallRequest

    r1 = ToolCallRequest(tool_name="calc", arguments={"a": 1, "b": 2})
    r2 = ToolCallRequest(tool_name="calc", arguments={"b": 2, "a": 1})
    assert r1.cache_key == r2.cache_key


async def test_cache_key_different_args():
    """Different args produce different cache keys."""
    from core.tool_executor import ToolCallRequest

    r1 = ToolCallRequest(tool_name="weather", arguments={"city": "Beijing"})
    r2 = ToolCallRequest(tool_name="weather", arguments={"city": "Shanghai"})
    assert r1.cache_key != r2.cache_key


# ── Additional: BatchExecutionResult properties ─────────────────────


async def test_batch_all_success():
    """BatchExecutionResult.all_success is True when all succeed."""
    from core.tool_executor import (
        ExecutionStatus,
        ToolCallRequest,
        ToolExecutor,
    )

    _register("b1", return_result=ToolResult(content="a"))
    _register("b2", return_result=ToolResult(content="b"))

    executor = ToolExecutor(registry=ToolRegistry())
    batch = await executor.execute_batch(
        [
            ToolCallRequest(tool_name="b1"),
            ToolCallRequest(tool_name="b2"),
        ],
    )

    assert batch.all_success is True
    assert batch.any_failed is False


async def test_empty_batch():
    """Empty batch has no results and all_success is True (vacuous)."""
    from core.tool_executor import ToolExecutor

    executor = ToolExecutor(registry=ToolRegistry())
    batch = await executor.execute_batch([])

    assert len(batch.results) == 0
    assert batch.all_success is True
    assert batch.any_failed is False


# ── Additional: Tool-level timeout override ────────────────────────


async def test_tool_timeout_override():
    """Tool's own timeout overrides global default."""
    from core.tool_executor import (
        ExecutionStatus,
        ToolCallRequest,
        ToolExecutor,
    )

    _register("short_timeout", delay=0.5, timeout=0.1)

    executor = ToolExecutor(registry=ToolRegistry(), default_timeout=10.0, max_retries=0)
    batch = await executor.execute_batch(
        [ToolCallRequest(tool_name="short_timeout")]
    )

    result = batch.results[0]
    assert not result.success
    # Should use tool-level timeout (0.1s), not global (10s)
