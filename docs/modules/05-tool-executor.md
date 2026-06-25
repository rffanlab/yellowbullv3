# 工具执行器（ToolExecutor）详细设计

## 1. 职责边界

| 职责 | 说明 |
|------|------|
| **超时控制** | 每个工具调用有独立超时，防止单个工具阻塞整个 Agent 循环 |
| **重试策略** | 可配置的重试次数 + 指数退避，自动恢复临时性错误 |
| **并行调度** | 同一轮次的多个 tool call 支持并发执行（默认）或串行执行 |
| **错误隔离** | 单个工具失败不影响其他工具的执行 |
| **结果缓存** | 相同参数的工具调用可复用缓存结果（可选开启） |
| **执行审计** | 记录每次执行的耗时、状态、输入输出摘要 |

---

## 2. 模块位置

```
core/
├── __init__.py
├── agent.py              # Agent Core：编排主循环
├── tool_executor.py      # ToolExecutor：工具执行引擎（新增）
├── session_manager.py    # Session Manager：会话管理
└── context_builder.py    # Context Builder：上下文构建
```

---

## 3. 核心类设计 `core/tool_executor.py`

### 3.1 数据模型

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import time


class ExecutionStatus(Enum):
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    RETRY_EXHAUSTED = "retry_exhausted"


@dataclass(frozen=True)
class ToolCallRequest:
    """单次工具调用请求"""
    tool_name: str
    arguments: dict[str, Any]

    @property
    def cache_key(self) -> str:
        """生成缓存键：tool_name + sorted args（参数顺序无关）"""
        import hashlib
        raw = f"{self.tool_name}:{json.dumps(self.arguments, sort_keys=True)}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class ExecutionResult:
    """单次工具调用结果"""
    tool_name: str
    status: ExecutionStatus
    content: str = ""           # 成功时的输出内容
    error: str = ""             # 失败时的错误信息
    duration_ms: float = 0.0    # 执行耗时（毫秒）
    retry_count: int = 0        # 实际重试次数

    @property
    def success(self) -> bool:
        return self.status == ExecutionStatus.SUCCESS


@dataclass(frozen=True)
class BatchExecutionResult:
    """一批工具调用的聚合结果"""
    results: list[ExecutionResult] = field(default_factory=list)
    total_duration_ms: float = 0.0

    @property
    def all_success(self) -> bool:
        return all(r.success for r in self.results) if self.results else True

    @property
    def any_failed(self) -> bool:
        return not self.all_success and len(self.results) > 0
```

### 3.2 ToolExecutor 核心类

```python
import asyncio
import json
import logging
from typing import Any, AsyncIterator

from tools.registry import ToolRegistry
from tools.base import ToolResult

logger = logging.getLogger(__name__)


class ToolExecutor:
    """
    工具执行引擎。

    职责：
    - 接收 Agent Core 传来的 tool call 列表
    - 对每个调用应用超时、重试策略
    - 支持并行/串行两种调度模式
    - 返回结构化结果供 Agent Core 回注到对话历史

    使用方式：
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
        registry: ToolRegistry,
        default_timeout: float = 30.0,       # 默认超时（秒）
        max_retries: int = 2,                # 最大重试次数
        retry_backoff_factor: float = 1.5,   # 退避因子
        parallel: bool = True,               # 是否并行执行同一批 tool calls
        enable_cache: bool = False,          # 是否启用结果缓存
    ):
        self._registry = registry
        self._default_timeout = default_timeout
        self._max_retries = max_retries
        self._backoff_factor = retry_backoff_factor
        self._parallel = parallel
        self._enable_cache = enable_cache

        # 简单内存缓存：cache_key -> ExecutionResult
        self._cache: dict[str, ExecutionResult] = {}

    async def execute_batch(
        self,
        requests: list[ToolCallRequest],
    ) -> BatchExecutionResult:
        """
        执行一批工具调用。

        Args:
            requests: 工具调用请求列表（同一轮次 LLM 返回的所有 tool calls）

        Returns:
            BatchExecutionResult：包含每个调用的结果和总耗时
        """
        start_time = time.monotonic()
        results: list[ExecutionResult] = []

        if self._parallel:
            # 并行执行：所有工具同时启动，各自独立超时/重试
            tasks = [self._execute_single(req) for req in requests]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            # 处理异常（asyncio.gather 返回的 Exception）
            resolved: list[ExecutionResult] = []
            for r in results:
                if isinstance(r, ExecutionResult):
                    resolved.append(r)
                else:
                    # 不应发生，但兜底处理
                    req = requests[len(resolved)]
                    resolved.append(ExecutionResult(
                        tool_name=req.tool_name,
                        status=ExecutionStatus.FAILED,
                        error=f"Unexpected error: {r}",
                    ))
            results = resolved
        else:
            # 串行执行：按顺序逐个执行（适用于有依赖关系的工具）
            for req in requests:
                result = await self._execute_single(req)
                results.append(result)

        total_ms = (time.monotonic() - start_time) * 1000
        return BatchExecutionResult(results=results, total_duration_ms=total_ms)

    async def _execute_single(
        self,
        request: ToolCallRequest,
    ) -> ExecutionResult:
        """执行单个工具调用（含超时 + 重试）"""
        # 1. 检查缓存
        if self._enable_cache and request.cache_key in self._cache:
            cached = self._cache[request.cache_key]
            logger.debug(f"Cache hit for {request.tool_name}({request.arguments})")
            return ExecutionResult(
                tool_name=request.tool_name,
                status=cached.status,
                content=cached.content,
                error=cached.error,
            )

        # 2. 查找工具
        tool = self._registry.get(request.tool_name)
        if tool is None:
            logger.warning(f"Tool not found: {request.tool_name}")
            return ExecutionResult(
                tool_name=request.tool_name,
                status=ExecutionStatus.FAILED,
                error=f"Unknown tool: {request.tool_name}",
            )

        # 3. 获取工具级超时（优先使用工具自身配置，否则用默认值）
        timeout = getattr(tool, "timeout", None) or self._default_timeout

        # 4. 重试循环
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
                # 超时保护
                tool_result: ToolResult = await asyncio.wait_for(
                    self._invoke_tool(tool, request.arguments),
                    timeout=timeout,
                )
                duration_ms = (time.monotonic() - start_ms) * 1000

                result = ExecutionResult(
                    tool_name=request.tool_name,
                    status=ExecutionStatus.SUCCESS if tool_result.success else ExecutionStatus.FAILED,
                    content=tool_result.content if tool_result.success else "",
                    error="" if tool_result.success else (tool_result.error or "Unknown error"),
                    duration_ms=duration_ms,
                    retry_count=attempt,
                )

                # 成功则缓存（如果启用）
                if result.success and self._enable_cache:
                    self._cache[request.cache_key] = result

                return result

            except asyncio.TimeoutError:
                last_error = f"Tool '{request.tool_name}' timed out after {timeout}s"
                logger.warning(last_error)
            except Exception as e:
                last_error = str(e)
                logger.error(f"Tool '{request.tool_name}' error on attempt {attempt}: {e}")

        # 5. 所有重试耗尽
        return ExecutionResult(
            tool_name=request.tool_name,
            status=ExecutionStatus.RETRY_EXHAUSTED if self._max_retries > 0 else ExecutionStatus.FAILED,
            error=f"Failed after {self._max_retries} retries: {last_error}",
        )

    async def _invoke_tool(
        self,
        tool: Any,
        arguments: dict[str, Any],
    ) -> ToolResult:
        """调用工具的实际执行方法"""
        # 如果工具是可调用的（有 execute 方法）
        if hasattr(tool, "execute"):
            return await tool.execute(**arguments)
        elif callable(tool):
            return await tool(**arguments)
        else:
            raise RuntimeError(f"Tool '{tool}' is not executable")

    def clear_cache(self) -> None:
        """清空缓存"""
        self._cache.clear()
```

---

## 4. 与 Agent Core 的集成

### 4.1 Agent Core 中的使用方式

```python
# core/agent.py（简化版）

from core.tool_executor import (
    ToolExecutor,
    ToolCallRequest,
    ExecutionStatus,
)


class Agent:
    def __init__(self, ...):
        self._executor = ToolExecutor(
            registry=tool_registry,
            default_timeout=config.agent.tool_timeout_seconds,
            max_retries=config.agent.tool_retry_limit,
            parallel=True,
        )

    async def _execute_tools(self, tool_calls: list[dict]) -> None:
        """Agent 主循环中的工具执行步骤"""
        # 将 LLM 的 tool calls 转换为执行请求
        requests = []
        for tc in tool_calls:
            func_info = tc.get("function", {})
            args_json = func_info.get("arguments", "{}")
            if isinstance(args_json, str):
                args = json.loads(args_json)
            else:
                args = args_json

            requests.append(ToolCallRequest(
                tool_name=func_info.get("name", ""),
                arguments=args,
            ))

        # 执行（并行/串行由 ToolExecutor 配置决定）
        batch_result = await self._executor.execute_batch(requests)

        # 将结果回注到会话历史（按顺序对应，无需 call_id）
        for request, result in zip(requests, batch_result.results):
            content = result.content if result.success else f"Error: {result.error}"
            tool_msg = Message(
                role="tool",
                content=content,
                tool_name=request.tool_name,
            )
            await self._session_manager.append_message(self._session_id, tool_msg)

        # 记录执行审计日志
        logger.info(
            f"Batch execution: {len(batch_result.results)} tools, "
            f"{sum(1 for r in batch_result.results if r.success)} succeeded, "
            f"total {batch_result.total_duration_ms:.0f}ms"
        )
```

### 4.2 数据流图

```
Agent Core                    ToolExecutor                  ToolRegistry
    │                              │                              │
    │── execute_batch(requests) ──>│                              │
    │                              │   ┌──────────────────────┐   │
    │                              │   │ for each request:     │   │
    │                              │   │  1. check cache       │   │
    │                              │   │  2. get(tool_name) ───┼──>│ (lookup)
    │                              │   │  3. retry loop        │<──┤
    │                              │   │     a. asyncio.wait_for│   │
    │                              │   │     b. tool.execute() │   │
    │                              │   │     c. on timeout/retry│  │
    │                              │   └──────────────────────┘   │
    │                              │                              │
    │<── BatchExecutionResult ─────│                              │
    │   (含每个 tool call 的结果)   │                              │
    │                              │                              │
```

---

## 5. 配置项 `default.yaml`

```yaml
agent:
  # ... 其他配置 ...
  tool_timeout_seconds: 30      # 工具默认超时（秒）
  tool_retry_limit: 2           # 最大重试次数
  retry_backoff_factor: 1.5     # 指数退避因子
  parallel_tool_calls: true     # 是否并行执行同一批 tool calls
  enable_tool_cache: false      # 是否启用工具结果缓存（默认关闭）

tools:
  builtin:
    current_time: { enabled: true }
    calculator:   { enabled: true }
    web_search:   { enabled: true, timeout: 15 }  # 可单独设置超时
```

---

## 6. 重试策略详解

### 6.1 退避时间表（max_retries=3, backoff_factor=1.5）

| 尝试 | 等待时间 | 累计耗时上限 |
|------|---------|-------------|
| 第 0 次（首次） | 无 | timeout |
| 第 1 次重试 | 1.5s | 2 × timeout + 1.5s |
| 第 2 次重试 | 2.25s | 3 × timeout + 3.75s |
| 第 3 次重试 | 3.38s | 4 × timeout + 7.13s |

### 6.2 可重试 vs 不可重试错误

| 错误类型 | 是否重试 | 原因 |
|---------|---------|------|
| `asyncio.TimeoutError` | ✅ 是 | 网络超时可能是临时性的 |
| HTTP 5xx / Connection Error | ✅ 是 | 服务端临时故障 |
| HTTP 400 / Invalid Input | ❌ 否 | 参数错误，重试无意义 |
| Tool not found | ❌ 否 | 配置问题，重试无意义 |

> **注意**: P0 阶段所有异常统一视为可重试。P1+ 可根据异常类型精细化处理。

---

## 7. 并行 vs 串行执行

### 7.1 默认：并行执行（`parallel=True`）

```
LLM 返回 3 个 tool calls: [get_time, calc, search]

并行模式:
  get_time ────────┐
  calc   ──────────┼──> 总耗时 = max(各工具耗时)
  search ──────────┘

串行模式:
  get_time ──> calc ──> search
              总耗时 = sum(各工具耗时)
```

### 7.2 何时使用串行

- 工具之间存在数据依赖（如：先查询数据库，再对结果做计算）
- 资源受限环境（并发连接数有限制）
- P0 MVP 阶段默认并行即可

---

## 8. 单元测试要点

```python
# tests/core/test_tool_executor.py

class TestToolExecutor:
    async def test_execute_single_success(self):
        """正常执行成功"""
        ...

    async def test_execute_timeout(self):
        """超时返回 TIMEOUT 状态"""
        ...

    async def test_retry_on_failure(self):
        """失败后重试，最终成功"""
        ...

    async def test_retry_exhausted(self):
        """超过最大重试次数返回 RETRY_EXHAUSTED"""
        ...

    async def test_parallel_execution(self):
        """多个 tool call 并行执行"""
        ...

    async def test_serial_execution(self):
        """串行模式按顺序执行"""
        ...

    async def test_unknown_tool(self):
        """不存在的工具返回 FAILED"""
        ...

    async def test_cache_hit(self):
        """缓存命中时直接返回缓存结果"""
        ...

    async def test_error_isolation(self):
        """单个工具失败不影响其他工具执行"""
        ...
```

---

## 9. 设计总结

| 特性 | P0 MVP 实现 | P1+ 扩展 |
|------|------------|---------|
| 超时控制 | ✅ asyncio.wait_for | 按工具类型分级超时 |
| 重试策略 | ✅ 指数退避 | 可重试错误分类 |
| 并行调度 | ✅ parallel/serial 切换 | 依赖图驱动的拓扑排序 |
| 结果缓存 | ✅ 内存缓存（可选） | Redis 分布式缓存 |
| 执行审计 | ✅ 日志记录 | Prometheus 指标 + Trace |
| 熔断器 | ❌ | 连续失败自动熔断 |
