# 错误处理详细设计

## 1. 设计目标

| 目标 | 说明 |
|------|------|
| **统一异常体系** | 所有模块使用一致的异常层次结构，便于全局捕获和处理 |
| **LLM 超时分类** | 区分连接超时、读取超时、处理超时，分别采取不同恢复策略 |
| **工具失败恢复** | 支持重试、降级、熔断三种恢复机制 |
| **Token 超限处理** | 自动截断、压缩或分页，避免请求被 LLM API 拒绝 |
| **优雅降级** | 非核心功能失败不影响主流程（如 RAG 不可用时仍可进行对话） |

---

## 2. 异常层次结构 `errors/exceptions.py`

```python
"""
YellowBull 统一异常体系。

层次结构：
    YellowBullError (基类)
    ├── LLMError              # LLM 相关错误
    │   ├── LLMAPIError       # API 调用失败（网络、认证等）
    │   ├── LLMTIMEOUT        # 超时错误
    │   ├── LLMRateLimit      # 速率限制
    │   └── TokenLimitError   # Token 超限
    ├── ToolError             # 工具执行错误
    │   ├── ToolTimeout       # 工具执行超时
    │   └── ToolRetryable     # 可重试的工具错误
    ├── SessionError          # 会话管理错误
    │   ├── SessionNotFound   # 会话不存在
    │   └── SessionExpired    # 会话已过期
    ├── RAGError              # RAG 相关错误
    │   ├── EmbeddingError    # 嵌入失败
    │   └── RetrievalError    # 检索失败
    ├── MemoryError           # 记忆系统错误
    │   └── MemoryFull        # 记忆容量超限
    ├── PluginError           # 插件相关错误
    │   ├── PluginLoadError   # 插件加载失败
    │   └── PluginExecutionError  # 插件执行异常
    ├── MultimodalError       # 多模态处理错误
    │   ├── ASRError          # 语音识别失败
    │   ├── TTSError          # 语音合成失败
    │   └── ImageAnalysisError  # 图像分析失败
    └── ConfigError           # 配置相关错误
        └── ValidationError   # 配置验证失败


每个异常包含：
- error_code: 机器可读的错误码（用于前端/客户端判断）
- retryable: 是否可重试
- severity: 严重程度 (info, warning, error, critical)
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional


class ErrorSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass(frozen=True)
class ErrorCode:
    """错误码定义"""
    value: str
    http_status: int
    retryable: bool = False
    severity: ErrorSeverity = ErrorSeverity.ERROR

    def __str__(self):
        return self.value


# ---------- 错误码注册表 ----------

ERRORS = {
    # LLM errors (1xxx)
    "LLM_API_ERROR": ErrorCode("LLM_API_ERROR", 502, retryable=True),
    "LLM_TIMEOUT": ErrorCode("LLM_TIMEOUT", 504, retryable=True),
    "LLM_RATE_LIMIT": ErrorCode("LLM_RATE_LIMIT", 429, retryable=True),
    "TOKEN_LIMIT_EXCEEDED": ErrorCode("TOKEN_LIMIT_EXCEEDED", 413, retryable=False),

    # Tool errors (2xxx)
    "TOOL_EXECUTION_ERROR": ErrorCode("TOOL_EXECUTION_ERROR", 500, retryable=False),
    "TOOL_TIMEOUT": ErrorCode("TOOL_TIMEOUT", 504, retryable=True),
    "TOOL_RETRYABLE": ErrorCode("TOOL_RETRYABLE", 503, retryable=True),

    # Session errors (3xxx)
    "SESSION_NOT_FOUND": ErrorCode("SESSION_NOT_FOUND", 404, retryable=False),
    "SESSION_EXPIRED": ErrorCode("SESSION_EXPIRED", 410, retryable=False),

    # RAG errors (4xxx)
    "EMBEDDING_ERROR": ErrorCode("EMBEDDING_ERROR", 500, retryable=True),
    "RETRIEVAL_ERROR": ErrorCode("RETRIEVAL_ERROR", 500, retryable=True),

    # Memory errors (5xxx)
    "MEMORY_FULL": ErrorCode("MEMORY_FULL", 413, retryable=False),

    # Plugin errors (6xxx)
    "PLUGIN_LOAD_ERROR": ErrorCode("PLUGIN_LOAD_ERROR", 500, retryable=False),
    "PLUGIN_EXECUTION_ERROR": ErrorCode("PLUGIN_EXECUTION_ERROR", 500, retryable=False),

    # Multimodal errors (7xxx)
    "ASR_ERROR": ErrorCode("ASR_ERROR", 500, retryable=True),
    "TTS_ERROR": ErrorCode("TTS_ERROR", 500, retryable=True),
    "IMAGE_ANALYSIS_ERROR": ErrorCode("IMAGE_ANALYSIS_ERROR", 500, retryable=True),

    # Config errors (8xxx)
    "CONFIG_VALIDATION_ERROR": ErrorCode("CONFIG_VALIDATION_ERROR", 400, retryable=False),
}


class YellowBullError(Exception):
    """YellowBull 应用基类异常"""

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        details: dict | None = None,
        cause: Exception | None = None,
    ):
        self.message = message
        self.error_code = error_code or "UNKNOWN_ERROR"
        self.details = details or {}
        self.cause = cause

        error_info = ERRORS.get(self.error_code)
        self.retryable = error_info.retryable if error_info else False
        self.severity = error_info.severity if error_info else ErrorSeverity.ERROR
        self.http_status = error_info.http_status if error_info else 500

        super().__init__(f"[{self.error_code}] {message}")


# ---------- LLM Errors ----------

class LLMAPIError(YellowBullError):
    """LLM API 调用失败"""
    def __init__(self, message: str, status_code: int | None = None, **kwargs):
        super().__init__(
            message=message,
            error_code="LLM_API_ERROR",
            details={"status_code": status_code},
            **kwargs,
        )


class LLMTIMEOUT(YellowBullError):
    """LLM 请求超时"""

    class TimeoutType(str, Enum):
        CONNECT = "connect"     # 连接建立超时
        READ = "read"           # 响应读取超时
        PROCESSING = "processing"  # 服务端处理超时（长时间无响应）

    def __init__(
        self,
        message: str,
        timeout_type: TimeoutType = TimeoutType.READ,
        elapsed_seconds: float | None = None,
        **kwargs,
    ):
        super().__init__(
            message=message,
            error_code="LLM_TIMEOUT",
            details={
                "timeout_type": timeout_type.value,
                "elapsed_seconds": elapsed_seconds,
            },
            **kwargs,
        )


class LLMRateLimit(YellowBullError):
    """LLM API 速率限制"""

    def __init__(
        self,
        message: str,
        retry_after_seconds: float | None = None,
        **kwargs,
    ):
        super().__init__(
            message=message,
            error_code="LLM_RATE_LIMIT",
            details={"retry_after_seconds": retry_after_seconds},
            **kwargs,
        )


class TokenLimitError(YellowBullError):
    """Token 数量超限"""

    def __init__(
        self,
        message: str,
        current_tokens: int,
        max_tokens: int,
        **kwargs,
    ):
        super().__init__(
            message=message,
            error_code="TOKEN_LIMIT_EXCEEDED",
            details={
                "current_tokens": current_tokens,
                "max_tokens": max_tokens,
            },
            **kwargs,
        )


# ---------- Tool Errors ----------

class ToolError(YellowBullError):
    """工具执行错误"""
    def __init__(self, message: str, tool_name: str = "", **kwargs):
        super().__init__(
            message=message,
            error_code="TOOL_EXECUTION_ERROR",
            details={"tool_name": tool_name},
            **kwargs,
        )


class ToolTimeout(YellowBullError):
    """工具执行超时"""
    def __init__(self, message: str, tool_name: str = "", elapsed_seconds: float | None = None, **kwargs):
        super().__init__(
            message=message,
            error_code="TOOL_TIMEOUT",
            details={"tool_name": tool_name, "elapsed_seconds": elapsed_seconds},
            **kwargs,
        )


class ToolRetryable(YellowBullError):
    """可重试的工具错误"""
    def __init__(self, message: str, tool_name: str = "", retry_count: int = 0, **kwargs):
        super().__init__(
            message=message,
            error_code="TOOL_RETRYABLE",
            details={"tool_name": tool_name, "retry_count": retry_count},
            **kwargs,
        )


# ---------- Session Errors ----------

class SessionError(YellowBullError):
    """会话管理错误"""
    pass


class SessionNotFound(SessionError):
    def __init__(self, session_id: str = "", **kwargs):
        super().__init__(
            message=f"Session not found: {session_id}",
            error_code="SESSION_NOT_FOUND",
            details={"session_id": session_id},
            **kwargs,
        )


class SessionExpired(SessionError):
    def __init__(self, session_id: str = "", expired_at: float | None = None, **kwargs):
        super().__init__(
            message=f"Session expired: {session_id}",
            error_code="SESSION_EXPIRED",
            details={"session_id": session_id, "expired_at": expired_at},
            **kwargs,
        )


# ---------- RAG Errors ----------

class RAGError(YellowBullError):
    """RAG 相关错误"""
    pass


class EmbeddingError(RAGError):
    def __init__(self, message: str, **kwargs):
        super().__init__(message=message, error_code="EMBEDDING_ERROR", **kwargs)


class RetrievalError(RAGError):
    def __init__(self, message: str, **kwargs):
        super().__init__(message=message, error_code="RETRIEVAL_ERROR", **kwargs)


# ---------- Memory Errors ----------

class MemoryError(YellowBullError):
    """记忆系统错误"""
    pass


class MemoryFull(MemoryError):
    def __init__(self, message: str, current_size: int = 0, max_size: int = 0, **kwargs):
        super().__init__(
            message=message,
            error_code="MEMORY_FULL",
            details={"current_size": current_size, "max_size": max_size},
            **kwargs,
        )


# ---------- Plugin Errors ----------

class PluginError(YellowBullError):
    """插件相关错误"""
    pass


class PluginLoadError(PluginError):
    def __init__(self, message: str, plugin_name: str = "", **kwargs):
        super().__init__(
            message=message,
            error_code="PLUGIN_LOAD_ERROR",
            details={"plugin_name": plugin_name},
            **kwargs,
        )


class PluginExecutionError(PluginError):
    def __init__(self, message: str, plugin_name: str = "", **kwargs):
        super().__init__(
            message=message,
            error_code="PLUGIN_EXECUTION_ERROR",
            details={"plugin_name": plugin_name},
            **kwargs,
        )


# ---------- Multimodal Errors ----------

class MultimodalError(YellowBullError):
    """多模态处理错误"""
    pass


class ASRError(MultimodalError):
    def __init__(self, message: str, engine: str = "", **kwargs):
        super().__init__(
            message=message,
            error_code="ASR_ERROR",
            details={"engine": engine},
            **kwargs,
        )


class TTSError(MultimodalError):
    def __init__(self, message: str, engine: str = "", **kwargs):
        super().__init__(
            message=message,
            error_code="TTS_ERROR",
            details={"engine": engine},
            **kwargs,
        )


class ImageAnalysisError(MultimodalError):
    def __init__(self, message: str, analyzer: str = "", **kwargs):
        super().__init__(
            message=message,
            error_code="IMAGE_ANALYSIS_ERROR",
            details={"analyzer": analyzer},
            **kwargs,
        )


# ---------- Config Errors ----------

class ConfigError(YellowBullError):
    """配置相关错误"""
    pass


class ValidationError(ConfigError):
    def __init__(self, message: str, field_name: str = "", **kwargs):
        super().__init__(
            message=message,
            error_code="CONFIG_VALIDATION_ERROR",
            details={"field_name": field_name},
            **kwargs,
        )
```

---

## 3. 重试与熔断 `errors/resilience.py`

```python
import time
import asyncio
import logging
from typing import Callable, Awaitable, TypeVar, Optional
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitBreaker:
    """
    熔断器模式实现。

    三种状态：
    - CLOSED（关闭）：正常通行，记录失败次数
    - OPEN（打开）：直接拒绝请求，等待冷却时间后进入 HALF-OPEN
    - HALF-OPEN（半开）：允许有限请求通过，成功则回到 CLOSED，失败则回到 OPEN

    适用场景：LLM API 持续不可用、数据库连接池耗尽等
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,      # 连续失败多少次触发熔断
        recovery_timeout: float = 30.0,   # 冷却时间（秒）
        half_open_max_calls: int = 3,     # 半开状态允许的最大请求数
    ):
        self._name = name
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._half_open_max_calls = half_open_max_calls

        self._state = "CLOSED"
        self._failure_count = 0
        self._last_failure_time: float | None = None
        self._half_open_successes = 0

    @property
    def state(self) -> str:
        if self._state == "OPEN":
            # 检查是否已过冷却时间
            if (self._last_failure_time and
                time.monotonic() - self._last_failure_time > self._recovery_timeout):
                self._state = "HALF-OPEN"
                self._half_open_successes = 0
        return self._state

    def record_success(self):
        if self._state == "HALF-OPEN":
            self._half_open_successes += 1
            if self._half_open_successes >= self._half_open_max_calls:
                logger.info(f"Circuit breaker '{self._name}' → CLOSED")
                self._state = "CLOSED"
                self._failure_count = 0
        else:
            self._failure_count = 0

    def record_failure(self):
        self._failure_count += 1
        self._last_failure_time = time.monotonic()

        if self._state == "HALF-OPEN":
            logger.warning(f"Circuit breaker '{self._name}' → OPEN (half-open failure)")
            self._state = "OPEN"
        elif self._failure_count >= self._failure_threshold:
            logger.warning(
                f"Circuit breaker '{self._name}' → OPEN "
                f"(failures: {self._failure_count}/{self._failure_threshold})"
            )
            self._state = "OPEN"

    def call(self, func: Callable[..., Awaitable[T]], *args, **kwargs) -> Awaitable[T]:
        """包装异步调用，应用熔断逻辑"""
        async def _wrapped() -> T:
            if self.state == "OPEN":
                raise YellowBullError(
                    f"Circuit breaker '{self._name}' is OPEN",
                    error_code="SERVICE_UNAVAILABLE",
                    retryable=True,
                )

            try:
                result = await func(*args, **kwargs)
                self.record_success()
                return result
            except YellowBullError as e:
                if e.retryable:
                    self.record_failure()
                raise
        return _wrapped()


async def retry_with_backoff(
    func: Callable[..., Awaitable[T]],
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential_base: float = 2.0,
    retry_on: tuple[type[Exception], ...] | None = None,
    *args,
    **kwargs,
) -> T:
    """
    指数退避重试。

    Args:
        func:          要执行的异步函数
        max_retries:   最大重试次数
        base_delay:    基础延迟（秒）
        max_delay:     最大延迟上限（秒）
        exponential_base: 指数底数
        retry_on:      哪些异常触发重试（默认只重试 YellowBullError.retryable=True）

    Returns:
        函数返回值

    Raises:
        最后一次失败的异常
    """
    last_exception: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_exception = e

            # 检查是否应该重试
            should_retry = False
            if retry_on and isinstance(e, retry_on):
                should_retry = True
            elif isinstance(e, YellowBullError) and e.retryable:
                should_retry = True

            if not should_retry or attempt >= max_retries:
                break

            delay = min(base_delay * (exponential_base ** attempt), max_delay)
            # 添加抖动，避免 thundering herd
            import random
            jitter = random.uniform(0, delay * 0.1)
            await asyncio.sleep(delay + jitter)
            logger.info(
                f"Retry {attempt + 1}/{max_retries} for '{func.__name__}' "
                f"after {delay + jitter:.2f}s: {e}"
            )

    raise last_exception


# ---------- FastAPI 全局异常处理器 ----------

from fastapi import Request, FastAPI
from fastapi.responses import JSONResponse


def register_error_handlers(app: FastAPI):
    """注册全局异常处理器"""

    @app.exception_handler(YellowBullError)
    async def yellowbull_handler(request: Request, exc: YellowBullError):
        logger.error(
            f"YellowBullError: [{exc.error_code}] {exc.message}",
            extra={"details": exc.details},
            exc_info=exc.cause,
        )

        return JSONResponse(
            status_code=exc.http_status,
            content={
                "error": {
                    "code": exc.error_code,
                    "message": exc.message,
                    "retryable": exc.retryable,
                    **exc.details,
                }
            },
        )

    @app.exception_handler(Exception)
    async def unexpected_handler(request: Request, exc: Exception):
        logger.critical(
            f"Unhandled exception: {type(exc).__name__}: {exc}",
            exc_info=exc,
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "An unexpected error occurred. Please try again later.",
                }
            },
        )
```

---

## 4. Token 超限处理 `errors/token_guard.py`

```python
"""
Token 数量守卫。

职责：
- 在发送给 LLM 之前检查总 token 数是否超限
- 自动截断、压缩或分页，避免请求被拒绝
- 提供多种策略供不同场景使用
"""

from typing import Callable


class TokenGuard:
    """
    Token 数量守卫。

    三种处理策略：
    1. TRUNCATE（截断）：从对话历史头部开始丢弃消息，直到 token 数在限制内
    2. COMPRESS（压缩）：将早期对话摘要化，保留关键信息
    3. PAGINATE（分页）：返回错误，由上层决定如何分页处理

    Usage:
        guard = TokenGuard(
            max_tokens=4096,
            reserved_tokens=1024,   # 为回复预留空间
            strategy="truncate",
        )
        safe_messages = await guard.ensure_fit(messages)
    """

    class Strategy(str):
        TRUNCATE = "truncate"
        COMPRESS = "compress"
        PAGINATE = "paginate"

    def __init__(
        self,
        max_tokens: int,
        reserved_tokens: int = 1024,
        strategy: str = Strategy.TRUNCATE,
        tokenizer: Callable[[str], list[int]] | None = None,
    ):
        self._max_tokens = max_tokens
        self._reserved_tokens = reserved_tokens
        self._strategy = strategy
        # 默认使用 tiktoken cl100k_base（GPT-4 / GPT-3.5）
        if tokenizer:
            self._count_tokens = lambda text: len(tokenizer(text))
        else:
            try:
                import tiktoken
                enc = tiktoken.get_encoding("cl100k_base")
                self._count_tokens = lambda text: len(enc.encode(text))
            except ImportError:
                # 粗略估算：英文 ~4 chars/token，中文 ~1.5 chars/token
                def rough_count(text: str) -> int:
                    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
                    other_chars = len(text) - chinese_chars
                    return int(chinese_chars / 1.5 + other_chars / 4)
                self._count_tokens = rough_count

    @property
    def available_tokens(self) -> int:
        """可用于输入的 token 数（扣除回复预留）"""
        return self._max_tokens - self._reserved_tokens

    async def ensure_fit(
        self,
        messages: list[dict],
        compressor: Callable[[list[dict]], str] | None = None,
    ) -> list[dict]:
        """
        确保消息列表在 token 限制内。

        Args:
            messages:   消息列表 [{"role": ..., "content": ...}, ...]
            compressor: 压缩函数（COMPRESS 策略需要）

        Returns:
            处理后的消息列表

        Raises:
            TokenLimitError: PAGINATE 策略时 token 超限
        """
        total_tokens = self._count_messages(messages)

        if total_tokens <= self.available_tokens:
            return messages

        if self._strategy == self.Strategy.TRUNCATE:
            return self._truncate(messages)
        elif self._strategy == self.Strategy.COMPRESS:
            if not compressor:
                raise ValueError("COMPRESS strategy requires a compressor function")
            return await self._compress(messages, compressor)
        elif self._strategy == self.Strategy.PAGINATE:
            raise TokenLimitError(
                message=f"Token limit exceeded: {total_tokens} > {self.available_tokens}",
                current_tokens=total_tokens,
                max_tokens=self.available_tokens,
            )

    def _count_messages(self, messages: list[dict]) -> int:
        """计算消息列表的总 token 数"""
        total = 0
        for msg in messages:
            # role + content 的 token 开销（约 4 tokens/条）
            total += 4
            if isinstance(msg.get("content"), str):
                total += self._count_tokens(msg["content"])
            elif isinstance(msg.get("content"), list):
                for part in msg["content"]:
                    if isinstance(part, dict) and "text" in part:
                        total += self._count_tokens(part["text"])
        return total

    def _truncate(self, messages: list[dict]) -> list[dict]:
        """从头部开始丢弃消息，直到 token 数在限制内"""
        # 始终保留 system message（如果存在）
        system_msgs = [m for m in messages if m.get("role") == "system"]
        other_msgs = [m for m in messages if m.get("role") != "system"]

        result = list(system_msgs)
        # 从尾部向前累积，保留最近的对话
        cumulative = 0
        for msg in reversed(other_msgs):
            msg_tokens = self._count_messages([msg])
            if cumulative + msg_tokens > self.available_tokens:
                break
            cumulative += msg_tokens
            result.insert(len(system_msgs), msg)

        return result

    async def _compress(
        self,
        messages: list[dict],
        compressor: Callable[[list[dict]], str],
    ) -> list[dict]:
        """将早期对话压缩为摘要"""
        system_msgs = [m for m in messages if m.get("role") == "system"]
        other_msgs = [m for m in messages if m.get("role") != "system"]

        # 保留最近 N 条消息，其余压缩
        recent_count = max(2, len(other_msgs) // 2)
        recent = other_msgs[-recent_count:]
        older = other_msgs[:-recent_count]

        summary = await compressor(older)

        return [
            *system_msgs,
            {"role": "user", content": f"[Previous conversation summary]\n{summary}"},
            *recent,
        ]
```

---

## 5. Agent Core 错误处理集成 `agent/error_integration.py`

```python
"""
Agent Core 中的错误处理集成。

关键原则：
1. LLM 失败 → 重试 → 降级到简单回复
2. 工具失败 → 标记为不可用，继续执行其他工具
3. RAG 失败 → 跳过检索增强，直接对话
4. 记忆失败 → 记录警告，不影响当前会话
"""

import logging
from typing import AsyncIterator, Optional

logger = logging.getLogger(__name__)


class AgentErrorHandler:
    """Agent Core 错误处理协调器"""

    def __init__(
        self,
        circuit_breaker: CircuitBreaker | None = None,
        token_guard: TokenGuard | None = None,
        max_retries: int = 3,
    ):
        self._cb = circuit_breaker or CircuitBreaker("llm_api")
        self._token_guard = token_guard
        self._max_retries = max_retries

    async def safe_llm_call(
        self,
        llm_provider,
        messages: list[dict],
        fallback_message: str = "I'm having trouble processing your request. Please try again.",
        **kwargs,
    ) -> Optional[str]:
        """
        安全的 LLM 调用。

        Flow:
            TokenGuard → CircuitBreaker → retry_with_backoff → fallback
        """
        # Step 1: Token guard
        if self._token_guard:
            try:
                messages = await self._token_guard.ensure_fit(messages)
            except TokenLimitError as e:
                logger.warning(f"Token limit exceeded, using fallback: {e}")
                return fallback_message

        # Step 2: Circuit breaker + retry
        async def _call():
            resp = await llm_provider.chat(messages, **kwargs)
            return resp.content

        try:
            result = await retry_with_backoff(
                self._cb.call(_call),
                max_retries=self._max_retries,
            )
            return result or fallback_message
        except Exception as e:
            logger.error(f"LLM call failed after retries: {e}")
            return fallback_message

    async def safe_tool_execution(
        self,
        tool,
        arguments: dict,
        timeout_seconds: float = 30.0,
    ) -> Optional[str]:
        """
        安全的工具执行。

        Flow:
            asyncio.wait_for → catch ToolError → return error message
        """
        try:
            result = await asyncio.wait_for(
                tool.execute(arguments),
                timeout=timeout_seconds,
            )
            return str(result) if result else None
        except asyncio.TimeoutError:
            raise ToolTimeout(
                f"Tool '{tool.name}' timed out after {timeout_seconds}s",
                tool_name=tool.name,
                elapsed_seconds=timeout_seconds,
            )
        except YellowBullError:
            raise
        except Exception as e:
            logger.error(f"Tool '{tool.name}' execution failed: {e}")
            return f"[Tool Error] Failed to execute '{tool.name}': {str(e)}"

    async def safe_rag_retrieval(
        self,
        rag_pipeline,
        query: str,
    ) -> list[dict]:
        """
        安全的 RAG 检索。

        失败时返回空列表，Agent 继续无增强对话。
        """
        try:
            return await rag_pipeline.retrieve(query)
        except RAGError as e:
            logger.warning(f"RAG retrieval failed (continuing without context): {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected RAG error: {e}")
            return []

    async def safe_memory_update(
        self,
        memory_manager,
        session_id: str,
        messages: list[dict],
    ) -> None:
        """
        安全的记忆更新。

        失败时记录警告，不影响当前会话。
        """
        try:
            await memory_manager.update(session_id, messages)
        except MemoryError as e:
            logger.warning(f"Memory update failed (non-blocking): {e}")
        except Exception as e:
            logger.error(f"Unexpected memory error: {e}")
```

---

## 6. WebSocket 错误处理 `websocket/error_handling.py`

```python
"""
WebSocket 连接中的错误处理。

关键场景：
- 客户端断开 → 清理会话资源
- 消息格式错误 → 返回错误帧，不断开连接
- 服务端异常 → 发送错误事件，保持连接可用
"""

import json
from typing import Any


class WSErrorEvent:
    """WebSocket 错误事件"""

    @staticmethod
    def format_error(error_code: str, message: str, details: dict | None = None) -> dict:
        return {
            "type": "error",
            "error": {
                "code": error_code,
                "message": message,
                **(details or {}),
            }
        }

    @staticmethod
    def format_recovery(error_code: str, message: str) -> dict:
        """发送错误 + 恢复提示"""
        return {
            "type": "recovery",
            "error": {"code": error_code, "message": message},
            "action": "retry_suggested",
        }


async def handle_websocket_message(
    websocket,
    raw_message: str | bytes,
    handler_func,  # 实际处理函数
) -> None:
    """
    WebSocket 消息安全处理器。

    保证：
    - 单个消息失败不会断开连接
    - 所有错误以结构化事件返回给客户端
    """
    try:
        if isinstance(raw_message, bytes):
            raw_message = raw_message.decode("utf-8")

        message = json.loads(raw_message)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        await websocket.send_json(
            WSErrorEvent.format_error(
                "INVALID_MESSAGE",
                f"Failed to parse message: {str(e)}",
            )
        )
        return

    try:
        result = await handler_func(message)
        if result:
            await websocket.send_json(result)
    except YellowBullError as e:
        error_event = WSErrorEvent.format_error(
            e.error_code,
            e.message,
            e.details,
        )
        await websocket.send_json(error_event)
    except Exception as e:
        logger.error(f"Unhandled WebSocket error: {e}", exc_info=True)
        await websocket.send_json(
            WSErrorEvent.format_error(
                "INTERNAL_ERROR",
                "An unexpected error occurred.",
            )
        )


async def handle_connection_close(websocket, session_manager):
    """WebSocket 断开时的资源清理"""
    session_id = getattr(websocket.state, 'session_id', None)
    if session_id:
        try:
            await session_manager.cleanup_on_disconnect(session_id)
        except Exception as e:
            logger.error(f"Failed to cleanup session {session_id}: {e}")
```

---

## 7. 架构总览

```
                    ┌─────────────────────┐
                    │   YellowBullError    │ ← 统一异常基类
                    │   (error_code,       │
                    │    retryable,        │
                    │    severity)         │
                    └──────────┬──────────┘
                               │
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                     ▼
   ┌─────────────┐    ┌─────────────┐      ┌─────────────┐
   │ LLM Errors  │    │ Tool Errors │      │ Session Err │
   │ API/Timeout │    │ Timeout     │      │ NotFound    │
   │ RateLimit   │    │ Retryable   │      │ Expired     │
   │ TokenLimit  │    └─────────────┘      └─────────────┘
   └──────┬──────┘                            ...
          ▼
   ┌─────────────────────┐
   │  Resilience Layer   │
   ├─────────────────────┤
   │ retry_with_backoff  │ ← 指数退避重试
   │ CircuitBreaker      │ ← 熔断保护
   │ TokenGuard          │ ← Token 超限处理
   └──────────┬──────────┘
              ▼
   ┌─────────────────────┐
   │ AgentErrorHandler   │ ← 协调各层错误处理
   │ safe_llm_call       │
   │ safe_tool_execution │
   │ safe_rag_retrieval  │
   │ safe_memory_update  │
   └──────────┬──────────┘
              ▼
   ┌─────────────────────┐
   │ FastAPI Handler     │ ← 全局异常 → JSON Response
   │ WebSocket Handler   │ ← 错误事件帧，保持连接
   └─────────────────────┘
```

---

## 8. 设计总结

| 特性 | 实现方式 |
|------|---------|
| **统一异常** | `YellowBullError` 基类 + 16 个子类，覆盖所有模块 |
| **错误码** | 机器可读的 error_code，映射 HTTP status、重试策略、严重程度 |
| **LLM 超时分类** | connect / read / processing 三种类型，差异化恢复 |
| **重试机制** | 指数退避 + 抖动，仅对 retryable=True 的异常生效 |
| **熔断保护** | Circuit Breaker 模式，连续失败后自动断开，冷却后试探恢复 |
| **Token 守卫** | truncate / compress / paginate 三种策略，防止请求被拒绝 |
| **优雅降级** | RAG/记忆失败不阻断主流程，LLM 失败有 fallback 回复 |
| **WebSocket** | 错误以事件帧返回，单条消息失败不断开连接 |
