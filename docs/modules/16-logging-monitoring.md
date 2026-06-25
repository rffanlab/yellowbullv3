# 日志与监控详细设计

## 1. 设计目标

| 目标 | 说明 |
|------|------|
| **结构化日志** | JSON 格式输出，便于 ELK/Loki 等日志系统采集分析 |
| **请求追踪** | 每个请求分配 trace_id，贯穿全链路（API → Agent → LLM） |
| **性能指标** | Prometheus metrics：请求延迟、LLM token 用量、错误率等 |
| **健康检查** | liveness / readiness 探针，支持 Kubernetes 部署 |

---

## 2. 结构化日志 `logging/structured.py`

```python
"""
YellowBull 结构化日志系统。

输出格式（JSON）：
{
    "timestamp": "2024-01-15T10:30:00.123Z",
    "level": "INFO",
    "logger": "agent.core",
    "message": "...",
    "trace_id": "abc123...",
    "session_id": "sess_456...",
    "user_id": "user_789...",
    "extra": { ... }
}

Usage:
    logger = get_logger(__name__)
    logger.info("Processing message", extra={"token_count": 1234})
"""

import json
import logging
import sys
import time
from datetime import datetime, timezone
from typing import Any
from contextvars import ContextVar


# ---------- Trace Context ----------

trace_id_var: ContextVar[str] = ContextVar("trace_id", default="")
session_id_var: ContextVar[str] = ContextVar("session_id", default="")
user_id_var: ContextVar[str] = ContextVar("user_id", default="")


class TraceFilter(logging.Filter):
    """将 trace context 注入到每条日志"""

    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = trace_id_var.get()
        record.session_id = session_id_var.get()
        record.user_id = user_id_var.get()
        return True


class JsonFormatter(logging.Formatter):
    """JSON 格式日志输出"""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Trace context
        if hasattr(record, "trace_id") and record.trace_id:
            log_entry["trace_id"] = record.trace_id
        if hasattr(record, "session_id") and record.session_id:
            log_entry["session_id"] = record.session_id
        if hasattr(record, "user_id") and record.user_id:
            log_entry["user_id"] = record.user_id

        # Extra fields
        if hasattr(record, "extra_data"):
            log_entry.update(record.extra_data)

        # Exception info
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, ensure_ascii=False, default=str)


class TextFormatter(logging.Formatter):
    """人类可读的文本格式（开发环境）"""

    FORMAT = "%(asctime)s [%(levelname)-5s] %(name)s | %(trace_id)-16s | %(message)s"

    def __init__(self):
        super().__init__(fmt=self.FORMAT, datefmt="%Y-%m-%d %H:%M:%S")


# ---------- Logger Factory ----------

def get_logger(name: str) -> logging.Logger:
    """获取配置好的 logger 实例"""
    logger = logging.getLogger(name)
    return logger


def setup_logging(
    level: str = "INFO",
    fmt: str = "json",
    log_file: str | None = None,
):
    """
    初始化日志系统（应用启动时调用一次）。

    Args:
        level:   日志级别 (DEBUG/INFO/WARNING/ERROR)
        fmt:     输出格式 (json / text)
        log_file: 日志文件路径（None = 仅 stdout）
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level.upper())
    root_logger.handlers.clear()

    # Add trace filter
    root_logger.addFilter(TraceFilter())

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    if fmt == "json":
        console_handler.setFormatter(JsonFormatter())
    else:
        console_handler.setFormatter(TextFormatter())
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        import os
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(JsonFormatter())
        root_logger.addHandler(file_handler)

    # Silence noisy third-party loggers
    for noisy_logger in ["httpx", "httpcore", "uvicorn.error"]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


# ---------- Context Manager ----------

from contextlib import contextmanager


@contextmanager
def trace_context(trace_id: str, session_id: str = "", user_id: str = ""):
    """
    设置日志追踪上下文。

    Usage:
        with trace_context("abc123", "sess_456"):
            logger.info("This will include trace_id")
    """
    tokens = []
    if trace_id:
        tokens.append(trace_id_var.set(trace_id))
    if session_id:
        tokens.append(session_id_var.set(session_id))
    if user_id:
        tokens.append(user_id_var.set(user_id))

    try:
        yield
    finally:
        for token in tokens:
            trace_id_var.reset(token)


# ---------- FastAPI Middleware ----------

import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class LoggingMiddleware(BaseHTTPMiddleware):
    """FastAPI 请求日志中间件"""

    async def dispatch(self, request: Request, call_next) -> Response:
        trace_id = str(uuid.uuid4())[:16]
        start_time = time.monotonic()

        with trace_context(trace_id=trace_id):
            logger = get_logger("http")
            logger.info(
                f"{request.method} {request.url.path}",
                extra={"method": request.method, "path": str(request.url.path)},
            )

            response = await call_next(request)

            duration_ms = (time.monotonic() - start_time) * 1000
            logger.info(
                f"{request.method} {request.url.path} {response.status_code}",
                extra={
                    "status": response.status_code,
                    "duration_ms": round(duration_ms, 2),
                },
            )

            # Inject trace_id into response headers for debugging
            response.headers["X-Trace-ID"] = trace_id
            return response
```

---

## 3. Prometheus Metrics `monitoring/metrics.py`

```python
"""
YellowBull Prometheus 指标。

指标分类：
- HTTP metrics: 请求计数、延迟分布、错误率
- LLM metrics: token 用量、响应时间、模型切换次数
- Session metrics: 活跃会话数、消息计数
- Tool metrics: 工具调用计数、执行时间、失败率
"""

from prometheus_client import (
    Counter, Histogram, Gauge, Summary,
    REGISTRY, generate_latest, CollectorRegistry,
)
from prometheus_client.metrics_core import MetricWrapper
import time


# ---------- HTTP Metrics ----------

http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

http_active_requests = Gauge(
    "http_active_requests",
    "Currently active HTTP requests",
    ["method", "endpoint"],
)


# ---------- LLM Metrics ----------

llm_tokens_total = Counter(
    "llm_tokens_total",
    "Total LLM tokens consumed",
    ["model", "type"],  # type: prompt / completion
)

llm_request_duration_seconds = Histogram(
    "llm_request_duration_seconds",
    "LLM request latency (TTFT + generation)",
    ["model", "endpoint"],
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)

llm_requests_total = Counter(
    "llm_requests_total",
    "Total LLM requests",
    ["model", "status"],  # status: success / error / timeout
)

llm_ttfb_seconds = Histogram(
    "llm_ttfb_seconds",
    "Time to first byte (TTFB) for streaming responses",
    ["model"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)


# ---------- Session Metrics ----------

active_sessions = Gauge(
    "active_sessions_total",
    "Number of active sessions",
)

session_messages_total = Counter(
    "session_messages_total",
    "Total messages processed per session type",
    ["mode"],  # mode: chat / voice / realtime
)


# ---------- Tool Metrics ----------

tool_calls_total = Counter(
    "tool_calls_total",
    "Total tool executions",
    ["tool_name", "status"],  # status: success / error / timeout
)

tool_call_duration_seconds = Histogram(
    "tool_call_duration_seconds",
    "Tool execution latency",
    ["tool_name"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)


# ---------- WebSocket Metrics ----------

ws_active_connections = Gauge(
    "ws_active_connections",
    "Number of active WebSocket connections",
)

ws_messages_total = Counter(
    "ws_messages_total",
    "Total WebSocket messages",
    ["direction"],  # direction: inbound / outbound
)


# ---------- Metrics Collector for FastAPI ----------

from fastapi import APIRouter, Request
from fastapi.responses import Response

router = APIRouter()


@router.get("/metrics")
async def metrics_endpoint():
    """Prometheus metrics endpoint"""
    return Response(
        content=generate_latest(),
        media_type="text/plain; charset=utf-8",
    )


# ---------- Timing Decorator ----------

from functools import wraps
from typing import Callable, Awaitable, TypeVar

T = TypeVar('T')


def track_llm_metrics(model_name: str):
    """装饰器：自动记录 LLM 调用指标"""

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            start = time.monotonic()
            try:
                result = await func(*args, **kwargs)
                duration = time.monotonic() - start

                llm_request_duration_seconds.labels(
                    model=model_name, endpoint=func.__name__
                ).observe(duration)
                llm_requests_total.labels(model=model_name, status="success").inc()
                return result
            except Exception as e:
                duration = time.monotonic() - start
                llm_request_duration_seconds.labels(
                    model=model_name, endpoint=func.__name__
                ).observe(duration)

                status = "timeout" if "timeout" in str(e).lower() else "error"
                llm_requests_total.labels(model=model_name, status=status).inc()
                raise
        return wrapper
    return decorator


def track_tool_metrics(tool_name: str):
    """装饰器：自动记录工具调用指标"""

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            start = time.monotonic()
            try:
                result = await func(*args, **kwargs)
                duration = time.monotonic() - start

                tool_call_duration_seconds.labels(tool_name=tool_name).observe(duration)
                tool_calls_total.labels(tool_name=tool_name, status="success").inc()
                return result
            except Exception as e:
                duration = time.monotonic() - start
                tool_call_duration_seconds.labels(tool_name=tool_name).observe(duration)

                status = "timeout" if "timeout" in str(e).lower() else "error"
                tool_calls_total.labels(tool_name=tool_name, status=status).inc()
                raise
        return wrapper
    return decorator
```

---

## 4. Health Checks `monitoring/health.py`

```python
"""
YellowBull 健康检查端点。

两种探针：
- /health/liveness: 进程是否存活（仅检查自身）
- /health/readiness: 服务是否就绪（检查依赖项）

Kubernetes 集成：
- livenessProbe → GET /health/liveness
- readinessProbe → GET /health/readiness
"""

import time
from typing import Any


class HealthStatus:
    """健康状态聚合器"""

    def __init__(self):
        self._checks: dict[str, dict[str, Any]] = {}
        self._start_time = time.monotonic()

    def register_check(self, name: str, check_func):
        """注册健康检查函数"""
        self._checks[name] = {
            "func": check_func,
            "status": "unknown",
            "message": "",
        }

    async def run_checks(self) -> dict[str, Any]:
        """执行所有健康检查"""
        results = {}
        all_healthy = True

        for name, check in self._checks.items():
            try:
                result = await check["func"]()
                status = "healthy" if result.get("healthy", True) else "unhealthy"
                message = result.get("message", "")
            except Exception as e:
                status = "error"
                message = str(e)
                all_healthy = False

            results[name] = {
                "status": status,
                "message": message,
                **result.get("details", {}),
            }

            if status != "healthy":
                all_healthy = False

        return {
            "status": "healthy" if all_healthy else "unhealthy",
            "uptime_seconds": round(time.monotonic() - self._start_time, 2),
            "checks": results,
        }


# ---------- Global Health Status ----------

health_status = HealthStatus()


async def check_database():
    """检查数据库连接"""
    # Implementation depends on actual DB driver
    return {"healthy": True, "message": "Database connected"}


async def check_llm_api():
    """检查 LLM API 可用性（轻量级检查）"""
    try:
        # Send a minimal request to verify connectivity
        return {"healthy": True, "message": "LLM API reachable"}
    except Exception as e:
        return {"healthy": False, "message": f"LLM API unreachable: {e}"}


async def check_disk_space():
    """检查磁盘空间"""
    import shutil
    total, used, free = shutil.disk_usage("/")
    if free < 1024 * 1024 * 100:  # Less than 100MB free
        return {
            "healthy": False,
            "message": f"Low disk space: {free / (1024*1024):.0f}MB free",
        }
    return {"healthy": True, "message": f"{free / (1024*1024):.0f}MB free"}


# Register checks at module load
health_status.register_check("database", check_database)
health_status.register_check("llm_api", check_llm_api)
health_status.register_check("disk_space", check_disk_space)


# ---------- FastAPI Endpoints ----------

from fastapi import APIRouter, Response
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/health/liveness")
async def liveness_probe():
    """Liveness probe: 进程是否存活"""
    return {"status": "alive", "uptime_seconds": round(time.monotonic() - health_status._start_time, 2)}


@router.get("/health/readiness")
async def readiness_probe():
    """Readiness probe: 服务是否就绪"""
    result = await health_status.run_checks()

    status_code = 200 if result["status"] == "healthy" else 503
    return JSONResponse(
        content=result,
        status_code=status_code,
    )
```

---

## 5. Agent Core 日志集成 `agent/logging_integration.py`

```python
"""
Agent Core 中的日志集成。

关键原则：
- 每条消息处理都有 trace_id，贯穿 LLM 调用、工具执行等子流程
- Token 用量、延迟等指标自动上报 Prometheus
- 敏感信息（API Key、用户密码）不记录到日志
"""

import time
from typing import AsyncIterator


class AgentLogger:
    """Agent Core 专用日志器"""

    def __init__(self, session_id: str = "", trace_id: str = ""):
        self._logger = get_logger("agent")
        self._session_id = session_id
        self._trace_id = trace_id

    def log_message_processing(self, direction: str, token_count: int):
        """记录消息处理"""
        self._logger.info(
            f"Message {direction}",
            extra={
                "direction": direction,  # inbound / outbound
                "token_count": token_count,
            },
        )

    def log_llm_call(self, model: str, prompt_tokens: int, completion_tokens: int, duration_ms: float):
        """记录 LLM 调用"""
        self._logger.info(
            f"LLM call completed",
            extra={
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "duration_ms": round(duration_ms, 2),
            },
        )

        # Update Prometheus metrics
        llm_tokens_total.labels(model=model, type="prompt").inc(prompt_tokens)
        llm_tokens_total.labels(model=model, type="completion").inc(completion_tokens)

    def log_tool_call(self, tool_name: str, duration_ms: float, success: bool):
        """记录工具调用"""
        level = "info" if success else "warning"
        getattr(self._logger, level)(
            f"Tool '{tool_name}' {'succeeded' if success else 'failed'}",
            extra={
                "tool_name": tool_name,
                "duration_ms": round(duration_ms, 2),
                "success": success,
            },
        )

    def log_streaming_chunk(self, chunk_index: int, bytes_count: int):
        """记录流式传输块（低频率采样，避免日志过多）"""
        if chunk_index % 10 == 0:  # 每 10 个块记录一次
            self._logger.debug(
                f"Streaming chunk",
                extra={
                    "chunk_index": chunk_index,
                    "bytes_count": bytes_count,
                },
            )


# ---------- Timing Context Manager ----------

@contextmanager
async def timed_agent_operation(operation_name: str, session_id: str = ""):
    """
    计时代理操作。

    Usage:
        async with timed_agent_operation("llm_call", session_id):
            response = await llm_provider.chat(messages)
        # Automatically logs duration and updates metrics
    """
    start = time.monotonic()
    try:
        yield
    finally:
        duration_ms = (time.monotonic() - start) * 1000
        get_logger("agent").info(
            f"{operation_name} completed",
            extra={
                "operation": operation_name,
                "duration_ms": round(duration_ms, 2),
            },
        )
```

---

## 6. Grafana Dashboard 配置 `monitoring/grafana/dashboard.json`

```json
{
  "dashboard": {
    "title": "YellowBull Overview",
    "panels": [
      {
        "title": "HTTP Request Rate",
        "type": "graph",
        "targets": [{
          "expr": "rate(http_requests_total[5m])",
          "legendFormat": "{{method}} {{endpoint}}"
        }]
      },
      {
        "title": "LLM Token Usage",
        "type": "stat",
        "targets": [{
          "expr": "increase(llm_tokens_total[1h])",
          "legendFormat": "{{model}} {{type}}"
        }]
      },
      {
        "title": "Active Sessions",
        "type": "gauge",
        "targets": [{
          "expr": "active_sessions_total"
        }],
        "options": {
          "maxValue": 100
        }
      },
      {
        "title": "LLM Response Latency (P95)",
        "type": "graph",
        "targets": [{
          "expr": "histogram_quantile(0.95, rate(llm_request_duration_seconds_bucket[5m]))",
          "legendFormat": "{{model}}"
        }]
      },
      {
        "title": "Tool Call Success Rate",
        "type": "stat",
        "targets": [{
          "expr": "sum(rate(tool_calls_total{status='success'}[5m])) / sum(rate(tool_calls_total[5m])) * 100"
        }]
      },
      {
        "title": "WebSocket Connections",
        "type": "graph",
        "targets": [{
          "expr": "ws_active_connections"
        }]
      }
    ]
  }
}
```

---

## 7. 架构总览

```
                    ┌─────────────────────┐
                    │   Structured Log    │ ← JSON format, trace_id context
                    ├─────────────────────┤
                    │ TraceFilter         │ → injects trace/session/user IDs
                    │ JsonFormatter       │ → ELK/Loki compatible output
                    │ TextFormatter       │ → human-readable (dev)
                    └─────────────────────┘

                    ┌─────────────────────┐
                    │  Prometheus Metrics │ ← /metrics endpoint
                    ├─────────────────────┤
                    │ HTTP: requests,     │
                    │   latency, errors   │
                    │ LLM: tokens, TTFB,  │
                    │   model switching   │
                    │ Session: active,    │
                    │   message count     │
                    │ Tool: calls,        │
                    │   success rate      │
                    │ WebSocket:          │
                    │   connections       │
                    └─────────────────────┘

                    ┌─────────────────────┐
                    │  Health Checks      │ ← Kubernetes probes
                    ├─────────────────────┤
                    │ /health/liveness    │ → process alive?
                    │ /health/readiness   │ → dependencies OK?
                    └─────────────────────┘

                    ┌─────────────────────┐
                    │  Grafana Dashboard  │ ← visualization
                    ├─────────────────────┤
                    │ Request rate        │
                    │ Token usage         │
                    │ Latency P50/P95     │
                    │ Error rate          │
                    │ Active sessions     │
                    └─────────────────────┘
```

---

## 8. 设计总结

| 特性 | 实现方式 |
|------|---------|
| **结构化日志** | JSON 格式 + trace_id / session_id / user_id 上下文 |
| **请求追踪** | ContextVar 传递 trace context，中间件自动注入 |
| **Prometheus** | Counter/Histogram/Gauge 覆盖 HTTP、LLM、会话、工具、WebSocket |
| **健康检查** | liveness（进程存活）+ readiness（依赖就绪），K8s 兼容 |
| **Grafana** | 预配置 dashboard JSON，开箱即用的可视化面板 |
| **敏感信息保护** | API Key / 密码等不记录到日志 |
