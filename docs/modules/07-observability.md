# 可观测性设计（Observability）

## 1. 职责边界

| 支柱 | 说明 |
|------|------|
| **结构化日志** | JSON 格式，含 trace_id、session_id、level、timestamp |
| **指标采集** | Prometheus metrics：请求量、延迟、LLM token 用量、工具调用统计 |
| **链路追踪** | OpenTelemetry tracing，跨模块 trace_id 传递 |
| **健康检查** | `/health` + `/ready` 端点 |

---

## 2. 结构化日志 `observability/logging.py`

```python
"""
JSON structured logging。

格式：
{
    "timestamp": "2026-01-01T00:00:00Z",
    "level": "INFO",
    "message": "...",
    "module": "agent.core",
    "trace_id": "abc-123",        # 链路追踪 ID（可选）
    "session_id": "uuid...",      # 会话 ID（可选）
    "duration_ms": 123.4,         # 耗时（可选）
}

配置来源：YAML logging.* → Python logging.config.dictConfig
"""

import json
import logging
import sys
from datetime import datetime, timezone
from logging import LogRecord


class JsonFormatter(logging.Formatter):
    """JSON 格式日志输出器"""

    def format(self, record: LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module or "",
            "function": record.funcName or "",
            "line": record.lineno,
        }

        # 附加 trace_id / session_id（通过 extra 参数传入）
        if hasattr(record, "trace_id"):
            log_entry["trace_id"] = record.trace_id
        if hasattr(record, "session_id"):
            log_entry["session_id"] = record.session_id
        if hasattr(record, "duration_ms"):
            log_entry["duration_ms"] = record.duration_ms

        # 异常信息
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, ensure_ascii=False)


def setup_logging(level: str = "INFO", format: str = "json"):
    """根据配置初始化全局 logging"""
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    handler = logging.StreamHandler(sys.stdout)
    if format == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        ))

    root.addHandler(handler)

    # 降低第三方库日志级别
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)
```

---

## 3. Prometheus 指标 `observability/metrics.py`

```python
"""
Prometheus metrics for Agent monitoring。

Metrics:
- http_requests_total (counter)          # HTTP 请求总量，按 method/path/status 分桶
- http_request_duration_seconds (histogram)  # 请求延迟分布
- llm_tokens_total (counter)            # LLM token 用量（input/output）
- llm_calls_total (counter)             # LLM 调用次数，按 provider/model 标签
- tool_calls_total (counter)            # 工具调用次数，按 tool_name/success 标签
- tool_call_duration_seconds (histogram)     # 工具执行延迟
- session_active_gauge (gauge)          # 活跃会话数
"""

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
from fastapi import APIRouter

router = APIRouter(tags=["Observability"])


# ---- HTTP Metrics ----
http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"],
)

http_request_duration = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "path"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)


# ---- LLM Metrics ----
llm_tokens_total = Counter(
    "llm_tokens_total",
    "Total LLM tokens consumed",
    ["provider", "model", "type"],   # type: input | output
)

llm_calls_total = Counter(
    "llm_calls_total",
    "Total LLM API calls",
    ["provider", "model", "status"],  # status: success | error
)


# ---- Tool Metrics ----
tool_calls_total = Counter(
    "tool_calls_total",
    "Total tool executions",
    ["tool_name", "status"],          # status: success | error
)

tool_call_duration = Histogram(
    "tool_call_duration_seconds",
    "Tool execution duration in seconds",
    ["tool_name"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)


# ---- Session Metrics ----
session_active_gauge = Gauge(
    "session_active_count",
    "Number of active sessions",
)


@router.get("/metrics")
async def prometheus_metrics():
    """Prometheus scrape endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
```

### 3.1 指标注入点

| 模块 | 指标 | 注入位置 |
|------|------|---------|
| API Server | `http_requests_total` | Middleware（请求进入/响应返回） |
| API Server | `http_request_duration` | Middleware |
| LLM Provider | `llm_tokens_total` | `chat()` / `chat_stream()` 回调 |
| LLM Provider | `llm_calls_total` | `chat()` / `chat_stream()` 入口/出口 |
| Tool Registry | `tool_calls_total` | `execute()` wrapper |
| Tool Registry | `tool_call_duration` | `execute()` wrapper |
| Session Manager | `session_active_gauge` | `create_session()` / `delete_session()` |

---

## 4. OpenTelemetry 链路追踪 `observability/tracing.py`

```python
"""
OpenTelemetry tracing integration。

Spans:
- http.request (API Server)
    ├── llm.chat (LLM Provider)
    │   └── http.call (external API call to OpenAI/Anthropic/etc.)
    └── tool.execute.{name} (Tool Registry)

Trace ID 通过请求头传递：X-Trace-ID → logging extra → span context
"""

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter


def setup_tracing(service_name: str = "yellowbull-agent", otlp_endpoint: str | None = None):
    """初始化 OpenTelemetry tracer"""
    provider = TracerProvider()
    trace.set_tracer_provider(provider)

    if otlp_endpoint:
        exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
        processor = BatchSpanProcessor(exporter)
        provider.add_span_processor(processor)

    tracer = trace.get_tracer(service_name)
    return tracer


# ---- Span helpers ----

tracer = None   # 由 main.py 初始化后注入


def set_tracer(t):
    global tracer
    tracer = t


def start_http_span(method: str, path: str):
    """在 HTTP middleware 中创建 span"""
    if not tracer:
        return None
    return tracer.start_span(f"http.{method.lower()}.{path}")


def start_llm_span(provider: str, model: str):
    """在 LLM provider 中创建 span"""
    if not tracer:
        return None
    return tracer.start_span(f"llm.chat.{provider}.{model}")


def start_tool_span(tool_name: str):
    """在 tool registry 中创建 span"""
    if not tracer:
        return None
    return tracer.start_span(f"tool.execute.{tool_name}")
```

---

## 5. 健康检查端点 `api/health.py`

```python
"""
Health & readiness probes。

- /health (liveness): 进程是否存活（简单返回 ok）
- /ready (readiness): 依赖服务是否就绪（DB、LLM API）
"""

from fastapi import APIRouter, HTTPException

router = APIRouter(tags=["Health"])


@router.get("/health")
async def liveness_probe():
    """Liveness probe —— 进程存活检查"""
    return {"status": "ok"}


@router.get("/ready")
async def readiness_probe():
    """Readiness probe —— 依赖就绪检查"""
    checks = {}

    # SQLite check
    try:
        from session.manager import get_session_manager
        sm = get_session_manager()
        db_ok = sm._storage._db is not None and not sm._storage._db.closed
        checks["sqlite"] = "ok" if db_ok else "error"
    except Exception:
        checks["sqlite"] = "error"

    # LLM provider check (ping)
    try:
        from llm.provider_factory import create_provider
        from config.manager import get_manager
        settings = get_manager().settings
        provider = create_provider(settings.llm)
        # 轻量级检查：不实际调用 API，只验证配置完整性
        checks["llm"] = "ok" if provider else "error"
    except Exception:
        checks["llm"] = "error"

    all_ok = all(v == "ok" for v in checks.values())
    status_code = 200 if all_ok else 503

    return {
        "status": "ready" if all_ok else "not_ready",
        "checks": checks,
    }


# K8s 配置示例：
# livenessProbe: httpGet /health port 8000, period=10s
# readinessProbe: httpGet /ready port 8000, initialDelay=5s, period=5s
```

---

## 6. YAML 配置 `observability` section

```yaml
# config/settings.yaml (新增)
observability:
  logging:
    level: "INFO"              # DEBUG | INFO | WARNING | ERROR
    format: "json"             # json | text

  metrics:
    enabled: true
    port: 9090                 # Prometheus scrape port（或复用主端口 /metrics）

  tracing:
    enabled: false             # 生产环境开启
    otlp_endpoint: ""          # Jaeger/Tempo endpoint，如 "jaeger:4317"
```

---

## 7. Grafana Dashboard 面板建议

| Panel | Metric | Aggregation |
|-------|--------|-------------|
| **请求 QPS** | `rate(http_requests_total[1m])` | by path |
| **P95 延迟** | `histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[1m]))` | by path |
| **LLM Token 用量** | `rate(llm_tokens_total[5m])` | by provider/model/type |
| **工具调用成功率** | `sum(rate(tool_calls_total{status="success"}[5m])) / sum(rate(tool_calls_total[5m]))` | by tool_name |
| **活跃会话数** | `session_active_count` | gauge |
| **错误率** | `sum(rate(http_requests_total{status=~"5.."}[5m])) / sum(rate(http_requests_total[5m]))` | — |

---

## 8. 设计总结

| 特性 | 实现方式 |
|------|---------|
| **结构化日志** | JSON formatter，trace_id/session_id 注入 |
| **Prometheus 指标** | Counter + Histogram + Gauge，`/metrics` endpoint |
| **链路追踪** | OpenTelemetry spans，跨模块 trace propagation |
| **健康检查** | `/health` (liveness) + `/ready` (readiness) |
| **Dashboard** | Grafana panels：QPS、延迟、Token 用量、工具成功率 |
