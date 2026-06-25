# 可观测性详细设计（Observability）

## 1. 职责边界

| 支柱 | 说明 |
|------|------|
| **日志** | 结构化 JSON 日志，分级输出，支持采样 |
| **指标** | Prometheus 风格计数器、计时器、直方图 |
| **追踪** | OpenTelemetry 风格的分布式链路追踪 |
| **健康检查** | liveness / readiness probe，系统状态报告 |

---

## 2. 结构化日志 `observability/logging.py`

```python
"""
结构化 JSON 日志。

输出格式：
    {
        "timestamp": "2025-01-15T10:30:00.123Z",
        "level": "INFO",
        "logger": "agent.core",
        "message": "Agent started processing",
        "session_id": "sess_abc123",
        "trace_id": "trace_xyz789",
        "extra": { ... }
    }
"""

import json
import logging
import sys
import time
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

current_trace_id: ContextVar[str] = ContextVar("trace_id", default="")
current_session_id: ContextVar[str] = ContextVar("session_id", default="")


class JSONFormatter(logging.Formatter):
    """JSON 格式日志格式化器"""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "trace_id": current_trace_id.get(),
            "session_id": current_session_id.get(),
        }

        if hasattr(record, "extra_data"):
            log_entry["extra"] = record.extra_data

        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, ensure_ascii=False, default=str)


class SamplingFilter(logging.Filter):
    """日志采样过滤器"""

    def __init__(self, rates: dict[str, float] | None = None):
        super().__init__()
        self._rates = rates or {"DEBUG": 0.1, "INFO": 1.0, "WARNING": 1.0, "ERROR": 1.0}

    def filter(self, record: logging.LogRecord) -> bool:
        import random
        rate = self._rates.get(record.levelname, 1.0)
        return random.random() < rate


def setup_logging(level: str = "INFO", json_format: bool = True, sampling: bool = False):
    """配置全局日志"""
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    root_logger.handlers.clear()

    formatter = JSONFormatter() if json_format else logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    if sampling:
        root_logger.addFilter(SamplingFilter())


class StructuredLogger:
    """结构化日志包装器"""

    def __init__(self, name: str):
        self._logger = logging.getLogger(name)

    def _log(self, level: int, message: str, **kwargs):
        extra_record = logging.LogRecord(
            name=self._logger.name, level=level, pathname="", lineno=0,
            msg=message, args=None, exc_info=None,
        )
        extra_record.extra_data = kwargs
        self._logger.handle(extra_record)

    def info(self, message: str, **kwargs):
        self._log(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs):
        self._log(logging.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs):
        self._log(logging.ERROR, message, **kwargs)

    def debug(self, message: str, **kwargs):
        self._log(logging.DEBUG, message, **kwargs)
```

---

## 3. 指标系统 `observability/metrics.py`

```python
"""
Prometheus 风格的指标系统。

支持：Counter（计数器）、Gauge（仪表盘）、Histogram（直方图）、Summary（摘要）
"""

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Counter:
    """计数器：只增不减"""
    name: str
    description: str = ""
    _value: float = 0.0
    _labels: dict[str, float] = field(default_factory=dict)

    def inc(self, value: float = 1.0, **labels):
        key = tuple(sorted(labels.items())) if labels else ()
        if key:
            self._labels[key] = self._labels.get(key, 0.0) + value
        else:
            self._value += value

    def get(self, **labels) -> float:
        key = tuple(sorted(labels.items())) if labels else ()
        return self._labels.get(key, self._value) if key else self._value


@dataclass
class Gauge:
    """仪表盘：可增可减"""
    name: str
    description: str = ""
    _value: float = 0.0
    _labels: dict[str, float] = field(default_factory=dict)

    def set(self, value: float, **labels):
        key = tuple(sorted(labels.items())) if labels else ()
        if key:
            self._labels[key] = value
        else:
            self._value = value

    def inc(self, value: float = 1.0, **labels):
        key = tuple(sorted(labels.items())) if labels else ()
        if key:
            self._labels[key] = self._labels.get(key, 0.0) + value
        else:
            self._value += value

    def dec(self, value: float = 1.0, **labels):
        key = tuple(sorted(labels.items())) if labels else ()
        if key:
            self._labels[key] = self._labels.get(key, 0.0) - value
        else:
            self._value -= value

    def get(self, **labels) -> float:
        key = tuple(sorted(labels.items())) if labels else ()
        return self._labels.get(key, self._value) if key else self._value


@dataclass
class Histogram:
    """直方图：记录值分布"""
    name: str
    description: str = ""
    buckets: list[float] = field(default_factory=lambda: [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0])
    _observations: dict[str, list[float]] = field(default_factory=dict)

    def observe(self, value: float, **labels):
        key = tuple(sorted(labels.items())) if labels else ("__default__",)
        if key not in self._observations:
            self._observations[key] = []
        self._observations[key].append(value)

    def get_buckets(self, **labels) -> dict[str, int]:
        key = tuple(sorted(labels.items())) if labels else ("__default__",)
        values = self._observations.get(key, [])
        result = {}
        for bucket in self.buckets:
            result[f"le_{bucket}"] = sum(1 for v in values if v <= bucket)
        result["le_+Inf"] = len(values)
        return result

    def get_summary(self, **labels) -> dict[str, float]:
        key = tuple(sorted(labels.items())) if labels else ("__default__",)
        values = self._observations.get(key, [])
        if not values:
            return {"count": 0, "sum": 0.0, "avg": 0.0}
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        return {
            "count": n,
            "sum": sum(sorted_vals),
            "avg": sum(sorted_vals) / n,
            "p50": sorted_vals[n // 2],
            "p95": sorted_vals[int(n * 0.95)],
            "p99": sorted_vals[int(n * 0.99)],
        }


class MetricsRegistry:
    """指标注册表"""

    def __init__(self):
        self._counters: dict[str, Counter] = {}
        self._gauges: dict[str, Gauge] = {}
        self._histograms: dict[str, Histogram] = {}

    def counter(self, name: str, description: str = "") -> Counter:
        if name not in self._counters:
            self._counters[name] = Counter(name, description)
        return self._counters[name]

    def gauge(self, name: str, description: str = "") -> Gauge:
        if name not in self._gauges:
            self._gauges[name] = Gauge(name, description)
        return self._gauges[name]

    def histogram(self, name: str, description: str = "", buckets: list[float] | None = None) -> Histogram:
        if name not in self._histograms:
            self._histograms[name] = Histogram(name, description, buckets or [])
        return self._histograms[name]

    def export_prometheus(self) -> str:
        """导出为 Prometheus 文本格式"""
        lines = []
        for name, counter in self._counters.items():
            lines.append(f"# HELP {name} {counter.description}")
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name} {counter.get()}")

        for name, gauge in self._gauges.items():
            lines.append(f"# HELP {name} {gauge.description}")
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {gauge.get()}")

        for name, hist in self._histograms.items():
            summary = hist.get_summary()
            lines.append(f"# HELP {name}_seconds {hist.description}")
            lines.append(f"# TYPE {name}_seconds histogram")
            lines.append(f"{name}_count {summary['count']}")
            lines.append(f"{name}_sum {summary['sum']:.3f}")

        return "\n".join(lines)


# 全局注册表
default_registry = MetricsRegistry()


class Timer:
    """上下文管理器：自动记录耗时"""

    def __init__(self, histogram: Histogram, **labels):
        self._histogram = histogram
        self._labels = labels
        self._start_time: float = 0

    async def __aenter__(self):
        self._start_time = time.time()
        return self

    async def __aexit__(self, *args):
        duration = time.time() - self._start_time
        self._histogram.observe(duration, **self._labels)


class TimingDecorator:
    """装饰器：自动记录函数耗时"""

    def __init__(self, registry: MetricsRegistry, metric_name: str | None = None):
        self._registry = registry
        self._metric_name = metric_name

    def __call__(self, func):
        import functools
        name = self._metric_name or f"{func.__module__}.{func.__name__}"
        histogram = self._registry.histogram(name, f"Execution time of {name}")

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start
                histogram.observe(duration)

        return wrapper
```

---

## 4. 分布式追踪 `observability/tracing.py`

```python
"""
OpenTelemetry 风格的链路追踪。

核心概念：
- Trace: 一次完整请求的生命周期（trace_id）
- Span: 一个操作单元（span_id, parent_span_id）
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Span:
    """追踪跨度"""
    trace_id: str
    span_id: str
    parent_span_id: str | None
    name: str
    start_time: float
    end_time: float = 0.0
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    status: str = "OK"           # OK | ERROR
    error_message: str | None = None

    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000 if self.end_time else 0


class TracerProvider:
    """追踪提供者"""

    def __init__(self, service_name: str = "agent-platform"):
        self._service_name = service_name
        self._spans: list[Span] = []
        self._span_counter = 0

    @property
    def spans(self) -> list[Span]:
        return self._spans

    def start_span(
        self, name: str, trace_id: str | None = None, parent_span_id: str | None = None
    ) -> Span:
        """开始一个新的 span"""
        from observability.logging import current_trace_id, current_session_id

        if not trace_id:
            # 尝试从上下文获取
            ctx_trace = current_trace_id.get()
            trace_id = ctx_trace if ctx_trace else uuid.uuid4().hex[:16]
            current_trace_id.set(trace_id)

        self._span_counter += 1
        span_id = f"span_{self._span_counter:08d}"

        span = Span(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            name=name,
            start_time=time.time(),
        )
        self._spans.append(span)
        return span

    def end_span(self, span: Span, error: str | None = None):
        """结束一个 span"""
        span.end_time = time.time()
        if error:
            span.status = "ERROR"
            span.error_message = error

    @asynccontextmanager
    async def trace(self, name: str, **attributes):
        """异步上下文管理器：自动管理 span 生命周期"""
        from observability.logging import current_trace_id

        span = self.start_span(
            name=name,
            trace_id=current_trace_id.get() or None,
        )
        span.attributes.update(attributes)

        try:
            yield span
        except Exception as e:
            self.end_span(span, error=str(e))
            raise
        else:
            self.end_span(span)

    def get_trace(self, trace_id: str) -> list[Span]:
        """获取某个 trace 的所有 spans"""
        return [s for s in self._spans if s.trace_id == trace_id]

    def export_json(self, trace_id: str | None = None) -> list[dict]:
        """导出为 JSON 格式"""
        spans = self.get_trace(trace_id) if trace_id else self._spans
        return [
            {
                "trace_id": s.trace_id,
                "span_id": s.span_id,
                "parent_span_id": s.parent_span_id,
                "name": s.name,
                "start_time": s.start_time,
                "duration_ms": round(s.duration_ms, 2),
                "status": s.status,
                "attributes": s.attributes,
            }
            for s in spans
        ]


# 全局 tracer
default_tracer = TracerProvider()
```

---

## 5. 健康检查 `observability/health.py`

```python
"""
系统健康检查。

提供：
- liveness probe: 进程是否存活
- readiness probe: 服务是否就绪（依赖组件是否正常）
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)


@dataclass
class ComponentHealth:
    """组件健康状态"""
    name: str
    status: str = "healthy"       # healthy | degraded | unhealthy
    latency_ms: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class HealthChecker:
    """健康检查器"""

    def __init__(self):
        self._checks: list[tuple[str, Callable[..., Coroutine[Any, Any, ComponentHealth]]]] = []

    def register_check(
        self, name: str, check_fn: Callable[..., Coroutine[Any, Any, ComponentHealth]]
    ):
        """注册健康检查函数"""
        self._checks.append((name, check_fn))

    async def check_all(self) -> dict[str, Any]:
        """执行所有健康检查"""
        start_time = time.time()
        components = []
        overall_status = "healthy"

        for name, check_fn in self._checks:
            try:
                result = await asyncio.wait_for(check_fn(), timeout=5.0)
                components.append(result)
                if result.status == "unhealthy":
                    overall_status = "unhealthy"
                elif result.status == "degraded" and overall_status != "unhealthy":
                    overall_status = "degraded"
            except Exception as e:
                logger.error(f"Health check '{name}' failed: {e}")
                components.append(ComponentHealth(name=name, status="unhealthy", error=str(e)))
                overall_status = "unhealthy"

        return {
            "status": overall_status,
            "timestamp": time.time(),
            "check_duration_ms": (time.time() - start_time) * 1000,
            "components": [
                {
                    "name": c.name,
                    "status": c.status,
                    "latency_ms": round(c.latency_ms, 2),
                    "details": c.details,
                    "error": c.error,
                }
                for c in components
            ],
        }

    async def liveness(self) -> dict[str, Any]:
        """Liveness probe（简单存活检查）"""
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "pid": asyncio.current_task().get_name() if asyncio.current_task() else "unknown",
        }


# 示例：常见健康检查函数
async def check_database(db: Any) -> ComponentHealth:
    """数据库连接检查"""
    start = time.time()
    try:
        await db.execute("SELECT 1")
        latency = (time.time() - start) * 1000
        return ComponentHealth(name="database", status="healthy", latency_ms=latency)
    except Exception as e:
        return ComponentHealth(
            name="database", status="unhealthy",
            latency_ms=(time.time() - start) * 1000, error=str(e),
        )


async def check_llm_provider(provider: Any) -> ComponentHealth:
    """LLM 提供商检查"""
    start = time.time()
    try:
        await provider.health_check()
        latency = (time.time() - start) * 1000
        return ComponentHealth(name="llm_provider", status="healthy", latency_ms=latency)
    except Exception as e:
        return ComponentHealth(
            name="llm_provider", status="degraded",
            latency_ms=(time.time() - start) * 1000, error=str(e),
        )
```

---

## 6. YAML 配置 `config/observability.yaml`

```yaml
observability:
  logging:
    level: "INFO"
    json_format: true
    sampling:
      enabled: false
      rates:
        DEBUG: 0.1
        INFO: 1.0

  metrics:
    enabled: true
    export_interval_seconds: 15
    prometheus_endpoint: "/metrics"

  tracing:
    enabled: true
    sample_rate: 0.1             # 生产环境采样率（10%）
    service_name: "agent-platform"
    exporters:
      - type: "console"           # console | jaeger | zipkin | otel
        format: "json"

  health:
    liveness_endpoint: "/health/live"
    readiness_endpoint: "/health/ready"
    check_interval_seconds: 30
```

---

## 7. 架构总览

```
                    ┌─────────────────────┐
                    │     Agent Core      │
                    │                     │
                    │  StructuredLogger   │──→ JSON logs (stdout / file)
                    │  MetricsRegistry    │──→ Prometheus metrics (/metrics)
                    │  TracerProvider     │──→ Distributed traces (JSON / Jaeger)
                    └──────────┬──────────┘
                               │
              ┌────────────────┴────────────────┐
              ▼                                  ▼
   ┌─────────────────────┐          ┌─────────────────────┐
   │    Logging Layer     │          │   Metrics Layer      │
   │                     │          │                       │
   │ • JSON format       │          │ • Counter (API calls) │
   │ • Context vars      │          │ • Gauge (active sess) │
   │ • Sampling filter   │          │ • Histogram (latency) │
   └─────────────────────┘          └──────────┬───────────┘
                                                │
              ┌─────────────────────────────────┤
              ▼                                 ▼
   ┌─────────────────────┐          ┌─────────────────────┐
   │    Tracing Layer     │          │  Health Check Layer  │
   │                     │          │                       │
   │ • Trace + Span      │          │ • Liveness probe      │
   │ • Async context mgr │          │ • Readiness probe     │
   │ • JSON export       │          │ • Component checks    │
   └─────────────────────┘          └───────────────────────┘
```

---

## 8. 设计总结

| 特性 | 实现方式 |
|------|---------|
| **结构化日志** | JSON Formatter + ContextVar（trace_id/session_id）+ 采样过滤器 |
| **指标系统** | Counter / Gauge / Histogram，支持 Prometheus 格式导出 |
| **链路追踪** | Trace/Span 模型，async context manager 自动管理生命周期 |
| **健康检查** | Liveness + Readiness probe，组件级健康状态报告 |
