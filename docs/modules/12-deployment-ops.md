# 部署与运维详细设计

## 1. 设计目标

| 目标 | 说明 |
|------|------|
| **灵活部署** | 支持本地开发、Docker 容器化、K8s 编排多种模式 |
| **高可用** | 健康检查、自动重启、负载均衡、故障转移 |
| **可观测性** | 结构化日志、指标采集、分布式追踪 |
| **配置管理** | 环境变量 + YAML 配置文件，支持多环境切换 |

---

## 2. Docker 部署 `docker/`

### Dockerfile

```dockerfile
# ==================== Build Stage ====================
FROM python:3.12-slim AS builder

WORKDIR /build

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ==================== Runtime Stage ====================
FROM python:3.12-slim AS runtime

WORKDIR /app

# 从 builder 复制已安装的包
COPY --from=builder /install /usr/local

# 创建非 root 用户
RUN groupadd -r appuser && useradd -r -g appuser appuser

# 复制应用代码
COPY . .

# 创建数据目录
RUN mkdir -p data/logs data/uploads data/vector_store && chown -R appuser:appuser /app/data

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### docker-compose.yml

```yaml
version: '3.9'

services:
  # ==================== Agent API Service ====================
  agent-api:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - LLM_API_KEY=${LLM_API_KEY}
      - REDIS_URL=redis://redis:6379/0
      - POSTGRES_URL=postgresql://agent:password@postgres:5432/agent_db
      - VECTOR_STORE_TYPE=qdrant
      - QDRANT_URL=http://qdrant:6333
    depends_on:
      redis:
        condition: service_healthy
      postgres:
        condition: service_healthy
      qdrant:
        condition: service_started
    volumes:
      - ./data/logs:/app/data/logs
      - ./data/uploads:/app/data/uploads
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G

  # ==================== Redis (Session + Cache) ====================
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 3

  # ==================== PostgreSQL (持久化存储) ====================
  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: agent_db
      POSTGRES_USER: agent
      POSTGRES_PASSWORD: password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U agent"]
      interval: 10s
      timeout: 5s
      retries: 3

  # ==================== Qdrant (向量存储) ====================
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage
    environment:
      QDRANT__SERVICE__API_KEY: ""

  # ==================== Ollama (本地 LLM，可选) ====================
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    profiles:
      - local-llm

  # ==================== Prometheus (指标采集) ====================
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./config/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    profiles:
      - monitoring

  # ==================== Grafana (可视化) ====================
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./config/grafana/dashboards:/etc/grafana/provisioning/dashboards
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
    depends_on:
      - prometheus
    profiles:
      - monitoring

volumes:
  redis_data:
  postgres_data:
  qdrant_data:
  ollama_data:
  prometheus_data:
  grafana_data:
```

---

## 3. Kubernetes 部署 `k8s/`

### agent-deployment.yaml

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-api
  namespace: yellowbull
spec:
  replicas: 3
  selector:
    matchLabels:
      app: agent-api
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  template:
    metadata:
      labels:
        app: agent-api
        version: v1.0.0
    spec:
      containers:
        - name: agent-api
          image: yellowbull/agent-api:v1.0.0
          ports:
            - containerPort: 8000
              name: http
          envFrom:
            - secretRef:
                name: agent-secrets
          env:
            - name: ENVIRONMENT
              value: "production"
          resources:
            requests:
              cpu: "2"
              memory: 4Gi
            limits:
              cpu: "4"
              memory: 8Gi
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 15
            periodSeconds: 30
            timeoutSeconds: 5
            failureThreshold: 3
          readinessProbe:
            httpGet:
              path: /ready
              port: 8000
            initialDelaySeconds: 10
            periodSeconds: 10
            timeoutSeconds: 3
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
            - weight: 100
              podAffinityTerm:
                labelSelector:
                  matchExpressions:
                    - key: app
                      operator: In
                      values:
                        - agent-api
                topologyKey: kubernetes.io/hostname
---
apiVersion: v1
kind: Service
metadata:
  name: agent-api-service
  namespace: yellowbull
spec:
  selector:
    app: agent-api
  ports:
    - port: 80
      targetPort: 8000
      protocol: TCP
  type: ClusterIP
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: agent-ingress
  namespace: yellowbull
  annotations:
    nginx.ingress.kubernetes.io/proxy-body-size: "50m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "300"
spec:
  ingressClassName: nginx
  rules:
    - host: api.yellowbull.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: agent-api-service
                port:
                  number: 80
```

### HPA (Horizontal Pod Autoscaler)

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: agent-api-hpa
  namespace: yellowbull
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: agent-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Pods
      pods:
        metric:
          name: active_connections_per_pod
        target:
          type: AverageValue
          averageValue: "100"
```

---

## 4. 配置管理 `config/`

### settings.py（应用配置）

```python
"""
统一配置管理。

优先级：环境变量 > .env 文件 > defaults.yaml > 硬编码默认值
"""

import os
from pathlib import Path


class Settings:
    """应用配置类"""

    def __init__(self, env_file: str | None = None):
        # 加载 .env 文件
        if env_file and Path(env_file).exists():
            from dotenv import load_dotenv
            load_dotenv(env_file)

        # ==================== 环境 ====================
        self.environment: str = os.getenv("ENVIRONMENT", "development")
        self.debug: bool = self.environment == "development"

        # ==================== LLM ====================
        self.llm_provider: str = os.getenv("LLM_PROVIDER", "openai")
        self.llm_model: str = os.getenv("LLM_MODEL", "gpt-4o")
        self.llm_api_key: str = os.getenv("LLM_API_KEY", "")
        self.llm_base_url: str | None = os.getenv("LLM_BASE_URL")
        self.llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
        self.llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "4096"))

        # ==================== 服务 ====================
        self.host: str = os.getenv("HOST", "0.0.0.0")
        self.port: int = int(os.getenv("PORT", "8000"))
        self.workers: int = int(os.getenv("WORKERS", "4"))

        # ==================== 数据库 ====================
        self.postgres_url: str = os.getenv(
            "POSTGRES_URL", "postgresql://agent:password@localhost:5432/agent_db"
        )
        self.redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

        # ==================== 向量存储 ====================
        self.vector_store_type: str = os.getenv("VECTOR_STORE_TYPE", "chroma")
        self.qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.chroma_persist_dir: str = os.getenv(
            "CHROMA_PERSIST_DIR", "./data/vector_store"
        )

        # ==================== 会话管理 ====================
        self.session_ttl_seconds: int = int(os.getenv("SESSION_TTL_SECONDS", "86400"))
        self.max_context_turns: int = int(os.getenv("MAX_CONTEXT_TURNS", "50"))

        # ==================== RAG ====================
        self.rag_enabled: bool = os.getenv("RAG_ENABLED", "true").lower() == "true"
        self.rag_top_k: int = int(os.getenv("RAG_TOP_K", "5"))
        self.embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

        # ==================== 安全 ====================
        self.cors_origins: list[str] = self._parse_list(
            os.getenv("CORS_ORIGINS", "*")
        )
        self.rate_limit_per_minute: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))

        # ==================== 日志 ====================
        self.log_level: str = os.getenv("LOG_LEVEL", "INFO")
        self.log_format: str = os.getenv("LOG_FORMAT", "json")

    @staticmethod
    def _parse_list(value: str) -> list[str]:
        """解析逗号分隔的列表"""
        if value == "*":
            return ["*"]
        return [v.strip() for v in value.split(",")]


# 全局配置实例
settings = Settings()
```

### defaults.yaml（默认配置参考）

```yaml
# Agent Core 默认配置
agent:
  name: "YellowBull Assistant"
  system_prompt: |
    你是一个智能助手，由 YellowBull AI 驱动。请友好、准确地回答用户问题。
  max_context_turns: 50
  temperature: 0.7
  top_p: 1.0

# LLM 配置
llm:
  provider: openai
  model: gpt-4o
  api_key: ""  # 通过环境变量设置
  base_url: null
  max_tokens: 4096
  timeout: 60

# RAG 配置
rag:
  enabled: true
  top_k: 5
  embedding_model: text-embedding-3-small
  vector_store: chroma
  chunk_size: 1000
  chunk_overlap: 200

# 工具配置
tools:
  web_search:
    enabled: true
    engine: google
    api_key: ""
  calculator:
    enabled: true
  file_parser:
    enabled: true
    max_file_size_mb: 50
  asr:
    enabled: false
    provider: whisper-api
  tts:
    enabled: false
    provider: openai-tts

# 会话管理
session:
  ttl_seconds: 86400
  storage: redis
  max_turns: 50

# 安全配置
security:
  rate_limit_per_minute: 60
  cors_origins: ["*"]
  api_key_required: false

# 日志配置
logging:
  level: INFO
  format: json
  file: data/logs/agent.log
  max_size_mb: 100
  backup_count: 5
```

---

## 5. 健康检查与监控 `monitoring/`

### health.py

```python
"""
健康检查端点。

提供 /health（存活）和 /ready（就绪）两个端点，供 K8s/LB 使用。
"""

from fastapi import APIRouter, Depends
import time

router = APIRouter()

# 启动时间戳
_start_time = time.monotonic()


@router.get("/health")
async def health_check():
    """存活检查：服务是否还在运行"""
    return {
        "status": "ok",
        "uptime_seconds": round(time.monotonic() - _start_time, 2),
    }


@router.get("/ready")
async def readiness_check(
    db=Depends(get_db_dependency),
    redis=Depends(get_redis_dependency),
):
    """就绪检查：依赖服务是否可用"""
    checks = {}

    # 数据库检查
    try:
        await db.execute("SELECT 1")
        checks["postgres"] = "ok"
    except Exception as e:
        checks["postgres"] = f"error: {str(e)}"

    # Redis 检查
    try:
        await redis.ping()
        checks["redis"] = "ok"
    except Exception as e:
        checks["redis"] = f"error: {str(e)}"

    all_ok = all(v == "ok" for v in checks.values())

    return {
        "status": "ready" if all_ok else "not_ready",
        "checks": checks,
    }


@router.get("/metrics")
async def metrics_endpoint():
    """Prometheus 指标端点"""
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
```

### metrics.py（Prometheus 指标）

```python
"""
Prometheus 自定义指标。

跟踪 Agent API 的关键性能指标。
"""

from prometheus_client import Counter, Histogram, Gauge, Registry

# 创建独立 registry，避免与默认指标冲突
agent_registry = Registry()

# ==================== 请求指标 ====================

# HTTP 请求总数（按方法、端点、状态码分类）
http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
    registry=agent_registry,
)

# HTTP 请求延迟（秒）
http_request_duration = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    registry=agent_registry,
)

# ==================== Agent 指标 ====================

# LLM API 调用次数
llm_calls_total = Counter(
    "llm_calls_total",
    "Total LLM API calls",
    ["model", "endpoint"],
    registry=agent_registry,
)

# LLM Token 使用量
llm_tokens_used = Counter(
    "llm_tokens_used_total",
    "Total tokens used by LLM",
    ["model", "type"],  # type: prompt | completion
    registry=agent_registry,
)

# 工具调用次数
tool_calls_total = Counter(
    "tool_calls_total",
    "Total tool invocations",
    ["tool_name", "status"],  # status: success | error
    registry=agent_registry,
)

# ==================== 会话指标 ====================

# 活跃会话数
active_sessions = Gauge(
    "active_sessions",
    "Number of active sessions",
    registry=agent_registry,
)

# 会话平均轮次
session_avg_turns = Histogram(
    "session_average_turns",
    "Average conversation turns per session",
    buckets=(1, 5, 10, 20, 50, 100),
    registry=agent_registry,
)

# ==================== RAG 指标 ====================

# RAG 检索次数
rag_retrievals_total = Counter(
    "rag_retrievals_total",
    "Total RAG retrieval operations",
    ["status"],  # status: hit | miss
    registry=agent_registry,
)

# RAG 检索延迟
rag_retrieval_duration = Histogram(
    "rag_retrieval_duration_seconds",
    "RAG retrieval duration in seconds",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0),
    registry=agent_registry,
)

# ==================== 错误指标 ====================

# 错误总数
errors_total = Counter(
    "errors_total",
    "Total errors by type",
    ["error_type"],
    registry=agent_registry,
)


class MetricsMiddleware:
    """FastAPI 中间件：自动采集请求指标"""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start_time = time.monotonic()
        method = scope["method"]
        endpoint = scope["path"]

        status_code = 200

        async def after_response(message):
            nonlocal status_code
            if message.get("type") == "http.response.start":
                status_code = message.get("status", 200)

            duration = time.monotonic() - start_time
            http_requests_total.labels(
                method=method, endpoint=endpoint, status_code=status_code
            ).inc()
            http_request_duration.labels(method=method, endpoint=endpoint).observe(duration)

        await self.app(scope, receive, after_response)
```

---

## 6. 日志系统 `logging_config.py`

```python
"""
结构化日志配置。

支持 JSON 格式（生产）和彩色文本格式（开发）。
集成请求 ID，便于分布式追踪。
"""

import json
import logging
import sys
import uuid
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    """JSON 格式化器"""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # 附加 request_id（如果存在）
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id

        # 附加 session_id（如果存在）
        if hasattr(record, "session_id"):
            log_data["session_id"] = record.session_id

        # 异常信息
        if record.exc_info and record.exc_info[0] is not None:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data, ensure_ascii=False)


class RequestIdFilter(logging.Filter):
    """将请求 ID 注入日志记录"""

    def filter(self, record: logging.LogRecord) -> bool:
        # 从上下文获取 request_id
        import contextvars
        ctx = contextvars.ContextVar("request_id", default=None)
        record.request_id = ctx.get() or "no-request"
        return True


def setup_logging(level: str = "INFO", fmt: str = "json") -> None:
    """配置全局日志"""
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # 清除已有 handler
    root_logger.handlers.clear()

    if fmt == "json":
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)-8s] %(name)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler.addFilter(RequestIdFilter())
    root_logger.addHandler(handler)

    # 降低第三方库日志级别
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(level.upper())


# ==================== FastAPI 请求日志中间件 ====================

class RequestLoggingMiddleware:
    """FastAPI 中间件：记录每个请求的关键信息"""

    def __init__(self, app):
        self.app = app
        self.logger = logging.getLogger("request")

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        import time
        import contextvars

        request_id = str(uuid.uuid4())[:8]
        ctx_token = contextvars.ContextVar("request_id", default=None).set(request_id)

        start_time = time.monotonic()
        method = scope["method"]
        path = scope["path"]
        client_host = scope.get("client", ("unknown", 0))[0]

        status_code = 200

        async def after_response(message):
            if message.get("type") == "http.response.start":
                nonlocal status_code
                status_code = message.get("status", 200)

            duration_ms = (time.monotonic() - start_time) * 1000
            self.logger.info(
                "%s %s → %d (%.1fms) from %s [%s]",
                method, path, status_code, duration_ms, client_host, request_id,
            )

        try:
            await self.app(scope, receive, after_response)
        finally:
            contextvars.ContextVar("request_id", default=None).reset(ctx_token)
```

---

## 7. 初始化脚本 `scripts/init.sql`

```sql
-- PostgreSQL 数据库初始化脚本

-- ==================== 会话表 ====================
CREATE TABLE IF NOT EXISTS sessions (
    session_id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(128),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_sessions_user_id ON sessions(user_id);
CREATE INDEX idx_sessions_updated_at ON sessions(updated_at DESC);

-- ==================== 消息表 ====================
CREATE TABLE IF NOT EXISTS messages (
    message_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(36) REFERENCES sessions(session_id) ON DELETE CASCADE,
    role VARCHAR(16) NOT NULL CHECK (role IN ('user', 'assistant', 'system', 'tool')),
    content TEXT NOT NULL,
    tool_calls JSONB,              -- 工具调用信息（如有）
    token_usage JSONB,             -- {"prompt_tokens": N, "completion_tokens": N}
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_messages_session_id ON messages(session_id);
CREATE INDEX idx_messages_created_at ON messages(created_at DESC);

-- ==================== 知识库文档表 ====================
CREATE TABLE IF NOT EXISTS knowledge_documents (
    doc_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source VARCHAR(512) NOT NULL,   -- 来源路径/URL
    mime_type VARCHAR(64),
    metadata JSONB DEFAULT '{}',
    chunk_count INTEGER DEFAULT 0,
    indexed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_knowledge_documents_source ON knowledge_documents(source);

-- ==================== API Key 管理表 ====================
CREATE TABLE IF NOT EXISTS api_keys (
    key_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    key_hash VARCHAR(64) NOT NULL UNIQUE,   -- bcrypt hash
    name VARCHAR(128),                       -- 用户可读名称
    permissions JSONB DEFAULT '["read"]',    -- 权限列表
    rate_limit INTEGER DEFAULT 60,           -- 每分钟请求数限制
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_api_keys_hash ON api_keys(key_hash);

-- ==================== 审计日志表 ====================
CREATE TABLE IF NOT EXISTS audit_logs (
    log_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    action VARCHAR(64) NOT NULL,             -- 操作类型
    target_type VARCHAR(32),                 -- 目标资源类型
    target_id VARCHAR(128),                  -- 目标资源 ID
    actor VARCHAR(128),                      -- 操作用户/API Key
    details JSONB,                           -- 详细信息
    ip_address INET,                         -- 客户端 IP
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_audit_logs_action ON audit_logs(action);
CREATE INDEX idx_audit_logs_created_at ON audit_logs(created_at DESC);
```

---

## 8. 架构总览

```
                    ┌─────────────────────────────┐
                    │         Client / Web        │
                    └──────────────┬──────────────┘
                                   │ HTTPS
                    ┌──────────────▼──────────────┐
                    │     Ingress / Load Balancer  │
                    │   (Nginx / K8s Ingress)      │
                    └──────────────┬──────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                     ▼
     ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
     │ Agent API   │      │ Agent API   │      │ Agent API   │
     │ Pod 1       │      │ Pod 2       │      │ Pod 3       │
     │ (HPA: 2-10) │      │             │      │             │
     └──────┬──────┘      └──────┬──────┘      └──────┬──────┘
            │                    │                     │
     ┌──────▼────────────────────▼─────────────────────▼──────┐
     │                   Shared Services                      │
     │                                                        │
     │  ┌──────────┐  ┌──────────┐  ┌──────────┐             │
     │  │  Redis   │  │PostgreSQL│  │  Qdrant  │             │
     │  │(Session) │  │ (Data)   │  │ (Vector) │             │
     │  └──────────┘  └──────────┘  └──────────┘             │
     └────────────────────────────────────────────────────────┘

                    ┌─────────────────────────────┐
                    │       Monitoring Stack      │
                    │                             │
                    │  Prometheus ← metrics       │
                    │  Grafana   ← dashboards     │
                    │  Loki/ELK ← logs             │
                    └─────────────────────────────┘

                    ┌─────────────────────────────┐
                    │         CI/CD Pipeline       │
                    │                             │
                    │  GitHub Actions / GitLab CI  │
                    │    ↓ build → test → deploy   │
                    │  Docker Hub / K8s Registry   │
                    └─────────────────────────────┘
```

---

## 9. 设计总结

| 特性 | 实现方式 |
|------|---------|
| **容器化** | Docker + docker-compose，一键启动完整环境 |
| **K8s 编排** | Deployment + Service + Ingress + HPA |
| **弹性伸缩** | HPA 按 CPU/自定义指标自动扩缩容（2-10 Pod） |
| **健康检查** | /health（存活）+ /ready（就绪），K8s Probe 集成 |
| **可观测性** | Prometheus 指标 + Grafana 面板 + JSON 结构化日志 |
| **配置管理** | Settings 类，环境变量 > .env > defaults.yaml |
| **数据库初始化** | init.sql 自动建表、索引 |
| **多环境支持** | development / staging / production 通过 ENVIRONMENT 切换 |
