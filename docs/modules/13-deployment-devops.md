# 容器化部署与 DevOps（Deployment & DevOps）详细设计

## 1. 职责边界

| 组件 | 说明 |
|------|------|
| **Dockerfile** | 多阶段构建，最小镜像体积 |
| **docker-compose.yml** | 本地开发环境一键启动（应用 + Redis + ChromaDB） |
| **健康检查** | /health（进程存活）、/ready（依赖就绪） |
| **环境变量** | 敏感配置通过 env vars 注入 |
| **日志收集** | JSON 格式 stdout，对接 ELK/Loki |
| **CI/CD** | GitHub Actions 自动化测试 + 构建镜像 |

---

## 2. Dockerfile

```dockerfile
# 多阶段构建：构建阶段 → 运行阶段

# ========== Stage 1: Builder ==========
FROM python:3.11-slim AS builder

WORKDIR /build

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# 复制并安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ========== Stage 2: Runtime ==========
FROM python:3.11-slim AS runtime

WORKDIR /app

# 从 builder 复制已安装的包
COPY --from=builder /install /usr/local

# 创建非 root 用户
RUN groupadd -r appuser && useradd -r -g appuser appuser

# 复制应用代码
COPY . .

# 创建数据目录
RUN mkdir -p /app/data/chroma /app/logs && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 3. docker-compose.yml

```yaml
# Local development environment
version: '3.9'

services:
  app:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - APP_ENV=development
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - REDIS_URL=redis://redis:6379/0
      - DATABASE_URL=sqlite:///./data/app.db
    volumes:
      - ./config:/app/config          # 挂载配置文件，支持热加载
      - app-data:/app/data            # 持久化数据
      - app-logs:/app/logs            # 日志目录
    depends_on:
      redis:
        condition: service_healthy
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    command: redis-server --appendonly yes
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 3

volumes:
  app-data:
  app-logs:
  redis-data:
```

---

## 4. Dockerfile.prod（生产环境）

```dockerfile
# 生产优化：更小的镜像、只读文件系统、非 root 运行

FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_ENV=production

WORKDIR /app

COPY --from=builder /install /usr/local

RUN groupadd -r appuser && useradd -r -g appuser appuser && \
    mkdir -p /app/data/chroma /app/logs /tmp/app-tmp && \
    chown -R appuser:appuser /app /tmp/app-tmp

COPY --chown=appuser:appuser . .

USER appuser

# 只读挂载应用目录（通过 volume 覆盖可变部分）
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=5s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Gunicorn + Uvicorn workers（生产级 WSGI server）
CMD ["gunicorn", "main:app", \
     "-k", "uvicorn.workers.UvicornWorker", \
     "-w", "4", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120", \
     "--graceful-timeout", "30"]
```

---

## 5. docker-compose.prod.yml（生产环境）

```yaml
version: '3.9'

services:
  app:
    build:
      context: .
      dockerfile: Dockerfile.prod
    ports:
      - "8000:8000"
    environment:
      - APP_ENV=production
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - REDIS_URL=redis://redis:6379/0
      - DATABASE_URL=postgresql://${DB_USER}:${DB_PASS}@postgres:5432/yellowbull
      - LOG_LEVEL=INFO
    volumes:
      - app-data:/app/data
      - app-logs:/app/logs
    depends_on:
      redis:
        condition: service_healthy
      postgres:
        condition: service_healthy
    restart: always
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 2G
        reservations:
          cpus: '1'
          memory: 512M

  redis:
    image: redis:7-alpine
    volumes:
      - redis-data:/data
    command: redis-server --appendonly yes --requirepass ${REDIS_PASSWORD}
    healthcheck:
      test: ["CMD", "redis-cli", "-a", "${REDIS_PASSWORD}", "ping"]
      interval: 10s
      timeout: 5s
      retries: 3

  postgres:
    image: postgres:16-alpine
    environment:
      - POSTGRES_DB=yellowbull
      - POSTGRES_USER=${DB_USER}
      - POSTGRES_PASSWORD=${DB_PASS}
    volumes:
      - postgres-data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${DB_USER}"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  app-data:
  app-logs:
  redis-data:
  postgres-data:
```

---

## 6. .env.example（环境变量模板）

```bash
# ========== Application ==========
APP_ENV=development
LOG_LEVEL=DEBUG
HOST=0.0.0.0
PORT=8000

# ========== LLM Provider ==========
OPENAI_API_KEY=sk-your-key-here
ANTHROPIC_API_KEY=sk-ant-your-key-here
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/

# ========== Database ==========
DATABASE_URL=sqlite:///./data/app.db
# DATABASE_URL=postgresql://user:pass@localhost:5432/yellowbull

# ========== Redis ==========
REDIS_URL=redis://localhost:6379/0

# ========== Security ==========
ADMIN_API_KEY=admin-secret-key-change-me
DEV_API_KEY=dev-secret-key-change-me

# ========== RAG ==========
RAG_ENABLED=false
```

---

## 7. GitHub Actions CI/CD `.github/workflows/ci.yml`

```yaml
name: CI/CD

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11"]

    services:
      redis:
        image: redis:7-alpine
        ports:
          - 6379:6379
        options: --health-cmd "redis-cli ping" --health-interval 10s --health-timeout 5s --health-retries 3

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install pytest pytest-asyncio pytest-cov

      - name: Run tests
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          REDIS_URL: redis://localhost:${{ job.services.redis.ports[6379] }}/0
        run: |
          pytest tests/ --cov=src --cov-report=xml -v

      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          file: coverage.xml

  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install linters
        run: |
          pip install ruff mypy types-requests

      - name: Run ruff
        run: ruff check src/

      - name: Run mypy
        run: mypy src/

  build:
    needs: [test, lint]
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'

    steps:
      - uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Login to Container Registry
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          file: Dockerfile.prod
          push: true
          tags: |
            ghcr.io/${{ github.repository }}/yellowbull:${{ github.sha }}
            ghcr.io/${{ github.repository }}/yellowbull:latest
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

---

## 8. Kubernetes Deployment（可选，生产级）

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: yellowbull
spec:
  replicas: 3
  selector:
    matchLabels:
      app: yellowbull
  template:
    metadata:
      labels:
        app: yellowbull
    spec:
      containers:
        - name: app
          image: ghcr.io/org/yellowbull:latest
          ports:
            - containerPort: 8000
          envFrom:
            - secretRef:
                name: yellowbull-secrets
          resources:
            requests:
              cpu: "500m"
              memory: "512Mi"
            limits:
              cpu: "2"
              memory: "2Gi"
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 10
            periodSeconds: 15
          readinessProbe:
            httpGet:
              path: /ready
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 10

---
apiVersion: v1
kind: Service
metadata:
  name: yellowbull-service
spec:
  selector:
    app: yellowbull
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: ClusterIP

---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: yellowbull-ingress
  annotations:
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "300"
spec:
  rules:
    - host: api.yellowbull.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: yellowbull-service
                port:
                  number: 80
```

---

## 9. 设计总结

| 特性 | 实现方式 |
|------|---------|
| **Docker** | 多阶段构建，最小镜像，非 root 运行 |
| **Compose** | 开发环境一键启动（app + Redis） |
| **生产部署** | Gunicorn + Uvicorn workers，资源限制 |
| **CI/CD** | GitHub Actions：测试 → lint → 构建镜像 |
| **K8s** | Deployment + Service + Ingress，健康检查 |
| **配置管理** | .env 文件 + Kubernetes Secrets |
