# 部署与 DevOps — 验收标准

## DEP-01: Dockerfile — 镜像构建

**优先级**: P0

- **Given** `Dockerfile` 存在且基于 Python slim image
- **When** 执行 `docker build -t yellowbull .`
- **Then** 构建成功；镜像包含 Python runtime + 依赖；不包含 dev dependencies；镜像大小 < 500MB
- **验证方式**: 手动验证 — docker build

## DEP-02: Dockerfile — 多阶段构建

**优先级**: P1

- **Given** `Dockerfile` 使用 multi-stage build
- **When** 构建完成
- **Then** builder stage 安装依赖并缓存；final stage 仅包含运行时文件；pip cache 不进入最终镜像
- **验证方式**: 手动验证 — docker history 检查 layers

## DEP-03: Docker Compose — 服务编排

**优先级**: P0

- **Given** `docker-compose.yml` 定义了 app、redis、chromadb 服务
- **When** 执行 `docker compose up -d`
- **Then** 所有服务启动成功；app 依赖 redis/chromadb（depends_on + healthcheck）；端口映射正确
- **验证方式**: 手动验证 — docker compose ps

## DEP-04: Docker Compose — 环境变量注入

**优先级**: P1

- **Given** `docker-compose.yml` 引用 `.env` 文件
- **When** 服务启动
- **Then** 所有 `${VAR}` 被替换为 .env 中的值；敏感信息不硬编码在 compose 文件中
- **验证方式**: 手动验证 — docker inspect 检查 env

## DEP-05: Health Check — /health 端点

**优先级**: P0

- **Given** Dockerfile 配置 `HEALTHCHECK` 指令调用 `/health`
- **When** 容器运行中
- **Then** Docker health status 最终变为 healthy；健康检查间隔合理（如 30s）
- **验证方式**: 手动验证 — docker inspect

## DEP-06: Health Check — /ready 端点

**优先级**: P1

- **Given** 服务依赖 Redis/LLM API
- **When** Redis 未就绪时访问 `/ready`
- **Then** 返回 HTTP 503 + `{"status": "not_ready", "dependencies": {"redis": "error"}}`；Redis 就绪后返回 200
- **验证方式**: 集成测试

## DEP-07: CI/CD — GitHub Actions Workflow

**优先级**: P1

- **Given** `.github/workflows/ci.yml` 存在
- **When** PR 提交到 main 分支
- **Then** 自动运行 lint → test → build；全部通过才允许合并；失败时显示具体错误
- **验证方式**: 手动验证 — 创建测试 PR

## DEP-08: CI/CD — 测试阶段

**优先级**: P1

- **Given** CI workflow 的 test stage
- **When** 运行测试
- **Then** 执行 `pytest tests/unit` 和 `pytest tests/integration`；生成覆盖率报告；覆盖率 >= 80% 才通过
- **验证方式**: 手动验证 — 查看 CI logs

## DEP-09: Entrypoint Script

**优先级**: P1

- **Given** `entrypoint.sh` 作为容器入口点
- **When** 容器启动
- **Then** 执行依赖检查（Redis/LLM API）→ 等待就绪 → 启动 uvicorn；任一依赖不可达时重试而非退出
- **验证方式**: 手动验证

## DEP-10: Graceful Shutdown

**优先级**: P1

- **Given** 容器接收 SIGTERM（docker stop）
- **When** 关闭过程中
- **Then** 完成正在处理的请求（最多 30s）；清理资源（连接池、定时任务）；不丢失数据
- **验证方式**: 集成测试 — docker stop + 日志检查

## DEP-11: Resource Limits

**优先级**: P2

- **Given** `docker-compose.yml` 配置了 deploy.resources.limits
- **When** 容器运行中
- **Then** CPU 和内存限制生效；超限时代杀进程而非影响宿主机
- **验证方式**: 手动验证 — docker stats

## DEP-12: Log Driver

**优先级**: P2

- **Given** `docker-compose.yml` 配置 logging driver
- **When** 服务运行中产生日志
- **Then** 日志输出到 stdout（json-file driver）；支持 `docker logs` 查看；日志轮转不丢失
- **验证方式**: 手动验证
