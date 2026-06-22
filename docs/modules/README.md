# Modules —— 模块详细设计索引

本文档按模块组织 YellowBull Agent 的详细设计，每个文档包含职责边界、核心代码示例、数据流图和接口契约。

## 阅读顺序建议

```
核心层:   Config → LLM → Tools → Session → Agent Core
服务层:   API Server → Observability
智能层:   RAG → Memory → Security
基础设施: Cache → MQ → Deployment
扩展层:   Code Execution → Multi-Agent → NL2SQL → Prompt Templates
```

## 文档列表

### 核心层 (Core Layer)

| # | 文档 | 模块 | 核心内容 |
|---|------|------|---------|
| 1 | [Config System](./01-config-system.md) | `config/` | YAML 配置、Pydantic 校验、热加载（watchdog）、多环境 Profile |
| 2 | [LLM Abstraction](./02-llm-abstraction.md) | `llm/` | Provider ABC、OpenAI/Anthropic/Azure 实现、流式协议、Fallback 策略 |
| 3 | [Tool System](./03-tool-system.md) | `tools/` | Tool ABC、Registry、JSON Schema 生成、热加载、内置工具示例 |
| 4 | [Session Manager](./04-session-manager.md) | `session/` | Session CRUD、SQLite WAL、上下文窗口裁剪、并发安全 |
| 5 | [Agent Core](./05-agent-core.md) | `agent/` | 主循环状态机、SSE 流式协议、工具编排、错误恢复 |

### 服务层 (Service Layer)

| # | 文档 | 模块 | 核心内容 |
|---|------|------|---------|
| 6 | [API Server](./06-api-server.md) | `api/` | FastAPI 路由、REST + SSE、Lifespan 管理、中间件 |
| 7 | [Observability](./07-observability.md) | `observability/` | JSON 日志、Prometheus 指标、OpenTelemetry 追踪、健康检查 |

### 智能层 (Intelligence Layer)

| # | 文档 | 模块 | 核心内容 |
|---|------|------|---------|
| 8 | [RAG](./08-rag.md) | `rag/` | 向量数据库、Embedding、检索策略、混合搜索 |
| 9 | [Memory System](./09-memory-system.md) | `memory/` | 短期记忆、长期记忆、记忆压缩、语义检索 |
| 10 | [Security Governance](./10-security-governance.md) | `security/` | 内容过滤、权限控制、审计日志、合规检查 |

### 基础设施层 (Infrastructure Layer)

| # | 文档 | 模块 | 核心内容 |
|---|------|------|---------|
| 11 | [Cache Layer](./11-cache-layer.md) | `cache/` | Redis 缓存、本地缓存、多级缓存策略 |
| 12 | [Message Queue](./12-message-queue.md) | `mq/` | 异步任务队列、消息持久化、重试机制 |
| 13 | [Deployment DevOps](./13-deployment-devops.md) | `deploy/` | Docker 部署、CI/CD、监控告警、灰度发布 |

### 扩展层 (Extension Layer)

| # | 文档 | 模块 | 核心内容 |
|---|------|------|---------|
| 14 | [Code Execution](./14-code-execution.md) | `tools/code/` | 沙箱隔离、Docker 容器执行、超时控制、资源限制 |
| 15 | [Multi-Agent Collaboration](./15-multi-agent-collaboration.md) | `multi_agent/` | Agent 注册表、Supervisor 路由、协作编排、结果聚合 |
| 16 | [NL2SQL Database Operations](./16-nl2sql-database-operations.md) | `tools/database/` | Schema 感知、NL→SQL 转换、安全执行、结果解释 |
| 17 | [Prompt Template Management](./17-prompt-template-management.md) | `prompt/` | 模板存储、Jinja2 渲染、热更新、A/B 测试 |

## 模块依赖关系

```
                    ┌─────────────┐
                    │ Config      │ ← YAML + Watchdog
                    │ System      │
                    └──────┬──────┘
                           │ (provides config to all)
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
   ┌─────────┐       ┌─────────┐       ┌─────────┐
   │ LLM     │       │ Tool    │       │ Session │
   │ Abstraction│    │ System  │       │ Manager │
   └────┬────┘       └────┬────┘       └────┬────┘
        │                  │                 │
        └────────┬─────────┘────────────────┘
                 ▼
          ┌─────────────┐
          │  Agent Core │ ← 核心编排引擎
          └──────┬──────┘
                 │
     ┌───────────┼───────────┬──────────────┐
     ▼           ▼           ▼              ▼
┌─────────┐ ┌─────────┐ ┌─────────┐  ┌──────────┐
│ RAG     │ │ Memory  │ │ Security│  │Multi-Agent│
│ System  │ │ System  │ │Governance│ │Collaboration│
└─────────┘ └─────────┘ └─────────┘  └────┬─────┘
                                           │
        ┌──────────────────────────────────┤
        ▼                                  ▼
   ┌─────────┐ ┌─────────┐          ┌──────────┐
   │Code     │ │ NL2SQL  │          │ Prompt   │
   │Execution│ │Database │          │Templates │
   └─────────┘ └─────────┘          └──────────┘

        ┌─────────────────────────────────────────────┐
        │           API Server (FastAPI)               │
        │  ← exposes all capabilities via REST + SSE   │
        └─────────────────────────────────────────────┘

        ┌─────────────────────────────────────────────┐
        │         Infrastructure Layer                 │
        │  Cache (Redis) → MQ (async tasks) → Deploy  │
        └─────────────────────────────────────────────┘

Observability ──→ hooks into all modules (non-invasive)
```

## 快速导航

- **如何添加新工具？** → [Tool System §8](./03-tool-system.md#8-新增工具步骤清单)
- **Agent 主循环怎么工作？** → [Agent Core §4](./05-agent-core.md#4-agent-主循环状态机)
- **SSE 流式协议格式？** → [Agent Core §3](./05-agent-core.md#3-sse-流式协议)
- **配置热加载流程？** → [Config System §6](./01-config-system.md#6-热加载完整流程示例)
- **如何监控 Agent？** → [Observability §6](./07-observability.md#6-yaml-configuration-section)
- **多智能体怎么协作？** → [Multi-Agent §7](./15-multi-agent-collaboration.md#7-协作模式图)
- **代码沙箱如何隔离？** → [Code Execution §4](./14-code-execution.md#4-sandbox-实现方式对比)
- **NL2SQL 安全策略？** → [NL2SQL §6](./16-nl2sql-database-operations.md#6-sql-安全策略)
