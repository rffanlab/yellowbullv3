# Modules —— 模块详细设计索引

本文档按模块组织 YellowBull Agent 的详细设计，每个文档包含职责边界、核心代码示例、数据流图和接口契约。

## 阅读顺序建议

```
基础层:   Config → LLM Abstraction → Tool System
核心层:   Session Manager → Memory System → Agent Core
智能层:   RAG → Multi-Agent Collaboration
服务层:   API Server → Observability
扩展层:   Code Execution → NL2SQL → Prompt Templates
保障层:   Security Governance → Evaluation Testing → Deployment Ops
```

## 文档列表

### 基础层 (Foundation Layer)

| # | 文档 | 模块 | 核心内容 |
|---|------|------|---------|
| 1 | [Config System](./01-config-system.md) | `config/` | YAML 配置、Pydantic 校验、热加载（watchdog）、多环境 Profile |
| 2 | [LLM Abstraction](./02-llm-abstraction.md) | `llm/` | Provider ABC、OpenAI/Anthropic/Azure/Ollama 实现、流式协议、Fallback 策略 |
| 3 | [Tool System](./03-tool-system.md) | `tools/` | Tool ABC、Registry、JSON Schema 生成、热加载、内置工具示例 |

### 核心层 (Core Layer)

| # | 文档 | 模块 | 核心内容 |
|---|------|------|---------|
| 4 | [Session Manager](./04-session-manager.md) | `session/` | Session CRUD、SQLite WAL、上下文窗口裁剪、并发安全 |
| 5 | [Memory System](./05-memory-system.md) | `memory/` | 短期记忆、长期记忆、记忆压缩、语义检索 |
| 6 | [Agent Core](./06-agent-core.md) | `agent/` | 主循环状态机、SSE 流式协议、工具编排、错误恢复 |

### 智能层 (Intelligence Layer)

| # | 文档 | 模块 | 核心内容 |
|---|------|------|---------|
| 7 | [RAG](./07-rag.md) | `rag/` | 向量数据库、Embedding、检索策略、混合搜索 |
| 8 | [Multi-Agent Collaboration](./08-multi-agent-collaboration.md) | `multi_agent/` | Agent 注册表、Supervisor 路由、协作编排、结果聚合 |

### 服务层 (Service Layer)

| # | 文档 | 模块 | 核心内容 |
|---|------|------|---------|
| 9 | [API Server](./09-api-server.md) | `api/` | FastAPI 路由、REST + SSE、Lifespan 管理、中间件 |
| 10 | [Observability](./10-observability.md) | `observability/` | JSON 日志、Prometheus 指标、OpenTelemetry 追踪、健康检查 |

### 扩展层 (Extension Layer)

| # | 文档 | 模块 | 核心内容 |
|---|------|------|---------|
| 11 | [Code Execution](./11-code-execution.md) | `tools/code/` | 沙箱隔离、Docker 容器执行、超时控制、资源限制 |
| 12 | [NL2SQL Database Operations](./12-nl2sql-database-operations.md) | `tools/database/` | Schema 感知、NL→SQL 转换、安全执行、结果解释 |
| 13 | [Prompt Template Management](./13-prompt-template-management.md) | `prompt/` | 模板存储、Jinja2 渲染、热更新、A/B 测试 |

### 保障层 (Assurance Layer)

| # | 文档 | 模块 | 核心内容 |
|---|------|------|---------|
| 14 | [Security Governance](./14-security-governance.md) | `security/` | Prompt Injection 防护、API Key 认证、RBAC、速率限制、输出过滤、审计日志 |
| 15 | [Evaluation Testing](./15-evaluation-testing.md) | `evaluation/` | BLEU/ROUGE/BERTScore、LLM-as-Judge、幻觉检测、多模型对比、CI/CD 集成 |
| 16 | [Deployment Ops](./16-deployment-ops.md) | `deploy/` | Docker Compose、K8s 编排、HPA 弹性伸缩、Prometheus/Grafana 监控、配置管理 |

## 模块依赖关系

```
                    ┌─────────────┐
                    │ Config      │ ← YAML + Watchdog
                    │ System      │
                    └──────┬──────┘
                           │ (provides config to all)
         ┌─────────────────┼─────────────────┐
         ▼                 ▼                 ▼
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
         │         Assurance Layer                     │
         │  Security → Evaluation → Deployment Ops     │
         └─────────────────────────────────────────────┘

 Observability ──→ hooks into all modules (non-invasive)
```

## 快速导航

- **如何添加新工具？** → [Tool System §8](./03-tool-system.md#8-新增工具步骤清单)
- **Agent 主循环怎么工作？** → [Agent Core §4](./06-agent-core.md#4-agent-主循环状态机)
- **SSE 流式协议格式？** → [Agent Core §3](./06-agent-core.md#3-sse-流式协议)
- **配置热加载流程？** → [Config System §6](./01-config-system.md#6-热加载完整流程示例)
- **如何监控 Agent？** → [Observability §6](./10-observability.md#6-yaml-configuration-section)
- **多智能体怎么协作？** → [Multi-Agent §7](./08-multi-agent-collaboration.md#7-协作模式图)
- **代码沙箱如何隔离？** → [Code Execution §4](./11-code-execution.md#4-sandbox-实现方式对比)
- **NL2SQL 安全策略？** → [NL2SQL §6](./12-nl2sql-database-operations.md#6-sql-安全策略)
- **如何评估 Agent 质量？** → [Evaluation Testing §5](./15-evaluation-testing.md#5-架构总览)
- **Prompt Injection 怎么防？** → [Security Governance §4](./14-security-governance.md#4-prompt-injection-防护)
