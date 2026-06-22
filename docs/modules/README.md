# Modules —— 模块详细设计索引

本文档按模块组织 YellowBull Agent 的详细设计，每个文档包含职责边界、核心代码示例、数据流图和接口契约。

## 阅读顺序建议

```
架构总览 → Config System → LLM Abstraction → Tool System
    ↓
Session Manager → Agent Core → API Server → Observability
```

## 文档列表

| # | 文档 | 模块 | 核心内容 |
|---|------|------|---------|
| 1 | [Config System](./01-config-system.md) | `config/` | YAML 配置、Pydantic 校验、热加载（watchdog）、多环境 Profile |
| 2 | [LLM Abstraction](./02-llm-abstraction.md) | `llm/` | Provider ABC、OpenAI/Anthropic/Azure 实现、流式协议、Fallback 策略 |
| 3 | [Tool System](./03-tool-system.md) | `tools/` | Tool ABC、Registry、JSON Schema 生成、热加载、内置工具示例 |
| 4 | [Session Manager](./04-session-manager.md) | `session/` | Session CRUD、SQLite WAL、上下文窗口裁剪、并发安全 |
| 5 | [Agent Core](./05-agent-core.md) | `agent/` | 主循环状态机、SSE 流式协议、工具编排、错误恢复 |
| 6 | [API Server](./06-api-server.md) | `api/` | FastAPI 路由、REST + SSE、Lifespan 管理、中间件 |
| 7 | [Observability](./07-observability.md) | `observability/` | JSON 日志、Prometheus 指标、OpenTelemetry 追踪、健康检查 |

## 模块依赖关系

```
API Server ──┐
             ├──→ Agent Core ───┬──→ LLM Provider
             │                  └──→ Tool Registry ──→ Tools
             │
             ├──→ Session Manager ──→ SQLite
             │
             └──→ ConfigManager ──────→ YAML + Watchdog
                     ↓ (on_change)
              [All modules reload]

Observability ──→ hooks into all modules (non-invasive)
```

## 快速导航

- **如何添加新工具？** → [Tool System §8](./03-tool-system.md#8-新增工具步骤清单)
- **Agent 主循环怎么工作？** → [Agent Core §4](./05-agent-core.md#4-agent-主循环状态机)
- **SSE 流式协议格式？** → [Agent Core §3](./05-agent-core.md#3-sse-流式协议)
- **配置热加载流程？** → [Config System §6](./01-config-system.md#6-热加载完整流程示例)
- **如何监控 Agent？** → [Observability §6](./07-observability.md#6-yaml-configuration-section)
