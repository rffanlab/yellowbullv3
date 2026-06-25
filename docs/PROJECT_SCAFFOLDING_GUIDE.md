# YellowBull Agent 项目脚手架指南

> **重要**: 本文档是 AI Agent 创建项目的唯一依据。严禁生成通用 CRUD 应用！

---

## 一、项目定位（必读）

| 属性 | 值 |
|------|-----|
| **项目名称** | yellowbull-agent (YellowBull v3) |
| **项目类型** | AI Agent 框架 —— 对话 → 意图识别 → 工具调用 → 回复 |
| **语言** | Python ≥ 3.11 |
| **MVP 目标** | 跑通「用户提问 → LLM 推理 → 自动调用工具 → 返回结果」最小闭环 |
| **NOT** | ❌ 不是 CRUD 应用 ❌ 不需要 MySQL ❌ 不需要用户管理系统 |

---

## 二、项目结构（严格按此创建）

```
yellowbullv3/
├── config/
│   ├── __init__.py
│   ├── settings.py          # Dataclass 配置 + YAML 加载 + 环境变量解析
│   └── default.yaml         # 默认配置文件
├── core/
│   ├── __init__.py
│   ├── agent.py             # Agent 核心编排（主循环：LLM → tool → LLM）
│   ├── session_manager.py   # 会话管理（内存存储，后续可换 Redis）
│   ├── context_builder.py   # Prompt 上下文构建（滑动窗口裁剪）
│   └── tool_executor.py     # 工具执行引擎（超时、重试、并行调度、缓存）
├── llm/
│   ├── __init__.py
│   ├── base.py              # BaseLLM ABC + Message / ToolDefinition / LLMResponse
│   ├── openai_llm.py        # OpenAI provider 实现
│   ├── anthropic_llm.py     # Anthropic provider 实现
│   ├── ollama_llm.py        # Ollama 本地模型实现
│   └── factory.py           # Provider 工厂（根据配置创建对应 LLM）
├── tools/
│   ├── __init__.py
│   ├── base.py              # BaseTool ABC + ToolInfo / ToolResult
│   ├── registry.py          # ToolRegistry 注册中心 + 装饰器
│   ├── builtin/
│   │   ├── __init__.py
│   │   ├── current_time.py  # 时间查询工具
│   │   ├── calculator.py    # 数学计算工具
│   │   └── web_search.py    # 网络搜索工具
│   └── function_calling.py  # Function Calling 协议适配
├── models/
│   ├── __init__.py
│   ├── message.py           # Message dataclass（含 tool_calls, tool_call_id）
│   ├── session.py           # Session + SessionState dataclass
│   └── tool_result.py       # ToolResult dataclass
├── api/
│   ├── __init__.py
│   └── server.py            # FastAPI HTTP + SSE 流式服务
├── tests/                   # 测试目录
├── main.py                  # 入口：加载配置 → 创建 LLM → 启动 Agent → uvicorn
├── pyproject.toml           # 项目依赖
└── .env.example             # 环境变量模板
```

---

## 三、核心模块清单（P0 MVP 必须实现）

### P0 必做模块

| # | 模块 | 文件 | 职责 | 详细设计文档 |
|---|------|------|------|------------|
| 1 | **配置系统** | `config/settings.py` + `default.yaml` | YAML 加载、环境变量 `${VAR}` 解析、dataclass 映射 | [01-config-system.md](./modules/01-config-system.md) |
| 2 | **LLM 抽象层** | `llm/base.py` + `openai_llm.py` + `factory.py` | Provider ABC、统一 Message/ToolDefinition 协议、工厂模式 | [02-llm-abstraction.md](./modules/02-llm-abstraction.md) |
| 3 | **工具系统** | `tools/base.py` + `registry.py` + `builtin/*.py` | Tool ABC、注册中心、JSON Schema 生成、内置工具 | [03-tool-system.md](./modules/03-tool-system.md) |
| 4 | **会话管理** | `core/session_manager.py` + `models/session.py` | Session CRUD（内存）、滑动窗口上下文裁剪 | [04-session-manager.md](./modules/04-session-manager.md) |
| 5 | **工具执行器** | `core/tool_executor.py` | 超时控制、指数退避重试、并行/串行调度、结果缓存、错误隔离 | [05-tool-executor.md](./modules/05-tool-executor.md) |
| 6 | **Agent Core** | `core/agent.py` + `context_builder.py` | 主循环状态机：LLM → tool_calls → execute → loop | [06-agent-core.md](./modules/06-agent-core.md) |
| 7 | **API Server** | `api/server.py` | FastAPI REST `/api/chat`、SSE 流式端点 | [09-api-server.md](./modules/09-api-server.md) |

### P1+ 后续模块（MVP 阶段不实现）

| 模块 | 说明 | MVP 阶段处理 |
|------|------|-------------|
| Memory System | 短期/长期记忆、语义检索 | ❌ 不做 |
| RAG | 向量数据库、Embedding、混合搜索 | ❌ 不做 |
| Multi-Agent | Supervisor 路由、多 Agent 协作 | ❌ 不做 |
| Code Execution | Docker 沙箱执行代码 | ❌ 不做 |
| NL2SQL | 自然语言 → SQL 转换 | ❌ 不做 |
| Security Governance | Prompt Injection 防护、RBAC | ❌ 不做 |
| Observability | Prometheus 指标、OpenTelemetry | 仅基础日志 |

---

## 四、核心数据流（Agent 主循环）

```
用户消息入会话
    ↓
[循环，最多 max_chain_depth 次]
    ├─ ContextBuilder: 构建上下文（system prompt + 滑动窗口历史消息）
    ├─ LLM.chat(messages, tools): 推理
    ├─ 返回无 tool_calls？ → 组装最终回复，退出循环 ✓
    └─ 返回有 tool_calls？
        ├─ 记录 assistant 消息（含 tool_calls）到会话
        ├─ ToolRegistry: 并行执行所有工具调用
        ├─ 将每个工具结果写入会话（role=tool, tool_call_id 关联）
        └─ 回到循环开头，LLM 根据工具结果继续推理
    ↓
超过最大深度 → 返回兜底回复
```

---

## 五、配置要点

### `default.yaml` 必须包含的配置段

```yaml
llm:
  provider: openai              # openai | anthropic | ollama
  settings:
    openai:
      api_key: "${OPENAI_API_KEY}"
      model: "gpt-4o"
      temperature: 0.7
      max_tokens: 4096

agent:
  system_prompt: "你是一个智能助手..."
  context_window: 48
  max_tool_calls_per_turn: 4
  max_chain_depth: 5
  tool_retry_limit: 3
  total_timeout_seconds: 60

tools:
  builtin:
    current_time: { enabled: true }
    calculator:   { enabled: true }
    web_search:   { enabled: true, engine: duckduckgo }

server:
  host: "0.0.0.0"
  port: 8000

logging:
  level: INFO
```

### `.env.example`

```bash
OPENAI_API_KEY=sk-xxx
ANTHROPIC_API_KEY=sk-ant-xxx
SEARCH_API_KEY=your-search-key
```

---

## 六、依赖清单 `pyproject.toml`

```toml
[project]
name = "yellowbull-agent"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.34",
    "pydantic>=2.10",
    "pyyaml>=6.0",
    "httpx>=0.28",
    "openai>=1.58",
    "anthropic>=0.42",
    "duckduckgo-search>=7.2",
    "tiktoken>=0.9",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-asyncio>=0.24", "ruff>=0.8", "mypy>=1.13"]
```

---

## 七、MVP 验收标准

| # | 验收项 | 验证方式 |
|---|--------|---------|
| 1 | `python main.py` 启动成功，`/api/chat` 可访问 | curl POST /api/chat |
| 2 | 发送普通问题 → 返回 LLM 回复（不调用工具） | 直接问答测试 |
| 3 | 问"现在几点" → 自动调用 `current_time` → 返回结果 | 工具调用测试 |
| 4 | 连续对话能记住上下文 | 多轮对话测试 |
| 5 | 复杂请求触发多次 LLM → tool → LLM 循环 | 链式调用测试 |
| 6 | 修改 `default.yaml` 中 provider → 重启切换 LLM | 配置切换测试 |
| 7 | 工具执行失败能重试并返回友好提示 | 错误处理测试 |

---

## 八、常见错误（避免踩坑）

| ❌ 错误做法 | ✅ 正确做法 |
|------------|-----------|
| 创建 MySQL/PostgreSQL 数据库连接 | MVP 阶段会话用内存存储，不需要数据库 |
| 实现用户注册/登录系统 | MVP 只有 user_id 字符串标识，无需认证 |
| 使用 SQLAlchemy ORM | MVP 只用 dataclass + dict |
| 创建 CRUD API（增删改查） | API 只暴露 `/api/chat`、会话管理等 Agent 相关端点 |
| 忽略工具调用机制 | 必须实现 tool_calls → execute → loop 主循环 |
| 硬编码 LLM provider | 必须通过配置切换，支持多 provider |

---

## 九、详细设计文档索引

创建项目时，每个模块的实现应参考对应的详细设计文档：

- [P0 核心能力详细设计](./p0-design.md) ← **总纲，必读**
- [01 配置系统](./modules/01-config-system.md)
- [02 LLM 抽象层](./modules/02-llm-abstraction.md)
- [03 工具系统](./modules/03-tool-system.md)
- [04 会话管理](./modules/04-session-manager.md)
- [05 Tool Executor](./modules/05-tool-executor.md)
- [06 Agent Core](./modules/06-agent-core.md)
- [09 API Server](./modules/09-api-server.md)

完整模块索引 → [docs/modules/README.md](./modules/README.md)
