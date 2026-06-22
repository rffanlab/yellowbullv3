# Agent 功能全景图

## 一、核心能力矩阵

### 1.1 LLM 多模型编排

| 能力 | 说明 |
|------|------|
| **多 Provider** | OpenAI (GPT-4o/4.1)、Anthropic (Claude)、Azure OpenAI，统一接口切换 |
| **Fallback 链** | 主模型失败 → 自动降级到备用模型（如 GPT-4o → Claude） |
| **流式输出** | SSE 逐 token 推送，用户实时看到 Agent 思考过程 |
| **Token 管理** | 上下文窗口滑动裁剪 + token 预算控制，防止超限 |

### 1.2 工具系统（Tool Use）

| 能力 | 说明 |
|------|------|
| **内置工具** | 时间查询、计算器、Web 搜索等开箱即用 |
| **自定义工具** | Python 类继承 `BaseTool`，自动注册到 Registry |
| **JSON Schema** | 从 Pydantic model 自动生成 OpenAI function calling schema |
| **热加载** | YAML 配置变更 → 工具启停/参数更新，零重启 |
| **并发执行** | 多个 tool call 并行执行（LLM 支持时） |

### 1.3 会话管理

| 能力 | 说明 |
|------|------|
| **多轮对话** | Session 持久化到 SQLite WAL，支持历史回溯 |
| **上下文裁剪** | 按 token 预算滑动窗口，保留最近有效对话 |
| **并发安全** | Per-session asyncio.Lock，同 session 串行、跨 session 并行 |
| **消息审计** | Append-only 写入，完整记录 user/assistant/tool 消息链 |

### 1.4 Agent 主循环

| 能力 | 说明 |
|------|------|
| **自主工具调用** | LLM 判断是否需要工具 → 执行工具 → 结果回注 → 继续推理 |
| **最大轮次保护** | tool_call 循环上限（默认 10 轮），防止死循环 |
| **错误恢复** | 工具失败 → 结构化错误回注 → LLM 自主决策重试或换方案 |
| **SSE 事件流** | chunk / tool_start / tool_end / done，前端实时渲染 Agent 思考过程 |

---

## 二、工程能力矩阵

### 2.1 配置系统

| 能力 | 说明 |
|------|------|
| **YAML 配置** | 单一 `settings.yaml` 管理所有模块参数 |
| **Pydantic 校验** | 启动时 schema 验证，错误早发现 |
| **热加载** | watchdog 文件监听 → YAML 变更 → 全模块回调刷新 |
| **多环境 Profile** | dev / staging / production profile 切换 |

### 2.2 API Server

| 能力 | 说明 |
|------|------|
| **REST API** | Session CRUD、Agent 同步调用、配置查询 |
| **SSE 流式端点** | `/agent/stream` → EventSource 实时推送 |
| **OpenAPI Spec** | FastAPI 自动生成 Swagger UI + ReDoc |
| **中间件链** | CORS、请求日志、可选 API Key 认证 |

### 2.3 可观测性

| 能力 | 说明 |
|------|------|
| **结构化日志** | JSON 格式，trace_id / session_id 注入 |
| **Prometheus 指标** | HTTP QPS/延迟、LLM token 用量、工具调用统计 |
| **链路追踪** | OpenTelemetry spans，跨模块 trace propagation |
| **健康检查** | `/health` (liveness) + `/ready` (readiness) |

---

## 三、扩展能力规划（Roadmap）

### Phase 1 — MVP（当前设计覆盖）

- [x] 单 Agent 主循环（LLM + tools）
- [x] SSE 流式输出
- [x] Session 持久化 + 上下文管理
- [x] YAML 配置 + 热加载
- [x] 内置工具（时间、计算器、搜索）
- [x] Prometheus 指标采集

### Phase 2 — 增强功能

| 能力 | 说明 | 优先级 |
|------|------|--------|
| **RAG** | 向量数据库 + Embedding，文档检索增强生成 | P0 |
| **多 Agent 编排** | Supervisor-Worker 模式，复杂任务分解 | P1 |
| **Memory System** | 长期记忆（向量存储）+ 短期记忆（对话摘要） | P1 |
| **Plugin Marketplace** | 第三方工具插件市场 + 沙箱执行 | P2 |
| **WebSocket 支持** | 双向实时通信，替代 SSE | P2 |

### Phase 3 — 企业级能力

| 能力 | 说明 | 优先级 |
|------|------|--------|
| **RBAC 权限** | 用户/角色/资源三级权限控制 | P0 |
| **审计日志** | 操作审计 + 合规报告 | P1 |
| **多租户隔离** | Tenant-level data isolation | P1 |
| **Rate Limiting** | 按用户/API key 限流 | P2 |
| **A/B Testing** | 模型/提示词灰度实验 | P3 |

---

## 四、技术栈总览

```
┌─────────────────────────────────────────────────────┐
│                    Frontend                          │
│  (React / Vue / Mobile) ← SSE/WebSocket → API       │
├─────────────────────────────────────────────────────┤
│                  API Server                          │
│  FastAPI + Uvicorn + Pydantic + OpenAPI             │
├──────────┬──────────┬──────────┬────────────────────┤
│ Agent    │ Session  │ Tool     │ LLM Provider       │
│ Core     │ Manager  │ Registry │ (OpenAI/Anthropic) │
├──────────┴──────────┴──────────┴────────────────────┤
│              Config + Observability                  │
│  YAML + Pydantic + Watchdog                          │
│  Prometheus + OpenTelemetry + JSON Logging           │
├─────────────────────────────────────────────────────┤
│              Storage + External                      │
│  SQLite WAL ───→ OpenAI API / Anthropic API          │
└─────────────────────────────────────────────────────┘
```

---

## 五、关键设计决策记录

| 决策 | 选择 | 理由 |
|------|------|------|
| **异步框架** | asyncio + async/await | I/O bound 场景（LLM API、DB），高并发低延迟 |
| **配置格式** | YAML | 人类可读，嵌套结构清晰，适合多环境管理 |
| **数据库** | SQLite WAL | 零运维、单文件部署，WAL mode 支持高并发读写 |
| **流式协议** | SSE (非 WebSocket) | 单向推送足够，HTTP/1.1 兼容性好，前端 EventSource API 原生支持 |
| **LLM 抽象** | ABC + Factory | 多 Provider 切换、Fallback 策略、统一接口契约 |
| **工具注册** | Registry + YAML 控制 | 代码定义能力，配置控制启停/参数，热加载友好 |

---

## 六、性能指标目标

| 指标 | 目标值 | 测量方式 |
|------|--------|---------|
| **API P95 延迟** | < 200ms（不含 LLM 调用） | Prometheus histogram |
| **LLM 首字延迟 (TTFT)** | < 2s | SSE `chunk` event 首个到达时间 |
| **并发会话数** | ≥ 100 sessions/instance | Load test + session_active_gauge |
| **工具执行 P95** | < 500ms（本地工具） | tool_call_duration histogram |
| **配置热加载延迟** | < 1s（YAML save → 生效） | watchdog callback timing |

---

## 七、安全考虑清单

- [ ] API Key 认证（X-API-Key header）
- [ ] 敏感配置遮蔽（API key 不写入日志/不暴露到 `/config`）
- [ ] SQLite 文件权限限制（owner-only read/write）
- [ ] Tool execution sandboxing（未来：第三方工具隔离执行）
- [ ] Rate limiting（按 API key / IP 限流）
- [ ] Input validation（消息长度限制、prompt injection 防护）
