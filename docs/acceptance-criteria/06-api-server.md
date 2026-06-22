# API Server — 验收标准

## API-01: POST /api/chat — 同步对话

**优先级**: P0

- **Given** AgentCore 已初始化，服务运行中
- **When** `POST /api/chat` 发送 `{"message": "hello", "session_id": "xxx"}`
- **Then** 返回 HTTP 200 + JSON `{"content": "...", "session_id": "xxx", "usage": {...}}`；无效请求体返回 422
- **验证方式**: 集成测试（TestClient）

## API-02: POST /api/chat — SSE 流式

**优先级**: P0

- **Given** AgentCore 支持 streaming
- **When** `POST /api/chat` 发送 `{"message": "hello", "stream": true}`
- **Then** 返回 HTTP 200 + `text/event-stream`；每个 chunk 以 `data: {...}\n\n` 格式输出；最后一个 chunk 包含 `[DONE]` 或 `finish_reason`
- **验证方式**: 集成测试 — 断言 Content-Type 和 SSE 格式

## API-03: GET /api/sessions — 会话列表

**优先级**: P1

- **Given** session_manager 中存在多个 sessions
- **When** `GET /api/sessions?user_id=xxx`
- **Then** 返回 JSON 数组，每个元素包含 `session_id`、`title`、`created_at`、`message_count`
- **验证方式**: 集成测试

## API-04: GET /api/sessions/{id} — 会话详情

**优先级**: P1

- **Given** session 存在
- **When** `GET /api/sessions/{session_id}`
- **Then** 返回完整 session 对象（含 messages）；不存在的 ID 返回 404
- **验证方式**: 集成测试

## API-05: DELETE /api/sessions/{id} — 删除会话

**优先级**: P1

- **Given** session 存在
- **When** `DELETE /api/sessions/{session_id}`
- **Then** 返回 200；再次 GET 返回 404
- **验证方式**: 集成测试

## API-06: WebSocket /ws/chat — 实时对话

**优先级**: P1

- **Given** WebSocket 端点已注册
- **When** 客户端连接 `/ws/chat`，发送 JSON `{"message": "hello"}`
- **Then** 服务端逐 chunk 返回 JSON `{"type": "chunk", "content": "..."}`；最后发送 `{"type": "done"}`
- **验证方式**: 集成测试（websockets test client）

## API-07: WebSocket — 错误处理

**优先级**: P1

- **Given** WebSocket 连接已建立
- **When** 发送无效 JSON 或空消息
- **Then** 服务端返回 `{"type": "error", "message": "..."}`，不断开连接
- **验证方式**: 集成测试

## API-08: GET /health — 健康检查

**优先级**: P0

- **Given** 服务运行中
- **When** `GET /health`
- **Then** 返回 HTTP 200 + `{"status": "ok", "version": "...", "uptime_seconds": N}`
- **验证方式**: 集成测试

## API-09: GET /ready — 就绪检查

**优先级**: P1

- **Given** Redis/LLM 依赖已配置
- **When** `GET /ready`
- **Then** 检查各依赖连通性；全部正常返回 200 + `{"status": "ready", "dependencies": {...}}`；任一失败返回 503
- **验证方式**: 集成测试 — mock 依赖状态

## API-10: CORS 中间件

**优先级**: P1

- **Given** 服务配置了 CORS origins
- **When** 浏览器发起跨域请求（含 Origin header）
- **Then** 响应包含 `Access-Control-Allow-Origin`；OPTIONS preflight 返回 200
- **验证方式**: 集成测试

## API-11: Request Validation

**优先级**: P0

- **Given** Pydantic request models
- **When** 发送缺少必填字段的请求（如空 message）
- **Then** 返回 HTTP 422 + 详细错误信息，不触发 AgentCore
- **验证方式**: 集成测试

## API-12: Request ID 追踪

**优先级**: P2

- **Given** 任意 API 请求
- **When** 请求处理中
- **Then** 响应 header 包含 `X-Request-ID`；日志中使用该 ID 关联所有相关条目
- **验证方式**: 集成测试 — 断言 response headers

## API-13: Lifespan 生命周期管理

**优先级**: P0

- **Given** FastAPI app 定义 lifespan
- **When** 服务启动 / 停止
- **Then** 启动时初始化 ConfigManager → LLM → ToolRegistry → AgentCore；停止时清理资源（连接池、定时任务）
- **验证方式**: 集成测试 — 断言 startup/shutdown 顺序
