# 安全治理 — 验收标准

## SEC-01: Rate Limiting — 按用户限流

**优先级**: P0

- **Given** rate limiter 配置 `max_requests=10, window_seconds=60`
- **When** 同一 user_id 在 60s 内发送 11 个请求
- **Then** 第 11 个请求返回 HTTP 429 + `{"detail": "Rate limit exceeded"}`；包含 `Retry-After` header
- **验证方式**: 集成测试 — 快速连续请求

## SEC-02: Rate Limiting — 按 IP 限流

**优先级**: P1

- **Given** rate limiter 配置了 per-IP 限制
- **When** 同一 IP（不同 user_id）超过限制
- **Then** 返回 429；不同 IP 互不影响
- **验证方式**: 集成测试 — mock client host

## SEC-03: Rate Limiting — Token Bucket 算法

**优先级**: P1

- **Given** token bucket rate limiter（max_tokens=10, refill_rate=1/s）
- **When** 连续发送 10 个请求消耗完 tokens，等待 1s 后发送第 11 个
- **Then** 第 11 个请求成功（token 已补充）；不采用固定窗口（避免边界突发）
- **验证方式**: 单元测试

## SEC-04: Input Validation — Prompt Injection 检测

**优先级**: P0

- **Given** input sanitizer 已注册为 middleware
- **When** 用户消息包含 "忽略之前的指令" / "ignore previous instructions" 等注入模式
- **Then** 记录 warning 日志；消息被标记或拒绝（返回 400）；不传递给 LLM
- **验证方式**: 集成测试 — 已知注入 payload

## SEC-05: Input Validation — 长度限制

**优先级**: P0

- **Given** 配置 `max_input_length=8192`
- **When** 用户消息超过限制
- **Then** 返回 HTTP 400 + 明确错误信息；不传递给 LLM
- **验证方式**: 集成测试

## SEC-06: Output Filtering — 敏感内容检测

**优先级**: P1

- **Given** output filter 已启用
- **When** LLM 响应包含检测到的高风险内容（如代码注入、恶意链接）
- **Then** 响应被拦截或标记；返回安全提示给用户；记录 security event 日志
- **验证方式**: 集成测试 — mock LLM 返回敏感内容

## SEC-07: API Key 认证

**优先级**: P0

- **Given** 服务配置了 `api_keys`（settings.yaml）
- **When** 请求未携带 `Authorization: Bearer <key>` 或 key 无效
- **Then** 返回 HTTP 401；有效 key 正常处理
- **验证方式**: 集成测试

## SEC-08: Audit Logging — API Key 使用审计

**优先级**: P1

- **Given** audit logger 已启用
- **When** 请求通过认证
- **Then** 记录 `api_key_usage` event，包含 key_id（脱敏）、user_agent、ip、timestamp、endpoint；不包含请求体敏感内容
- **验证方式**: 集成测试 — 断言日志条目

## SEC-09: Audit Logging — LLM 调用审计

**优先级**: P1

- **Given** audit logger 已启用
- **When** AgentCore 调用 LLM
- **Then** 记录 `llm_call` event，包含 model、input_tokens、output_tokens、duration_ms、cost（如可计算）；messages 内容不记录（体积大+敏感）
- **验证方式**: 集成测试

## SEC-10: Audit Logging — 工具调用审计

**优先级**: P1

- **Given** audit logger 已启用
- **When** AgentCore 执行工具
- **Then** 记录 `tool_call` event，包含 tool_name、arguments（脱敏）、result_summary、duration_ms
- **验证方式**: 集成测试

## SEC-11: Audit Log 存储与查询

**优先级**: P2

- **Given** audit events 写入 SQLite/文件
- **When** `GET /admin/audit?start=...&end=...`（管理接口）
- **Then** 返回时间范围内的审计事件；支持按 event_type 过滤
- **验证方式**: 集成测试

## SEC-12: Content Policy — 工具白名单

**优先级**: P1

- **Given** settings.yaml 中 `tools.allowed` 配置了白名单
- **When** LLM 请求执行不在白名单中的工具
- **Then** 拒绝执行，返回错误 ToolResult；记录 security event
- **验证方式**: 集成测试
