# Agent Core — 验收标准

## AGT-01: AgentCore 初始化

**优先级**: P0

- **Given** `BaseLLM` 实例、`ToolRegistry` 实例、`SessionManager` 实例
- **When** 创建 `AgentCore(llm, tool_registry, session_manager)`
- **Then** 三个依赖正确注入；可选参数 `max_tool_rounds` 默认为合理值（如 5）
- **验证方式**: 单元测试

## AGT-02: chat() — 无工具调用

**优先级**: P0

- **Given** AgentCore 实例，LLM 返回纯文本响应（无 tool_calls）
- **When** 调用 `chat(user_message, session_id)`
- **Then** 返回 `AgentResponse(content=..., tool_calls=[], finished=True)`；用户消息和 AI 回复都追加到 session
- **验证方式**: 单元测试（mock LLM）

## AGT-03: chat() — 工具调用循环

**优先级**: P0

- **Given** AgentCore 实例，LLM 返回包含 tool_calls 的响应
- **When** 调用 `chat(user_message, session_id)`
- **Then** 自动执行工具 → 将结果追加到 messages → 再次调用 LLM；循环直到 LLM 不再请求工具或达到 `max_tool_rounds`
- **验证方式**: 单元测试（mock LLM + mock tools）

## AGT-04: chat() — 多轮工具调用

**优先级**: P1

- **Given** LLM 连续 3 轮返回不同的 tool_calls
- **When** 调用 `chat()`
- **Then** 每轮工具结果正确追加到消息链；最终返回包含所有中间结果的完整响应
- **验证方式**: 单元测试 — mock LLM 依次返回不同 tool_calls

## AGT-05: chat() — max_tool_rounds 限制

**优先级**: P1

- **Given** `max_tool_rounds=2`，LLM 持续返回 tool_calls
- **When** 调用 `chat()`
- **Then** 执行 2 轮工具后停止，返回当前结果并记录 warning；不无限循环
- **验证方式**: 单元测试

## AGT-06: chat_stream() — 流式输出

**优先级**: P0

- **Given** AgentCore 实例，LLM 支持 streaming
- **When** 调用 `chat_stream(user_message, session_id)`
- **Then** 返回 async generator；先 yield 工具执行结果（如有），然后逐 chunk yield LLM 文本流
- **验证方式**: 单元测试（mock LLM chat_stream）

## AGT-07: chat_stream() — 工具调用后流式

**优先级**: P1

- **Given** LLM 先返回 tool_calls，执行后再返回文本流
- **When** 调用 `chat_stream()`
- **Then** yield 工具结果 → 然后逐 chunk yield 最终回复；stream 结束后 session 已完整保存
- **验证方式**: 单元测试

## AGT-08: chat() — LLM 异常处理

**优先级**: P1

- **Given** LLM `chat()` 抛出 `APIError`
- **When** 调用 AgentCore `chat()`
- **Then** 返回 `AgentResponse(content="错误提示", finished=True, is_error=True)`；不向 session 追加无效消息
- **验证方式**: 单元测试 — mock LLM 抛异常

## AGT-09: chat() — 工具执行失败处理

**优先级**: P1

- **Given** 工具执行返回 `ToolResult(is_error=True)`
- **When** AgentCore 处理工具结果
- **Then** 错误结果追加到 messages；LLM 收到错误信息后可自行决定重试或结束
- **验证方式**: 单元测试 — mock tool 返回错误

## AGT-10: Session 消息完整性

**优先级**: P0

- **Given** `chat()` 完成（含多轮工具调用）
- **When** 从 session_manager 获取 session messages
- **Then** 包含 user message → assistant tool_calls → tool results → final response，顺序正确
- **验证方式**: 单元测试 — 断言 messages 列表结构和顺序

## AGT-11: System Prompt 注入

**优先级**: P1

- **Given** AgentCore 初始化时传入 `system_prompt`
- **When** 调用 `chat()`
- **Then** system message 作为第一条消息发送给 LLM；session 中已有 system message 时不重复添加
- **验证方式**: 单元测试

## AGT-12: Token 预算控制

**优先级**: P2

- **Given** AgentCore 配置了 `max_input_tokens`
- **When** session messages + user_message 超过 token 限制
- **Then** 自动裁剪最早的消息对直到满足限制；记录 info 日志
- **验证方式**: 单元测试 — mock LLM count_tokens
