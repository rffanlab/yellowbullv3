# 可观测性 — 验收标准

## OBS-01: JSON 格式日志

**优先级**: P0

- **Given** 服务运行中，日志级别 INFO
- **When** 处理一次 chat 请求
- **Then** stdout 输出 JSON 行，包含 `timestamp`、`level`、`message`、`request_id`、`session_id`（如有）；非 JSON 格式日志为 0
- **验证方式**: 集成测试 — 捕获 stdout 并解析 JSON

## OBS-02: 结构化日志字段

**优先级**: P1

- **Given** AgentCore 执行工具调用
- **When** 记录日志
- **Then** 包含 `tool_name`、`duration_ms`、`success` 等结构化字段；敏感信息（API key）被脱敏
- **验证方式**: 单元测试 — 断言日志字段

## OBS-03: Prometheus Metrics — 请求计数

**优先级**: P1

- **Given** prometheus middleware 已注册
- **When** 处理 5 次 chat 请求（2 成功，3 失败）
- **Then** `http_requests_total{method="POST",path="/api/chat",status="200"}` = 2；`status="500"` = 3
- **验证方式**: 集成测试 — 断言 metrics endpoint

## OBS-04: Prometheus Metrics — 延迟直方图

**优先级**: P1

- **Given** prometheus middleware 已注册
- **When** 处理请求（不同耗时）
- **Then** `http_request_duration_seconds_bucket` 包含各 bucket 计数；`_sum` 和 `_count` 正确
- **验证方式**: 集成测试

## OBS-05: Prometheus Metrics — Token 使用量

**优先级**: P1

- **Given** AgentCore 完成对话，LLM 返回 usage 信息
- **When** 请求处理完成
- **Then** `llm_tokens_total{type="input"}` 和 `{type="output"}` 递增；`llm_requests_total` 递增
- **验证方式**: 集成测试 — mock LLM usage

## OBS-06: Prometheus Metrics — 工具调用统计

**优先级**: P2

- **Given** AgentCore 执行了多个工具
- **When** 请求处理完成
- **Then** `tool_calls_total{tool_name="web_search"}` 按工具名递增；`tool_call_duration_seconds` 记录耗时
- **验证方式**: 集成测试

## OBS-07: Prometheus Metrics Endpoint

**优先级**: P1

- **Given** 服务运行中
- **When** `GET /metrics`
- **Then** 返回 HTTP 200 + Prometheus text format；包含所有注册的 metrics
- **验证方式**: 集成测试

## OBS-08: OpenTelemetry Tracing — Span 创建

**优先级**: P1

- **Given** opentelemetry 已配置，exporter 指向 collector
- **When** 处理一次 chat 请求（含工具调用）
- **Then** 生成 root span `chat_request`；子 span 包含 `tool_execution.web_search`、`llm_chat` 等
- **验证方式**: 集成测试 — 使用 in-memory exporter 断言 spans

## OBS-09: OpenTelemetry Tracing — Context Propagation

**优先级**: P1

- **Given** 请求携带 `traceparent` header（W3C TraceContext）
- **When** 处理请求
- **Then** 生成的 span 使用传入的 trace_id；span.parent_span_id = root span id
- **验证方式**: 集成测试 — 注入 trace context

## OBS-10: OpenTelemetry Tracing — Error Recording

**优先级**: P2

- **Given** 请求处理中发生异常
- **When** span 结束
- **Then** span status = ERROR；包含 exception event（类型、消息、stacktrace）
- **验证方式**: 集成测试 — mock 异常场景

## OBS-11: Log Level 配置

**优先级**: P1

- **Given** `settings.yaml` 中 `logging.level` 设置为 DEBUG/WARNING/ERROR
- **When** 服务启动
- **Then** root logger 级别正确设置；各子模块遵循父级别
- **验证方式**: 单元测试

## OBS-12: Metrics 清理与重置

**优先级**: P2

- **Given** metrics endpoint 被多次访问
- **When** `POST /metrics/reset`（管理接口，需认证）
- **Then** counter/gauge 重置为 0；histogram buckets 清空
- **验证方式**: 集成测试
