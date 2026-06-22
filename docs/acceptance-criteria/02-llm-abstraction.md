# LLM 抽象层 — 验收标准

## LLM-01: BaseLLM 抽象基类

**优先级**: P0

- **Given** `BaseLLM` 定义了 `chat()`、`chat_stream()`、`count_tokens()` 三个抽象方法
- **When** 尝试直接实例化 `BaseLLM`
- **Then** 抛出 `TypeError`（抽象类不可实例化）
- **验证方式**: 单元测试

## LLM-02: OpenAI Provider

**优先级**: P0

- **Given** `ProviderConfig(provider="openai", model="gpt-4o")` + 有效 API key
- **When** 调用 `create_llm(config).chat(messages)`
- **Then** 返回 `LLMResponse`，包含 `content`、`finish_reason`、`usage`；支持 tool_calls 字段
- **验证方式**: 单元测试（mock HTTP）+ 集成测试（真实 API）

## LLM-03: Anthropic Provider

**优先级**: P0

- **Given** `ProviderConfig(provider="anthropic", model="claude-sonnet-4-20250514")`
- **When** 调用 `chat()` 包含 tool_calls 的 messages
- **Then** 正确转换 OpenAI format → Anthropic format，返回结果转回统一格式
- **验证方式**: 单元测试（mock）+ 集成测试

## LLM-04: Ollama Provider

**优先级**: P1

- **Given** `ProviderConfig(provider="ollama", model="qwen3:8b")` + 本地 Ollama 服务运行中
- **When** 调用 `chat()`
- **Then** 通过 HTTP 请求 `http://localhost:11434/api/chat`，返回统一格式响应
- **验证方式**: 集成测试（需本地 Ollama）

## LLM-05: Streaming 流式响应

**优先级**: P0

- **Given** 任意 ProviderConfig
- **When** 调用 `chat_stream(messages)`
- **Then** 返回 async generator，每次 yield `LLMStreamChunk(content=..., finish_reason=None/stop)`；最后一个 chunk 的 `finish_reason` 非 None
- **验证方式**: 单元测试（mock SSE stream）

## LLM-06: Token 计数

**优先级**: P1

- **Given** 任意 ProviderConfig
- **When** 调用 `count_tokens(text)` / `count_messages(messages)`
- **Then** 返回 int，误差 < 5%（使用 tiktoken 或近似算法）
- **验证方式**: 单元测试 — 已知文本断言 token 数范围

## LLM-07: FailoverLLM 故障转移

**优先级**: P1

- **Given** `FailoverLLM([primary=OpenAILLM, fallback=OllamaLLM])`
- **When** primary 抛出 `APIError`（5xx / timeout）
- **Then** 自动切换到 fallback provider，返回结果；记录 warning 日志
- **验证方式**: 单元测试 — mock primary 抛异常，断言 fallback 被调用

## LLM-08: FailoverLLM 流式降级

**优先级**: P2

- **Given** `FailoverLLM` 配置了 primary + fallback
- **When** 调用 `chat_stream()`，primary 失败
- **Then** 降级为 `chat()` 一次性返回，不丢失内容
- **验证方式**: 单元测试

## LLM-09: Factory 创建

**优先级**: P0

- **Given** `ProviderConfig(provider="openai", ...)`
- **When** 调用 `create_llm(config)`
- **Then** 返回对应 Provider 实例；未知 provider 抛出 `ValueError`
- **验证方式**: 单元测试

## LLM-10: Config Bridge YAML → ProviderConfig

**优先级**: P1

- **Given** YAML llm section 包含 `provider`、`model`、`api_key`（含 `${ENV_VAR}`）
- **When** 调用 `create_llm_from_yaml(yaml_dict)`
- **Then** 环境变量被替换，返回正确配置的 BaseLLM 实例
- **验证方式**: 单元测试

## LLM-11: Config Bridge Fallback Providers

**优先级**: P1

- **Given** YAML llm section 包含 `fallback_providers: [ollama]`
- **When** 调用 `create_llm_from_yaml(yaml_dict)`
- **Then** 返回 `FailoverLLM(primary=..., fallbacks=[...])`
- **验证方式**: 单元测试

## LLM-12: ProviderConfig 脱敏

**优先级**: P0

- **Given** `ProviderConfig(api_key="sk-secret")`
- **When** 打印或日志输出该对象
- **Then** `api_key` 显示为 `sk-***`，不泄露真实值
- **验证方式**: 单元测试 — 断言 `__repr__`

## LLM-13: 新增 Provider 扩展性

**优先级**: P2

- **Given** 新 Provider 继承 `BaseLLM` 并实现三个抽象方法
- **When** 在 factory.py 中注册到 `_provider_map`
- **Then** `create_llm()` 可正确创建该 Provider，无需修改 AgentCore
- **验证方式**: 单元测试 — 创建 mock provider 验证工厂模式
