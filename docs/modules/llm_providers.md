# LLM Providers 模块详细设计

## 1. 概述

LLM Provider 层提供统一的 LLM 调用接口，屏蔽不同供应商 API 的差异。通过工厂模式根据配置动态创建对应的 Provider 实例。

**对应源码:** `llm/base.py`, `llm/factory.py`, `llm/providers/openai_provider.py`, `llm/providers/anthropic_provider.py`

### 职责
- 统一 LLM 调用接口（同步 chat + 流式 stream_chat）
- Token 用量统计
- 多 Provider 适配与切换
- Function Calling 格式转换

## 2. 基础抽象

### Message (LLM 消息)

```python
@dataclass
class Message:
    role: Role              # SYSTEM, USER, ASSISTANT, TOOL
    content: str | None     # 消息内容
    tool_calls: list[dict] | None      # Assistant 的工具调用
    tool_call_id: str | None           # Tool 结果对应的调用 ID
```

### LLMResponse (响应模型)

```python
@dataclass
class LLMResponse:
    content: str | None                     # 文本回复内容
    tool_calls: list[dict[str, Any]] = []   # 工具调用列表
    usage: dict[str, int] | None = None     # Token 用量统计
```

### BaseLLM (抽象基类)

```python
class BaseLLM(ABC):
    @abstractmethod
    async def chat(self, messages: list[Message], tools: list[dict] | None = None) -> LLMResponse
    @abstractmethod
    def count_tokens(self, text: str) -> int
```

## 3. Provider 实现

### OpenAIProvider

**配置字段:**

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `api_key` | string | — | OpenAI API Key |
| `model` | string | `"gpt-4o-mini"` | 模型名称 |
| `max_tokens` | int | `8192` | 最大输出 token 数 |
| `temperature` | float | `0.7` | 采样温度 |

**chat 方法:**
- 使用 `openai.AsyncOpenAI` SDK
- 将内部 Message 转换为 OpenAI 格式字典
- 支持 Function Calling（tools 参数透传）
- 返回 LLMResponse，包含 content、tool_calls、usage

**stream_chat 方法:**
```python
async def stream_chat(
    self, messages: list[Message], tools: list[dict] | None = None
) -> AsyncGenerator[str, None]
```
- 使用 `stream=True` 参数启用 SSE 流式响应
- 逐块提取 `delta.content`，跳过 None 值
- 仅支持文本输出，不支持工具调用的流式响应

### AnthropicProvider

**配置字段:**

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `api_key` | string | — | Anthropic API Key |
| `model` | string | `"claude-sonnet-4-20250514"` | 模型名称 |
| `max_tokens` | int | `8192` | 最大输出 token 数 |

**特殊处理:**
- **System Prompt:** Anthropic API 通过独立的 `system` 参数传递，而非 messages[0]。代码在调用前从消息列表中分离 System 消息。
- **Tool 格式:** 使用与 OpenAI 兼容的 JSON Schema 格式

## 4. Factory 模式

### create_llm

```python
def create_llm(provider: str, config: dict[str, Any]) -> BaseLLM
```

根据 provider 名称创建对应的 LLM 实例。

**支持的 Provider:**

| Provider | 类名 | 配置 key |
|---|---|---|
| `openai` | OpenAIProvider | `"openai"` |
| `anthropic` | AnthropicProvider | `"anthropic"` |

## 5. Token 计数

### count_tokens

```python
def count_tokens(self, text: str) -> int
```

估算文本的 token 数量。当前实现使用简单启发式：`len(text) // 4`。

**用途:**
- Agent 记录每次 LLM 调用的 prompt/completion tokens
- 返回给客户端作为 usage 信息
- 未来可用于上下文窗口控制

## 6. 依赖关系

```
llm/
    ├── base.py          (抽象定义)
    ├── factory.py       (工厂创建)
    └── providers/
        ├── openai_provider.py     (openai SDK)
        └── anthropic_provider.py  (anthropic SDK)
```

## 7. 注意事项

- Provider 名称需与 `config/settings.py` 中的配置 key 一致
- API Key 等敏感信息通过环境变量注入，禁止硬编码
- Token 计数为近似值，精确计数需接入各 Provider 的官方 token counter
- 流式响应目前仅 OpenAI 实现，Anthropic 待补充
- 未来可扩展更多 Provider（如 Google Gemini、Ollama 本地模型）
