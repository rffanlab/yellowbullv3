# LLM 抽象层详细设计

## 1. 设计目标

| 目标 | 说明 |
|------|------|
| **统一协议** | 所有 provider 收敛到同一接口，Agent 侧零感知差异 |
| **即插即用** | 新增 provider = 实现基类 + 注册工厂，不影响已有代码 |
| **Function Calling 适配** | 各 provider FC 格式不同，内部统一为 OpenAI 格式 |
| **流式支持** | SSE 流式输出，所有 provider 统一 `AsyncIterator[str]` |
| **故障降级** | 主 provider 失败自动切 fallback，透明重试 |
| **Token 估算** | 统一 token counting 接口，用于上下文窗口控制 |

---

## 2. 协议设计（内部统一格式）

选择 OpenAI Function Calling 格式作为内部标准——生态最广、文档最全。

### 2.1 消息协议 `llm/protocol.py`

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True)
class Message:
    """
    不可变消息对象。

    role=SYSTEM:   content 必填，其余忽略
    role=USER:     content 必填
    role=ASSISTANT: content 和 tool_calls 至少一个非空
    role=TOOL:     tool_call_id + content 必填
    """
    role: Role
    content: str | None = None
    tool_calls: list["ToolCall"] | None = None
    tool_call_id: str | None = None


@dataclass(frozen=True)
class ToolCall:
    """LLM 发出的工具调用请求（对应 OpenAI chat.completion.tool_choice）"""
    id: str                                    # 唯一 ID，用于关联 tool response
    name: str                                  # 工具名
    arguments: dict[str, Any]                  # 已解析的 JSON dict


@dataclass(frozen=True)
class ToolDefinition:
    """
    发送给 LLM 的工具定义。

    内部统一使用 OpenAI function calling 格式：
    {
        "name": "...",
        "description": "...",
        "parameters": { JSON Schema }
    }
    """
    name: str
    description: str
    parameters: dict[str, Any]   # JSON Schema

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class UsageInfo:
    """Token 消耗统计"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @property
    def cost_estimate(self) -> float:
        """粗略费用估算（实际应在 provider 层按模型计费）"""
        return 0.0


@dataclass
class LLMResponse:
    """LLM 推理结果"""
    content: str | None                          # 文本回复（可能为 None，当只有 tool_calls 时）
    tool_calls: list[ToolCall] | None = None     # 工具调用列表
    finish_reason: str | None = None             # "stop" | "tool_calls" | "length" | "content_filter"
    usage: UsageInfo | None = None


@dataclass
class StreamChunk:
    """流式输出片段"""
    delta_content: str | None = None             # 文本增量
    delta_tool_call: ToolCall | None = None      # 工具调用增量（较少用）
    finish_reason: str | None = None
    usage: UsageInfo | None = None               # 仅在最后一个 chunk 中非空
```

### 2.2 协议设计考量

| 决策 | 理由 |
|------|------|
| `frozen=True`（不可变） | 消息一旦创建不应被修改，避免并发 bug；每次变更创建新对象 |
| `ToolCall.arguments` 存 dict 而非 str | LLM 返回的是 JSON string，内部立即解析为 dict，下游直接用 |
| `UsageInfo` 独立出来 | 便于日志记录、成本统计、监控告警 |
| `StreamChunk.delta_content` 用 str | 流式只传增量文本，不传完整消息，节省内存 |

---

## 3. 基类设计 `llm/base.py`

```python
from abc import ABC, abstractmethod
from typing import AsyncIterator
from llm.protocol import (
    Message, ToolDefinition, LLMResponse, StreamChunk, UsageInfo,
)


class BaseLLM(ABC):
    """
    所有 LLM provider 必须实现的接口。

    设计原则：
    - 所有方法均为 async，统一异步 IO
    - 输入输出使用内部协议对象，不暴露 provider SDK 类型
    - 每个实例绑定一个 provider + model，线程安全
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """提供商标识，如 "openai"、"anthropic""""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """当前模型名，如 "gpt-4o""""
        ...

    # ==================== 核心方法 ====================

    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """
        非流式对话。

        Args:
            messages:   消息列表（含 system/user/assistant/tool）
            tools:      工具定义列表（为空则不做 function calling）
            temperature: 覆盖默认温度值
            max_tokens:  覆盖默认最大 token 数

        Returns:
            LLMResponse（content / tool_calls 至少一个非空）
        """
        ...

    @abstractmethod
    async def chat_stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """
        流式对话。

        Yields:
            StreamChunk（最后一个 chunk 包含 usage）

        Usage:
            async for chunk in llm.chat_stream(messages):
                print(chunk.delta_content, end="", flush=True)
        """
        ...

    # ==================== 辅助方法 ====================

    @abstractmethod
    async def count_tokens(self, messages: list[Message]) -> int:
        """估算消息列表的 token 数（用于上下文窗口控制）"""
        ...

    async def health_check(self) -> bool:
        """
        健康检查。默认实现：发一条最短请求看是否超时。
        可被子类覆盖为更高效的检查方式。
        """
        import asyncio
        try:
            await asyncio.wait_for(
                self.chat([Message(role=Role.SYSTEM, content="ok")], max_tokens=1),
                timeout=10.0,
            )
            return True
        except Exception:
            return False

    def __repr__(self):
        return f"<{self.__class__.__name__} provider={self.provider_name} model={self.model_name}>"
```

---

## 4. Provider 实现

### 4.1 OpenAI `llm/openai_llm.py`

```python
from typing import AsyncIterator
from openai import AsyncOpenAI, APIStatusError, APITimeoutError

from llm.base import BaseLLM
from llm.protocol import (
    Message, ToolDefinition, LLMResponse, StreamChunk,
    ToolCall, UsageInfo, Role,
)


class OpenAILLM(BaseLLM):
    """
    OpenAI 兼容 provider。

    支持：
    - OpenAI 官方 API
    - 任何兼容 OpenAI 接口的服务（vLLM、Ollama with openai compat、本地部署等）
      → 通过 base_url 配置即可
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        base_url: str | None = None,
        organization: str | None = None,
    ):
        self._api_key = api_key
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
        )

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self._model

    # ---------- 协议转换 ----------

    @staticmethod
    def _to_openai_msg(msg: Message) -> dict:
        if msg.role == Role.TOOL:
            return {
                "role": "tool",
                "tool_call_id": msg.tool_call_id,
                "content": msg.content or "",
            }
        result = {"role": msg.role.value, "content": msg.content}
        if msg.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in msg.tool_calls
            ]
        return result

    @staticmethod
    def _from_openai_choice(choice) -> LLMResponse:
        msg = choice.message
        tool_calls = None
        if msg.tool_calls:
            import json
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments),
                )
                for tc in msg.tool_calls
            ]
        usage = None
        if choice.usage:
            usage = UsageInfo(
                prompt_tokens=choice.usage.prompt_tokens,
                completion_tokens=choice.usage.completion_tokens,
                total_tokens=choice.usage.total_tokens,
            )
        return LLMResponse(
            content=msg.content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason,
            usage=usage,
        )

    # ---------- 核心方法 ----------

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        kwargs = {
            "model": self._model,
            "messages": [self._to_openai_msg(m) for m in messages],
            "temperature": temperature if temperature is not None else self._temperature,
            "max_tokens": max_tokens or self._max_tokens,
        }
        if tools:
            kwargs["tools"] = [t.to_dict() for t in tools]
            kwargs["tool_choice"] = "auto"

        resp = await self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0]

        response = self._from_openai_choice(choice)
        if resp.usage:
            response.usage = UsageInfo(
                prompt_tokens=resp.usage.prompt_tokens,
                completion_tokens=resp.usage.completion_tokens,
                total_tokens=resp.usage.total_tokens,
            )
        return response

    async def chat_stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamChunk]:
        kwargs = {
            "model": self._model,
            "messages": [self._to_openai_msg(m) for m in messages],
            "temperature": temperature if temperature is not None else self._temperature,
            "max_tokens": max_tokens or self._max_tokens,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = [t.to_dict() for t in tools]
            kwargs["tool_choice"] = "auto"

        stream = await self._client.chat.completions.create(**kwargs)
        async for chunk in stream:
            choice = chunk.choices[0]
            yield StreamChunk(
                delta_content=choice.delta.content,
                finish_reason=choice.finish_reason,
                usage=(UsageInfo(
                    prompt_tokens=chunk.usage.prompt_tokens,
                    completion_tokens=chunk.usage.completion_tokens,
                    total_tokens=chunk.usage.total_tokens,
                ) if chunk.usage else None),
            )

    async def count_tokens(self, messages: list[Message]) -> int:
        import tiktoken
        try:
            enc = tiktoken.encoding_for_model(self._model)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")   # gpt-4 默认 encoder

        total = 0
        for msg in messages:
            # 粗略估算：role + content +  overhead
            total += 4   # 每消息 overhead
            total += len(enc.encode(msg.role.value))
            if msg.content:
                total += len(enc.encode(msg.content))
            if msg.tool_calls:
                import json
                total += len(enc.encode(json.dumps([
                    {"name": tc.name, "args": tc.arguments}
                    for tc in msg.tool_calls
                ])))
        return total


# ---------- 兼容 OpenAI 接口的通用 wrapper ----------

class OpenAICompatLLM(OpenAILLM):
    """
    用于任何兼容 OpenAI API 的服务。

    Usage:
        llm = OpenAICompatLLM(
            api_key="dummy",
            model="qwen2.5-72b",
            base_url="http://localhost:8000/v1",
        )
    """
    pass   # 完全复用 OpenAILLM，仅语义不同
```

### 4.2 Anthropic `llm/anthropic_llm.py`

```python
from typing import AsyncIterator
import json
from anthropic import AsyncAnthropic

from llm.base import BaseLLM
from llm.protocol import (
    Message, ToolDefinition, LLMResponse, StreamChunk,
    ToolCall, UsageInfo, Role,
)


class AnthropicLLM(BaseLLM):
    """
    Anthropic Claude provider。

    注意：Anthropic 的 tool use 协议与 OpenAI 不同，需要双向转换：
    - OpenAI format: {"type": "function", "function": {...}}
    - Anthropic format: {"name": "...", "input_schema": {...}}
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ):
        self._api_key = api_key
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client = AsyncAnthropic(api_key=api_key)

    @property
    def provider_name(self) -> str:
        return "anthropic"

    @property
    def model_name(self) -> str:
        return self._model

    # ---------- 协议转换 ----------

    @staticmethod
    def _tools_to_anthropic(tools: list[ToolDefinition]) -> list[dict]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": {"type": "object", **t.parameters},
            }
            for t in tools
        ]

    def _messages_to_anthropic(self, messages: list[Message]) -> tuple[str, list[dict]]:
        """
        Anthropic 要求 system 单独传，不在 messages 中。
        返回 (system_prompt, messages_list)
        """
        system = ""
        conv = []
        current_user = None

        for msg in messages:
            if msg.role == Role.SYSTEM:
                system = msg.content or ""
                continue

            if msg.role == Role.USER:
                current_user = {"role": "user", "content": []}
                conv.append(current_user)
            elif msg.role == Role.ASSISTANT:
                content = []
                if msg.content:
                    content.append({"type": "text", "text": msg.content})
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        content.append({
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments,
                        })
                conv.append({"role": "assistant", "content": content})
            elif msg.role == Role.TOOL:
                # Anthropic tool_result 必须跟在 assistant 之后，放在 user 消息中
                conv.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id,
                        "content": msg.content or "",
                    }],
                })

        return system, conv

    @staticmethod
    def _response_from_anthropic(resp) -> LLMResponse:
        content = None
        tool_calls = []

        for block in resp.content:
            if block.type == "text":
                content = (content or "") + block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input,
                ))

        usage = UsageInfo(
            prompt_tokens=resp.usage.input_tokens,
            completion_tokens=resp.usage.output_tokens,
            total_tokens=resp.usage.input_tokens + resp.usage.output_tokens,
        )

        return LLMResponse(
            content=content or None,
            tool_calls=tool_calls or None,
            finish_reason=resp.stop_reason,
            usage=usage,
        )

    # ---------- 核心方法 ----------

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        system, conv = self._messages_to_anthropic(messages)

        kwargs = {
            "model": self._model,
            "max_tokens": max_tokens or self._max_tokens,
            "temperature": temperature if temperature is not None else self._temperature,
            "messages": conv,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = self._tools_to_anthropic(tools)

        resp = await self._client.messages.create(**kwargs)
        return self._response_from_anthropic(resp)

    async def chat_stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamChunk]:
        system, conv = self._messages_to_anthropic(messages)

        kwargs = {
            "model": self._model,
            "max_tokens": max_tokens or self._max_tokens,
            "temperature": temperature if temperature is not None else self._temperature,
            "messages": conv,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = self._tools_to_anthropic(tools)

        stream = await self._client.messages.stream(**kwargs)
        async with stream:
            async for event in stream:
                if event.type == "content_block_delta":
                    yield StreamChunk(delta_content=event.delta.text)
                elif event.type == "message_stop" and hasattr(stream, 'final_message'):
                    final = stream.final_message
                    usage = UsageInfo(
                        prompt_tokens=final.usage.input_tokens,
                        completion_tokens=final.usage.output_tokens,
                        total_tokens=final.usage.input_tokens + final.usage.output_tokens,
                    )
                    yield StreamChunk(finish_reason=final.stop_reason, usage=usage)

    async def count_tokens(self, messages: list[Message]) -> int:
        # Anthropic 提供 token counting API（较新模型）
        try:
            response = await self._client.messages.count_tokens(
                model=self._model,
                messages=[self._to_anthropic_msg(m) for m in messages],
            )
            return response.input_tokens
        except Exception:
            # fallback: 粗略估算
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return sum(len(enc.encode(m.content or "")) for m in messages)
```

### 4.3 Ollama `llm/ollama_llm.py`

```python
from typing import AsyncIterator
import httpx

from llm.base import BaseLLM
from llm.protocol import (
    Message, ToolDefinition, LLMResponse, StreamChunk,
    ToolCall, UsageInfo, Role,
)


class OllamaLLM(BaseLLM):
    """
    Ollama 本地模型 provider。

    Ollama 的 API 与 OpenAI 不完全兼容：
    - endpoint: POST /api/chat（非 /v1/chat/completions）
    - tools 格式略有不同
    - 无 tiktoken，用内置 token_count 端点估算

    注意：Ollama 也支持 OpenAI 兼容模式（/api/openai/v1/...），
    如果启用了该模式，直接用 OpenAICompatLLM 更简单。
    """

    def __init__(
        self,
        model: str = "qwen2.5-72b",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        base_url: str = "http://localhost:11434",
    ):
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=120.0)

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self._model

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        payload = {
            "model": self._model,
            "messages": [self._to_ollama_msg(m) for m in messages],
            "options": {
                "temperature": temperature if temperature is not None else self._temperature,
                "num_predict": max_tokens or self._max_tokens,
            },
            "stream": False,
        }
        if tools:
            payload["tools"] = [t.to_dict() for t in tools]

        resp = await self._client.post(f"{self._base_url}/api/chat", json=payload)
        data = resp.json()

        message = data.get("message", {})
        tool_calls = None
        if message.get("tool_calls"):
            tool_calls = [
                ToolCall(
                    id=tc.get("id", f"call_{i}"),
                    name=tc["function"]["name"],
                    arguments=json.loads(tc["function"]["arguments"]),
                )
                for i, tc in enumerate(message["tool_calls"])
            ]

        usage = UsageInfo(
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
        )
        usage.total_tokens = usage.prompt_tokens + usage.completion_tokens

        return LLMResponse(
            content=message.get("content"),
            tool_calls=tool_calls,
            finish_reason=data.get("done_reason"),
            usage=usage,
        )

    async def chat_stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamChunk]:
        payload = {
            "model": self._model,
            "messages": [self._to_ollama_msg(m) for m in messages],
            "options": {
                "temperature": temperature if temperature is not None else self._temperature,
                "num_predict": max_tokens or self._max_tokens,
            },
            "stream": True,
        }
        if tools:
            payload["tools"] = [t.to_dict() for t in tools]

        async with self._client.stream("POST", f"{self._base_url}/api/chat", json=payload) as resp:
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                data = json.loads(line)
                message = data.get("message", {})
                yield StreamChunk(
                    delta_content=message.get("content"),
                    finish_reason=data.get("done_reason"),
                )

    async def count_tokens(self, messages: list[Message]) -> int:
        payload = {
            "model": self._model,
            "prompt": "\n".join(m.content or "" for m in messages if m.content),
        }
        resp = await self._client.post(f"{self._base_url}/api/tokenize", json=payload)
        data = resp.json()
        return data.get("n_tokens", 0)

    async def health_check(self) -> bool:
        try:
            resp = await self._client.get(f"{self._base_url}/api/tags")
            return resp.status_code == 200
        except Exception:
            return False

    @staticmethod
    def _to_ollama_msg(msg: Message) -> dict:
        result = {"role": msg.role.value, "content": msg.content or ""}
        if msg.tool_calls:
            import json
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in msg.tool_calls
            ]
        return result

    async def __del__(self):
        await self._client.aclose()
```

---

## 5. 工厂与注册 `llm/factory.py`

```python
from typing import Type
from llm.base import BaseLLM


# ==================== Provider 注册表 ====================

_PROVIDER_REGISTRY: dict[str, Type[BaseLLM]] = {}


def register_provider(name: str, cls: Type[BaseLLM]):
    """注册 provider 类"""
    _PROVIDER_REGISTRY[name] = cls


# 自动注册（导入即注册）
def _auto_register():
    from llm.openai_llm import OpenAILLM, OpenAICompatLLM
    from llm.anthropic_llm import AnthropicLLM
    from llm.ollama_llm import OllamaLLM

    register_provider("openai", OpenAILLM)
    register_provider("openai-compat", OpenAICompatLLM)
    register_provider("anthropic", AnthropicLLM)
    register_provider("ollama", OllamaLLM)


_auto_register()


# ==================== 工厂函数 ====================

def create_llm(provider: str, config: dict) -> BaseLLM:
    """
    根据 provider name + config dict 创建 LLM 实例。

    Args:
        provider: "openai" | "anthropic" | "ollama" | "openai-compat"
        config:   ProviderConfig.asdict()，包含 api_key / model / base_url 等

    Raises:
        ValueError: 未知 provider
    """
    cls = _PROVIDER_REGISTRY.get(provider)
    if cls is None:
        available = ", ".join(_PROVIDER_REGISTRY.keys())
        raise ValueError(f"Unknown LLM provider '{provider}'. Available: {available}")

    # 每个 provider __init__ 签名不同，传全部 config 让 provider 自己取需要的字段
    import inspect
    sig = inspect.signature(cls.__init__)
    params = {k: v for k, v in config.items() if k in sig.parameters and k != "self"}
    return cls(**params)


def list_providers() -> list[str]:
    """列出已注册的 provider"""
    return list(_PROVIDER_REGISTRY.keys())
```

### 5.1 工厂设计考量

| 决策 | 理由 |
|------|------|
| `inspect.signature` 过滤参数 | 不同 provider 签名不同，自动过滤不需要的字段，新增参数无需改工厂 |
| `_auto_register()` 在模块加载时执行 | import llm.factory 即可用，无需手动注册 |
| `openai-compat` 独立注册名 | 语义上区分官方 OpenAI 和兼容服务，配置更清晰 |

---

## 6. 配置桥接器 `llm/config_bridge.py`

```python
"""
YAML Config → ProviderConfig dict 转换层。

职责：
- 读取 YAML 中 llm.providers[] 列表
- 转换为 provider + config dict，供 factory.create_llm() 使用
- 处理 api_key 的环境变量替换（${ENV_VAR}）
- 构建 FailoverLLM 链（当配置了 fallback_providers 时）

依赖：
- app.config: YAML 解析后的 dict
- llm.factory: create_llm()
- llm.failover: FailoverLLM
"""

import os
import re
import logging
from typing import Any

from llm.base import BaseLLM
from llm.factory import create_llm
from llm.failover import FailoverLLM

logger = logging.getLogger(__name__)

# 匹配 ${ENV_VAR} 格式的环境变量引用
_ENV_VAR_PATTERN = re.compile(r"\$\{(\w+)\}")


def _resolve_env_vars(value: str) -> str:
    """
    替换字符串中的 ${ENV_VAR} 为环境变量值。

    Example:
        "${OPENAI_API_KEY}" → "sk-xxx"
        "https://${API_HOST}/v1" → "https://api.example.com/v1"

    如果环境变量不存在，保留原值并打 warning。
    """
    def _replace(match: re.Match) -> str:
        env_name = match.group(1)
        env_value = os.environ.get(env_name)
        if env_value is None:
            logger.warning(f"Environment variable '{env_name}' not set, using literal")
            return match.group(0)
        return env_value

    return _ENV_VAR_PATTERN.sub(_replace, value)


def _resolve_config_values(config: dict[str, Any]) -> dict[str, Any]:
    """递归处理 config dict 中所有 string 值的 env var 替换"""
    resolved = {}
    for key, value in config.items():
        if isinstance(value, str):
            resolved[key] = _resolve_env_vars(value)
        elif isinstance(value, dict):
            resolved[key] = _resolve_config_values(value)
        else:
            resolved[key] = value
    return resolved


def create_llm_from_yaml(config_dict: dict[str, Any]) -> BaseLLM:
    """
    从 YAML 配置创建 LLM 实例（支持 fallback 链）。

    Args:
        config_dict: YAML 中 llm 部分的完整配置

            Example YAML:
                llm:
                  provider: openai          # 主 provider
                  model: gpt-4o
                  api_key: ${OPENAI_API_KEY}
                  fallback_providers:       # 可选降级链
                    - provider: ollama
                      model: qwen3
                      base_url: http://localhost:11434

    Returns:
        BaseLLM（可能是 FailoverLLM）
    """
    primary_provider = config_dict.get("provider", "openai")
    primary_model = config_dict.get("model", "gpt-4o")

    # 构建主 provider config
    primary_config = _resolve_config_values({
        k: v for k, v in config_dict.items()
        if k not in ("fallback_providers",)
    })
    primary_config["provider"] = primary_provider
    primary_config["model"] = primary_model

    logger.info(f"Creating primary LLM: {primary_provider}/{primary_model}")
    primary_llm = create_llm(primary_provider, primary_config)

    # 检查是否有 fallback providers
    fallback_configs = config_dict.get("fallback_providers", [])
    if not fallback_configs:
        return primary_llm

    # 构建 fallback chain
    fallback_llms = []
    for fb in fallback_configs:
        fb_provider = fb.get("provider", "ollama")
        fb_model = fb.get("model", "")
        fb_config = _resolve_config_values(fb)
        fb_config["provider"] = fb_provider
        if fb_model:
            fb_config["model"] = fb_model

        logger.info(f"Creating fallback LLM: {fb_provider}/{fb_model}")
        try:
            fallback_llms.append(create_llm(fb_provider, fb_config))
        except Exception as e:
            logger.warning(f"Failed to create fallback LLM {fb_provider}: {e}")

    if not fallback_llms:
        return primary_llm

    logger.info(
        f"Creating FailoverLLM: {primary_provider} → "
        f"{[llm.provider_name for llm in fallback_llms]}"
    )
    return FailoverLLM(primary_llm, fallback_llms)


def list_available_providers() -> list[str]:
    """列出所有可用的 provider（供 UI 选择）"""
    from llm.factory import list_providers
    return list_providers()
```

### 6.1 Config Bridge 设计考量

| 决策 | 理由 |
|------|------|
| `${ENV_VAR}` 替换 | API Key 等敏感信息不进 YAML，通过环境变量注入 |
| `create_llm_from_yaml()` 单一入口 | AgentCore 初始化只需调这一个函数，不关心 provider 细节 |
| fallback_providers 自动包装 FailoverLLM | 配置即降级，无需代码干预 |
| fallback 创建失败只 warning | 降级链本身是可选的，不应阻塞主流程 |

---

## 7. 故障降级器 `llm/failover.py`

```python
from typing import AsyncIterator
import asyncio
import logging

from llm.base import BaseLLM
from llm.protocol import Message, ToolDefinition, LLMResponse, StreamChunk

logger = logging.getLogger(__name__)


class FailoverLLM(BaseLLM):
    """
    故障降级包装器。

    主 provider 失败时，按 fallback_providers 顺序依次尝试。

    Usage:
        primary = OpenAILLM(...)
        fallback1 = OllamaLLM(...)
        fallback2 = AnthropicLLM(...)

        llm = FailoverLLM(primary, [fallback1, fallback2])
    """

    def __init__(self, primary: BaseLLM, fallbacks: list[BaseLLM], retry_count: int = 1):
        self._chain = [primary] + fallbacks
        self._retry_count = retry_count
        self._active_index = 0   # 当前活跃的 provider 索引

    @property
    def provider_name(self) -> str:
        active = self._chain[self._active_index]
        return f"failover({active.provider_name})"

    @property
    def model_name(self) -> str:
        return self._chain[self._active_index].model_name

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        last_error = None

        for attempt in range(self._retry_count + 1):
            for idx in range(len(self._chain)):
                llm = self._chain[idx]
                try:
                    result = await llm.chat(messages, tools, temperature, max_tokens)
                    # 成功 → 记录活跃 provider
                    if idx != self._active_index:
                        logger.info(f"Failover active: {llm.provider_name}/{llm.model_name}")
                        self._active_index = idx
                    return result
                except Exception as e:
                    last_error = e
                    logger.warning(f"Provider {llm.provider_name} failed (attempt {attempt}): {e}")

        raise last_error

    async def chat_stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamChunk]:
        # 流式降级：先非流式试探，成功后再流式
        await self.chat(messages, tools, temperature, max_tokens)
        active = self._chain[self._active_index]
        async for chunk in active.chat_stream(messages, tools, temperature, max_tokens):
            yield chunk

    async def count_tokens(self, messages: list[Message]) -> int:
        return await self._chain[self._active_index].count_tokens(messages)

    async def health_check(self) -> bool:
        for llm in self._chain:
            if await llm.health_check():
                return True
        return False
```

---

## 8. 架构总览

```
                    ┌─────────────────────┐
                    │     Agent Core      │
                    │   (零 provider       │
                    │    感知代码)         │
                    └──────────┬──────────┘
                               │ 使用内部协议
                               ▼
                    ┌─────────────────────┐
                    │    BaseLLM (ABC)     │
                    │  chat() / stream()   │
                    └──────────┬──────────┘
                               │ implements
              ┌────────────────┼────────────────┐
              ▼                ▼                 ▼
    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
    │ OpenAILLM   │  │ AnthropicLLM│  │ OllamaLLM   │
    │ (含 compat) │  │             │  │             │
    └─────────────┘  └─────────────┘  └─────────────┘

    ┌──────────────────────────────────────────┐
    │         FailoverLLM (装饰器模式)          │
    │  包装任意 BaseLLM + fallback chain       │
    └──────────────────────────────────────────┘

    ┌──────────────────────────────────────────┐
    │           Factory + Registry             │
    │  create_llm(provider, config) → BaseLLM  │
    └──────────────────────────────────────────┘

    ┌──────────────────────────────────────────┐
    │         Config Bridge (配置桥接)          │
    │  YAML → env var resolve → create_llm()   │
    │  fallback_providers → FailoverLLM        │
    └──────────────────────────────────────────┘
```

---

## 9. 新增 Provider 步骤清单

以添加 `Google Gemini` 为例：

```
1. 创建 llm/gemini_llm.py
   - class GeminiLLM(BaseLLM)
   - 实现 chat() / chat_stream() / count_tokens()
   - 实现 protocol ↔ Gemini SDK 双向转换

2. 在 factory.py 的 _auto_register() 中添加：
   register_provider("gemini", GeminiLLM)

3. 在 config/default.yaml 中添加 providers.gemini 配置段

4. 测试：
   pytest tests/llm/test_gemini_llm.py
```

**无需修改：** Agent Core、Session Manager、Tool Registry、API Server —— 任何已有代码。

---

## 10. 设计总结

| 特性 | 实现方式 |
|------|---------|
| **统一协议** | `Message` / `ToolCall` / `LLMResponse` / `StreamChunk` 内部标准对象 |
| **即插即用** | ABC + Factory + Registry，新增 provider = 1 个文件 |
| **FC 适配** | 各 provider 自行转换 ↔ OpenAI format，Agent 侧只认一种格式 |
| **流式统一** | `AsyncIterator[StreamChunk]`，SSE / Server-Sent Events / 自定义协议均可适配 |
| **故障降级** | `FailoverLLM` 装饰器模式，配置 `fallback_providers` 链 |
| **Token 估算** | 抽象 `count_tokens()`，各 provider 用最优方式实现 |
| **热更新友好** | `create_llm()` 随时重建实例，ConfigManager 回调触发切换 |

---

## 11. 结构化输出增强（JSON Mode / Function Calling）

### 11.1. 设计目标

在 LLM 抽象层原生支持结构化输出约束：
- **JSON Mode**：强制 LLM 返回合法 JSON
- **Function Calling**：通过 schema 定义约束输出结构
- **Pydantic Parser**：自动将响应解析为 Pydantic 模型

### 11.2. OpenAI Provider 增强 `llm/providers/openai_provider.py`

```python
class OpenAIProvider(LLMProvider):
    """OpenAI provider with structured output support"""

    async def generate_structured(
        self,
        messages: list[dict[str, str]],
        response_format: dict[str, Any] | None = None,
        function_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """
        生成结构化输出。

        Args:
            messages:       消息列表
            response_format: JSON Schema 格式约束（OpenAI JSON mode）
            function_schema: Function calling schema
        """
        kwargs: dict[str, Any] = {
            "model": self._config.model_name,
            "messages": messages,
            "temperature": self._config.temperature,
        }

        if response_format:
            kwargs["response_format"] = {"type": "json_schema", "json_schema": response_format}

        if function_schema:
            kwargs["tools"] = [function_schema]
            kwargs["tool_choice"] = "required"

        response = await self._client.chat.completions.create(**kwargs)

        # 处理 function calling 响应
        if function_schema and response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            content = json.loads(tool_call.function.arguments)
            return LLMResponse(
                text=json.dumps(content),
                structured_data=content,
                usage=response.usage.model_dump() if response.usage else None,
            )

        return LLMResponse(
            text=response.choices[0].message.content or "",
            usage=response.usage.model_dump() if response.usage else None,
        )
```

### 11.3. Anthropic Provider 增强 `llm/providers/anthropic_provider.py`

```python
class AnthropicProvider(LLMProvider):
    """Anthropic provider with tool use support"""

    async def generate_structured(
        self,
        messages: list[dict[str, str]],
        response_format: dict[str, Any] | None = None,
        function_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Anthropic 的结构化输出（通过 tool_use）"""
        kwargs: dict[str, Any] = {
            "model": self._config.model_name,
            "messages": messages,
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_tokens or 4096,
        }

        system_msg = next((m["content"] for m in messages if m["role"] == "system"), None)
        if system_msg:
            kwargs["system"] = system_msg

        if function_schema:
            # Anthropic 使用 tools 参数
            anthropic_tool = {
                "name": function_schema.get("function", {}).get("name", "default"),
                "description": function_schema.get("function", {}).get("description", ""),
                "input_schema": function_schema.get("function", {}).get("parameters", {}),
            }
            kwargs["tools"] = [anthropic_tool]

        response = await self._client.messages.create(**kwargs)

        # 处理 tool_use 响应
        if function_schema:
            for block in response.content:
                if block.type == "tool_use":
                    return LLMResponse(
                        text=json.dumps(block.input),
                        structured_data=block.input,
                        usage={"input_tokens": response.usage.input_tokens,
                               "output_tokens": response.usage.output_tokens},
                    )

        content = "\n".join(b.text for b in response.content if b.type == "text")
        return LLMResponse(
            text=content,
            usage={"input_tokens": response.usage.input_tokens,
                   "output_tokens": response.usage.output_tokens},
        )
```

### 11.4. LLMProvider ABC 扩展 `llm/protocol.py`

```python
class LLMProvider(Protocol):
    """LLM provider protocol with structured output"""

    async def generate(self, messages: list[dict[str, str]]) -> LLMResponse: ...

    async def generate_stream(self, messages: list[dict[str, str]]) -> AsyncIterator[StreamChunk]: ...

    async def generate_structured(
        self,
        messages: list[dict[str, str]],
        response_format: dict[str, Any] | None = None,
        function_schema: dict[str, Any] | None = None,
    ) -> LLMResponse: ...  # 新增：结构化输出

    def count_tokens(self, text: str) -> int: ...
```

---

## 12. 更新后的设计总结

| 特性 | 实现方式 |
|------|---------|
| **统一协议** | `Message` / `ToolCall` / `LLMResponse` / `StreamChunk` 内部标准对象 |
| **即插即用** | ABC + Factory + Registry，新增 provider = 1 个文件 |
| **FC 适配** | 各 provider 自行转换 ↔ OpenAI format，Agent 侧只认一种格式 |
| **流式统一** | `AsyncIterator[StreamChunk]`，SSE / Server-Sent Events / 自定义协议均可适配 |
| **故障降级** | `FailoverLLM` 装饰器模式，配置 `fallback_providers` 链 |
| **Token 估算** | 抽象 `count_tokens()`，各 provider 用最优方式实现 |
| **热更新友好** | `create_llm()` 随时重建实例，ConfigManager 回调触发切换 |
| **结构化输出** | `generate_structured()` 原生支持 JSON mode / function calling |
