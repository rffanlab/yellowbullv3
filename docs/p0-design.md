# P0 核心能力详细设计

> **定位**: 通用 AI 助手 MVP  
> **语言**: Python  
> **LLM**: 可配置（支持多提供商切换）  
> **目标**: 跑通「对话 → 意图识别 → 工具调用 → 回复」最小闭环

---

## 一、项目结构

```
yellowbullv3/
├── config/
│   ├── __init__.py
│   ├── settings.py          # 配置加载与管理
│   └── default.yaml         # 默认配置文件
├── core/
│   ├── __init__.py
│   ├── agent.py             # Agent 核心编排
│   ├── session_manager.py   # 会话管理
│   ├── tool_executor.py     # 工具执行引擎（超时、重试、并行调度）
│   └── context_builder.py   # Prompt 上下文构建
├── llm/
│   ├── __init__.py
│   ├── base.py              # LLM 抽象基类
│   ├── openai_llm.py        # OpenAI 适配
│   ├── anthropic_llm.py     # Anthropic 适配
│   └── ollama_llm.py        # Ollama 本地模型适配
├── tools/
│   ├── __init__.py
│   ├── registry.py          # 工具注册中心
│   ├── base.py              # 工具抽象基类
│   ├── builtin/
│   │   ├── current_time.py
│   │   ├── calculator.py
│   │   └── web_search.py
│   └── function_calling.py  # Function Calling 协议适配
├── models/
│   ├── __init__.py
│   ├── message.py           # 消息模型
│   ├── session.py           # 会话模型
│   └── tool_result.py       # 工具结果模型
├── api/
│   ├── __init__.py
│   └── server.py            # FastAPI HTTP + WebSocket 服务
├── main.py                  # 入口
├── pyproject.toml
└── README.md
```

---

## 二、配置系统设计

### 1. 配置文件 `config/default.yaml`

```yaml
# ========== LLM 配置（可切换提供商）==========
llm:
  provider: openai            # openai | anthropic | ollama | azure | custom
  settings:
    openai:
      api_key: "${OPENAI_API_KEY}"        # 支持环境变量引用
      model: "gpt-4o"
      temperature: 0.7
      max_tokens: 4096
      base_url: null                    # 可选，用于兼容 OpenAI 接口的第三方服务

    anthropic:
      api_key: "${ANTHROPIC_API_KEY}"
      model: "claude-sonnet-4-20250514"
      temperature: 0.7
      max_tokens: 4096

    ollama:
      base_url: "http://localhost:11434"
      model: "qwen2.5-72b"
      temperature: 0.7
      max_tokens: 4096

    azure:
      api_key: "${AZURE_API_KEY}"
      endpoint: "${AZURE_ENDPOINT}"
      deployment_name: "gpt-4o"
      temperature: 0.7
      max_tokens: 4096

# ========== Agent 配置 ==========
agent:
  system_prompt: |
    你是一个智能助手，可以帮助用户回答问题、执行任务。
    根据用户需求选择合适的工具，如果没有合适的工具则直接回答。
  context_window: 48            # 保留最近 N 条消息
  max_tool_calls_per_turn: 4    # 单轮最多并行工具数
  max_chain_depth: 5            # 最大链式调用深度
  tool_retry_limit: 3           # 工具调用失败重试次数
  total_timeout_seconds: 60     # 单次请求总超时

# ========== 工具配置 ==========
tools:
  builtin:
    current_time:
      enabled: true
    calculator:
      enabled: true
    web_search:
      enabled: true
      engine: google             # google | bing | duckduckgo
      api_key: "${SEARCH_API_KEY}"
      max_results: 5

# ========== 服务配置 ==========
server:
  host: "0.0.0.0"
  port: 8000
  cors_origins: ["*"]

# ========== 日志配置 ==========
logging:
  level: INFO                    # DEBUG | INFO | WARNING | ERROR
  file: "./logs/agent.log"
  rotation: "10 MB"
```

### 2. 配置加载 `config/settings.py`

```python
import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

@dataclass
class LLMConfig:
    provider: str = "openai"
    settings: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def active(self) -> dict[str, Any]:
        """获取当前激活的 LLM 配置"""
        return self.settings.get(self.provider, {})

@dataclass
class AgentConfig:
    system_prompt: str = ""
    context_window: int = 48
    max_tool_calls_per_turn: int = 4
    max_chain_depth: int = 5
    tool_retry_limit: int = 3
    total_timeout_seconds: int = 60

@dataclass
class ToolConfig:
    builtin: dict[str, dict[str, Any]] = field(default_factory=dict)

@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = field(default_factory=lambda: ["*"])

@dataclass
class Settings:
    llm: LLMConfig = field(default_factory=LLMConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    tools: ToolConfig = field(default_factory=ToolConfig)
    server: ServerConfig = field(default_factory=ServerConfig)

def resolve_env_vars(value: Any) -> Any:
    """解析字符串中的 ${ENV_VAR} 引用"""
    if isinstance(value, str):
        import re
        def replacer(match):
            env_key = match.group(1)
            return os.getenv(env_key, match.group(0))
        return re.sub(r"\$\{(\w+)\}", replacer, value)
    elif isinstance(value, dict):
        return {k: resolve_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [resolve_env_vars(item) for item in value]
    return value

def load_settings(config_path: str = "config/default.yaml") -> Settings:
    path = Path(config_path)
    with open(path) as f:
        raw = yaml.safe_load(f)
    raw = resolve_env_vars(raw)
    # 递归构建 Settings 对象
    ...
```

---

## 三、LLM 抽象层设计

### 1. 基类 `llm/base.py`

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from enum import Enum

class Role(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

@dataclass
class Message:
    role: Role
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None   # 工具调用请求
    tool_call_id: str | None = None                   # 工具响应关联 ID
    name: str | None = None                           # 工具名称（role=tool 时）

@dataclass
class ToolDefinition:
    """发送给 LLM 的工具定义（Function Calling 格式）"""
    name: str
    description: str
    parameters: dict[str, Any]   # JSON Schema

@dataclass
class LLMResponse:
    content: str | None                          # 文本回复
    tool_calls: list[dict[str, Any]] | None = None  # 工具调用列表
    finish_reason: str | None = None             # stop | tool_calls | length
    usage: dict[str, int] | None = None          # {prompt_tokens, completion_tokens}

@dataclass
class LLMConfig:
    model: str
    temperature: float = 0.7
    max_tokens: int = 4096
    extra: dict[str, Any] = field(default_factory=dict)

class BaseLLM(ABC):
    """所有 LLM provider 必须实现的接口"""

    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        stream: bool = False,
    ) -> LLMResponse:
        """非流式对话"""
        ...

    @abstractmethod
    async def chat_stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
    ) -> AsyncIterator[str]:
        """流式输出（SSE）"""
        ...

    @abstractmethod
    async def count_tokens(self, messages: list[Message]) -> int:
        """Token 估算"""
        ...
```

### 2. OpenAI 适配 `llm/openai_llm.py`

```python
from .base import BaseLLM, Message, LLMConfig, LLMResponse, ToolDefinition, Role
from openai import AsyncOpenAI
from typing import AsyncIterator

class OpenAILLM(BaseLLM):
    def __init__(self, config: LLMConfig, api_key: str, base_url: str | None = None):
        self.config = config
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    def _to_openai_message(self, msg: Message) -> dict:
        if msg.role == Role.TOOL:
            return {
                "role": "tool",
                "tool_call_id": msg.tool_call_id,
                "content": msg.content or "",
            }
        result = {
            "role": msg.role.value,
            "content": msg.content,
        }
        if msg.tool_calls:
            result["tool_calls"] = msg.tool_calls
        return result

    def _to_openai_tools(self, tools: list[ToolDefinition]) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in tools
        ]

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        stream: bool = False,
    ) -> LLMResponse:
        kwargs = {
            "model": self.config.model,
            "messages": [self._to_openai_message(m) for m in messages],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if tools:
            kwargs["tools"] = self._to_openai_tools(tools)
            kwargs["tool_choice"] = "auto"

        resp = await self.client.chat.completions.create(**kwargs)
        choice = resp.choices[0]

        return LLMResponse(
            content=choice.message.content,
            tool_calls=choice.message.tool_calls,
            finish_reason=choice.finish_reason,
            usage={
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
            } if resp.usage else None,
        )

    async def chat_stream(self, messages: list[Message], tools: list[ToolDefinition] | None = None):
        ...  # SSE 流式实现

    async def count_tokens(self, messages: list[Message]) -> int:
        import tiktoken
        enc = tiktoken.encoding_for_model(self.config.model)
        total = 0
        for msg in messages:
            total += len(enc.encode(f"{msg.role.value}: {msg.content or ''}"))
        return total
```

### 3. LLM 工厂 `llm/factory.py`

```python
from typing import Type
from .base import BaseLLM, LLMConfig

PROVIDER_MAP: dict[str, Type[BaseLLM]] = {
    "openai": "OpenAILLM",        # → llm/openai_llm.py
    "anthropic": "AnthropicLLM",  # → llm/anthropic_llm.py
    "ollama": "OllamaLLM",        # → llm/ollama_llm.py
}

def create_llm(provider: str, config: dict) -> BaseLLM:
    """根据配置创建 LLM 实例"""
    cls_name = PROVIDER_MAP[provider]
    module = __import__(f"llm.{provider}_llm", fromlist=[cls_name])
    cls = getattr(module, cls_name)
    return cls(**config)
```

**设计要点：**
- 新增 LLM provider 只需实现 `BaseLLM` 接口 + 注册到工厂
- 所有 provider 统一转换为内部 `Message` / `ToolDefinition` 协议
- 兼容 OpenAI 接口的第三方服务（vLLM、Ollama、本地部署）直接复用 `OpenAILLM`，配置 `base_url` 即可

---

## 四、工具系统设计

### 1. 工具基类 `tools/base.py`

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

@dataclass
class ToolInfo:
    """工具元数据（用于注册和 LLM function calling）"""
    name: str
    description: str
    parameters: dict[str, Any]   # JSON Schema format

@dataclass
class ToolResult:
    content: str
    success: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

class BaseTool(ABC):
    @property
    @abstractmethod
    def info(self) -> ToolInfo:
        """工具描述，用于生成 function calling schema"""
        ...

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """执行业务逻辑"""
        ...

    def validate(self, params: dict[str, Any]) -> list[str]:
        """参数校验，返回错误列表（空表示通过）"""
        return []
```

### 2. 工具注册中心 `tools/registry.py`

```python
from .base import BaseTool, ToolInfo

class ToolRegistry:
    _tools: dict[str, BaseTool] = {}

    @classmethod
    def register(cls, tool: BaseTool):
        cls._tools[tool.info.name] = tool

    @classmethod
    def get(cls, name: str) -> BaseTool | None:
        return cls._tools.get(name)

    @classmethod
    def list_enabled(cls) -> list[BaseTool]:
        """返回所有启用的工具"""
        return list(cls._tools.values())

    @classmethod
    def to_function_definitions(cls) -> list[dict[str, Any]]:
        """转换为 LLM function calling 格式"""
        return [
            {
                "name": t.info.name,
                "description": t.info.description,
                "parameters": t.info.parameters,
            }
            for t in cls._tools.values()
        ]

# 装饰器注册方式
def register_tool(name: str, description: str, parameters: dict[str, Any]):
    def decorator(cls: type[BaseTool]):
        instance = cls()
        instance._info = ToolInfo(name=name, description=description, parameters=parameters)
        ToolRegistry.register(instance)
        return cls
    return decorator
```

### 3. 内置工具实现

#### `current_time.py` — 时间查询

```python
from tools.base import BaseTool, ToolInfo, ToolResult
from tools.registry import register_tool
from datetime import datetime, timezone, timedelta

@register_tool(
    name="current_time",
    description="获取当前日期和时间。支持指定时区。",
    parameters={
        "type": "object",
        "properties": {
            "timezone": {
                "type": "string",
                "description": "时区，如 'Asia/Shanghai'、'UTC'。默认为 UTC。",
            }
        },
        "required": [],
    }
)
class CurrentTimeTool(BaseTool):
    @property
    def info(self) -> ToolInfo:
        return self._info

    async def execute(self, timezone: str = "UTC") -> ToolResult:
        try:
            tz = timezone(timezone) if timezone != "UTC" else timezone.utc
            now = datetime.now(tz)
            return ToolResult(content=now.strftime("%Y-%m-%d %H:%M:%S %Z"))
        except Exception as e:
            return ToolResult(content=f"时区获取失败: {e}", success=False)
```

#### `calculator.py` — 数学计算

```python
from tools.base import BaseTool, ToolInfo, ToolResult
from tools.registry import register_tool
import math

@register_tool(
    name="calculator",
    description="执行数学计算。支持四则运算、幂运算、三角函数等。输入标准数学表达式。",
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "数学表达式，如 '2 + 3 * 4'、'sin(3.14)'、'sqrt(16)'",
            }
        },
        "required": ["expression"],
    }
)
class CalculatorTool(BaseTool):
    @property
    def info(self) -> ToolInfo:
        return self._info

    async def execute(self, expression: str) -> ToolResult:
        try:
            # 安全计算：仅允许数学运算
            allowed = {"abs", "round", "min", "max", "sum",
                       "pow", "sqrt", "sin", "cos", "tan", "log", "pi", "e"}
            safe_dict = {k: getattr(__builtins__, k, None) or getattr(math, k, None)
                         for k in allowed}
            result = eval(expression, {"__builtins__": {}}, safe_dict)
            return ToolResult(content=str(result))
        except Exception as e:
            return ToolResult(content=f"计算错误: {e}", success=False)
```

#### `web_search.py` — 网络搜索

```python
from tools.base import BaseTool, ToolInfo, ToolResult
from tools.registry import register_tool
from typing import Any

@register_tool(
    name="web_search",
    description="搜索互联网获取最新信息。适用于新闻、事实查询等需要实时信息的场景。",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索关键词",
            },
            "max_results": {
                "type": "integer",
                "description": "返回结果数量，默认 5",
            }
        },
        "required": ["query"],
    }
)
class WebSearchTool(BaseTool):
    def __init__(self, engine: str = "duckduckgo", api_key: str | None = None):
        self.engine = engine
        self.api_key = api_key

    @property
    def info(self) -> ToolInfo:
        return self._info

    async def execute(self, query: str, max_results: int = 5) -> ToolResult:
        try:
            if self.engine == "duckduckgo":
                from duckduckgo_search import DDGS
                results = DDGS().text(query, max_results=max_results)
            elif self.engine == "google":
                from googlesearch import search as gsearch
                results = [{"title": r.title, "href": r.url, "body": r.description}
                           for r in gsearch(query, num_results=max_results)]
            else:
                return ToolResult(content=f"不支持的搜索引擎: {self.engine}", success=False)

            formatted = "\n\n".join(
                f"[{r['title']}]({r.get('href', '')})\n{r.get('body', r.get('description', ''))}"
                for r in results[:max_results]
            )
            return ToolResult(content=formatted or "未找到相关结果")
        except Exception as e:
            return ToolResult(content=f"搜索失败: {e}", success=False)
```

---

## 五、会话与上下文管理

### 1. 消息模型 `models/message.py`

```python
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

@dataclass
class Message:
    id: str
    role: MessageRole
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None   # LLM 发出的工具调用
    tool_call_id: str | None = None                   # 工具返回时的关联 ID
    created_at: datetime = field(default_factory=datetime.now)
```

### 2. 会话模型 `models/session.py`

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
import uuid

@dataclass
class SessionState:
    """多步任务状态跟踪"""
    step: int = 0                        # 当前步骤
    chain_depth: int = 0                 # 当前链式调用深度
    tool_retry_counts: dict[str, int] = field(default_factory=dict)  # 工具重试计数

@dataclass
class Session:
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    messages: list[Message] = field(default_factory=list)
    state: SessionState = field(default_factory=SessionState)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def add_message(self, msg: Message):
        self.messages.append(msg)
        self.updated_at = datetime.now()

    def get_context_messages(self, window_size: int) -> list[Message]:
        """滑动窗口：保留 system + 最近 N 条"""
        non_system = [m for m in self.messages if m.role != MessageRole.SYSTEM]
        return [m for m in self.messages if m.role == MessageRole.SYSTEM] + non_system[-window_size:]
```

### 3. 会话管理器 `core/session_manager.py`

```python
from typing import dict, Optional
from models.session import Session

class SessionManager:
    def __init__(self):
        self._sessions: dict[str, Session] = {}   # MVP 用内存，后续可换 Redis

    def create(self, user_id: str) -> Session:
        session = Session(user_id=user_id)
        self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)

    def delete(self, session_id: str) -> bool:
        return self._sessions.pop(session_id, None) is not None

    def cleanup_expired(self, max_age_seconds: int = 3600 * 24):
        """清理过期会话"""
        import datetime
        now = datetime.datetime.now()
        expired = [
            sid for sid, s in self._sessions.items()
            if (now - s.updated_at).total_seconds() > max_age_seconds
        ]
        for sid in expired:
            del self._sessions[sid]
```

### 4. 上下文构建器 `core/context_builder.py`

```python
from llm.base import Message as LLMMessage, Role
from models.session import Session
from models.message import MessageRole

class ContextBuilder:
    """将内部会话消息转换为 LLM 请求格式"""

    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt

    def build(self, session: Session, window_size: int = 48) -> list[LLMMessage]:
        context_msgs = session.get_context_messages(window_size)
        messages = [LLMMessage(role=Role.SYSTEM, content=self.system_prompt)]

        for msg in context_msgs:
            if msg.role == MessageRole.USER:
                messages.append(LLMMessage(role=Role.USER, content=msg.content))
            elif msg.role == MessageRole.ASSISTANT:
                messages.append(LLMMessage(
                    role=Role.ASSISTANT,
                    content=msg.content,
                    tool_calls=msg.tool_calls,
                ))
            elif msg.role == MessageRole.TOOL:
                messages.append(LLMMessage(
                    role=Role.TOOL,
                    content=msg.content,
                    tool_call_id=msg.tool_call_id,
                ))

        return messages
```

---

## 六、Agent 核心编排

### `core/agent.py` — 主循环

```python
import asyncio
import uuid
from typing import AsyncIterator
from dataclasses import dataclass, field

from config.settings import Settings
from llm.base import BaseLLM, Message as LLMMessage, ToolDefinition
from core.session_manager import SessionManager
from core.context_builder import ContextBuilder
from tools.registry import ToolRegistry
from models.message import Message, MessageRole
from models.session import Session

@dataclass
class ChatRequest:
    session_id: str | None   # None 表示新建会话
    user_id: str
    message: str

@dataclass
class ChatResponse:
    content: str
    session_id: str
    tool_results: list[dict] = field(default_factory=list)
    needs_clarification: bool = False
    usage: dict | None = None

class Agent:
    def __init__(self, settings: Settings, llm: BaseLLM):
        self.settings = settings
        self.llm = llm
        self.session_manager = SessionManager()
        self.context_builder = ContextBuilder(settings.agent.system_prompt)

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """主入口：处理用户请求"""
        # 1. 获取或创建会话
        session = self._get_or_create_session(request)

        # 2. 添加用户消息
        user_msg = Message(
            id=str(uuid.uuid4()),
            role=MessageRole.USER,
            content=request.message,
        )
        session.add_message(user_msg)

        # 3. 主循环：LLM → 工具执行 → 直到返回最终回复
        max_depth = self.settings.agent.max_chain_depth
        for _ in range(max_depth):
            session.state.chain_depth += 1

            # 构建上下文 → LLM 推理
            context = self.context_builder.build(
                session, self.settings.agent.context_window
            )
            tools = ToolRegistry.to_function_definitions()
            response = await self.llm.chat(messages=context, tools=tools)

            # 不需要工具调用 → 直接返回
            if not response.tool_calls:
                assistant_msg = Message(
                    id=str(uuid.uuid4()),
                    role=MessageRole.ASSISTANT,
                    content=response.content,
                )
                session.add_message(assistant_msg)
                return ChatResponse(
                    content=response.content or "",
                    session_id=session.session_id,
                    usage=response.usage,
                )

            # 需要工具调用 → 执行并追加结果
            tool_calls = response.tool_calls
            assistant_msg = Message(
                id=str(uuid.uuid4()),
                role=MessageRole.ASSISTANT,
                content=response.content,
                tool_calls=tool_calls,
            )
            session.add_message(assistant_msg)

            # 并行执行所有工具调用
            tool_results = await self._execute_tools(tool_calls, session)

            # 追加工具结果到上下文，继续下一轮 LLM 推理
            for tr in tool_results:
                tool_msg = Message(
                    id=str(uuid.uuid4()),
                    role=MessageRole.TOOL,
                    content=tr["content"],
                    tool_call_id=tr["tool_call_id"],
                )
                session.add_message(tool_msg)

        # 超过最大深度，返回兜底回复
        return ChatResponse(
            content="抱歉，任务处理步骤过多，请简化您的请求。",
            session_id=session.session_id,
        )

    async def _execute_tools(self, tool_calls: list[dict], session: Session) -> list[dict]:
        """并行执行工具调用"""
        tasks = []
        for tc in tool_calls[:self.settings.agent.max_tool_calls_per_turn]:
            tasks.append(self._run_single_tool(tc))
        return await asyncio.gather(*tasks)

    async def _run_single_tool(self, tool_call: dict) -> dict:
        """执行单个工具，含重试逻辑"""
        func_name = tool_call["function"]["name"]
        func_args = json.loads(tool_call["function"]["arguments"])
        tool = ToolRegistry.get(func_name)

        if not tool:
            return {
                "tool_call_id": tool_call["id"],
                "content": f"未知工具: {func_name}",
            }

        # 重试机制
        for attempt in range(self.settings.agent.tool_retry_limit):
            result = await tool.execute(**func_args)
            if result.success:
                return {
                    "tool_call_id": tool_call["id"],
                    "content": result.content,
                }
            # 记录重试，最后一次返回错误内容
        return {
            "tool_call_id": tool_call["id"],
            "content": f"工具 '{func_name}' 执行失败: {result.content}",
        }

    def _get_or_create_session(self, request: ChatRequest) -> Session:
        if request.session_id:
            session = self.session_manager.get(request.session_id)
            if session:
                return session
        return self.session_manager.create(request.user_id)
```

**核心循环逻辑：**
```
用户消息入会话
    ↓
[循环，最多 max_chain_depth 次]
    ├─ 构建上下文（system + 滑动窗口历史）
    ├─ LLM 推理（携带工具定义）
    ├─ 无 tool_calls？ → 返回最终回复 ✓
    └─ 有 tool_calls？
        ├─ 记录 assistant 消息（含 tool_calls）
        ├─ 并行执行所有工具
        ├─ 将工具结果写入会话（role=tool）
        └─ 回到循环开头，LLM 根据工具结果继续推理
    ↓
超过最大深度 → 兜底回复
```

---

## 七、API 服务层

### `api/server.py` — FastAPI

```python
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="YellowBull Agent API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
)

# agent 实例通过依赖注入或全局初始化
agent: Agent | None = None

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    user_id: str
    message: str

class ChatResponse(BaseModel):
    content: str
    session_id: str
    tool_results: list[dict] = []
    needs_clarification: bool = False
    usage: Optional[dict] = None

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not agent:
        raise HTTPException(500, "Agent not initialized")
    result = await agent.chat(ChatRequest(**request.model_dump()))
    return ChatResponse(**result.__dict__)

@app.post("/api/sessions/{session_id}", status_code=204)
async def delete_session(session_id: str):
    if agent:
        agent.session_manager.delete(session_id)

@app.get("/api/sessions/{session_id}/history")
async def get_history(session_id: str):
    if not agent:
        raise HTTPException(500, "Agent not initialized")
    session = agent.session_manager.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    return {
        "session_id": session.session_id,
        "messages": [
            {"role": m.role.value, "content": m.content, "created_at": str(m.created_at)}
            for m in session.messages
        ],
    }
```

---

## 八、入口 `main.py`

```python
import asyncio
from config.settings import load_settings
from llm.factory import create_llm
from core.agent import Agent
import uvicorn

def main():
    settings = load_settings()
    llm_config = settings.llm.active
    llm = create_llm(settings.llm.provider, llm_config)
    global agent
    agent = Agent(settings, llm)
    uvicorn.run("api.server:app", host=settings.server.host, port=settings.server.port)

if __name__ == "__main__":
    main()
```

---

## 九、依赖 `pyproject.toml`（核心部分）

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

    # LLM providers（按需安装）
    "openai>=1.58",
    "anthropic>=0.42",

    # 工具依赖
    "duckduckgo-search>=7.2",
    "google>=3.0",           # googlesearch-python

    # Token 计算
    "tiktoken>=0.9",
]
```

---

## 十、请求完整时序图

```
Client                API Server              Agent               LLM               Tool
 │                       │                      │                  │                 │
 │── POST /api/chat ──→│                      │                  │                 │
 │                       ├── chat(request) →──│                  │                 │
 │                       │                      ├── build context │                 │
 │                       │                      │    (session +    │                 │
 │                       │                      │     window)      │                 │
 │                       │                      │                  │                 │
 │                       │                      ├─ chat(msgs, ───→│                 │
 │                       │                      │   tools)         │                 │
 │                       │                      │                  │                 │
 │                       │                      │←─ LLMResponse ──│                 │
 │                       │                      │  (has tool_calls)│                 │
 │                       │                      │                  │                 │
 │                       │                      ├─ execute ───────────────────────→│
 │                       │                      │   tools()         │                 │
 │                       │                      │                  │←────────────────│
 │                       │                      │← ToolResult ─────│                 │
 │                       │                      │                  │                 │
 │                       │                      │ (loop: chat      │                 │
 │                       │                      │  again with       │                 │
 │                       │                      │  tool results)    │                 │
 │                       │                      │                  │                 │
 │                       │                      ├─ chat(msgs) ───→│                 │
 │                       │                      │                  │                 │
 │                       │                      │← LLMResponse ───│                 │
 │                       │                      │  (no tool_calls)  │                 │
 │                       │                      │                  │                 │
 │                       │← ChatResponse ←──────│                  │                 │
 │← JSON response ──────│                      │                  │                 │
```

---

## 十一、MVP 验收标准

| # | 验收项 | 说明 |
|---|--------|------|
| 1 | 启动服务 | `python main.py` 后，`/api/chat` 可访问 |
| 2 | 直接问答 | 发送普通问题，返回 LLM 回复（不调用工具） |
| 3 | 工具调用 | 问"现在几点"，自动调用 `current_time` 并返回结果 |
| 4 | 多轮对话 | 连续对话能记住上下文 |
| 5 | 链式调用 | 复杂请求触发多次 LLM → tool → LLM 循环 |
| 6 | 切换模型 | 修改 `default.yaml` 中 `provider`，重启即可切换 LLM |
| 7 | 错误处理 | 工具执行失败能重试并返回友好提示 |

---

## 十二、后续迭代路线（P1+）

```
P0 (当前)          P1                    P2                  P3
─────────        ──────────            ──────────          ──────────
意图识别         知识库 RAG           多模态              多 Agent 编排
工具调用         持久化存储           工作流编排          插件市场
对话管理         日志监控             权限控制            成本优化
可配置 LLM       记忆系统             语音交互            A/B Test
```
