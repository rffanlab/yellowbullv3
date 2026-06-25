# 插件系统详细设计（Plugin System）

## 1. 职责边界

| 领域 | 说明 |
|------|------|
| **工具插件** | 可插拔的外部工具（搜索、API 调用等） |
| **模型插件** | 多 LLM 提供商适配（OpenAI、Anthropic、本地模型） |
| **存储插件** | 向量数据库、缓存后端切换 |
| **生命周期** | 发现 → 加载 → 初始化 → 使用 → 卸载 |

---

## 2. 工具插件 `plugins/tool_plugin.py`

```python
"""
可插拔工具系统。

每个工具实现 BaseTool 接口，通过 ToolRegistry 注册和管理。
"""

import asyncio
import importlib
import inspect
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """工具执行结果"""
    content: str
    success: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class ToolParameter:
    """工具参数定义"""
    name: str
    type: str              # "string" | "number" | "boolean" | "array" | "object"
    description: str = ""
    required: bool = False
    default: Any = None


@dataclass
class ToolManifest:
    """工具清单（声明式描述）"""
    name: str
    version: str = "1.0.0"
    description: str = ""
    parameters: list[ToolParameter] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


class BaseTool(ABC):
    """工具基类"""

    manifest: ToolManifest | None = None

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """执行工具逻辑"""
        ...

    def validate_params(self, **kwargs) -> list[str]:
        """验证参数，返回错误列表"""
        errors = []
        if not self.manifest:
            return errors

        for param in self.manifest.parameters:
            value = kwargs.get(param.name)
            if param.required and value is None:
                errors.append(f"Missing required parameter: {param.name}")
            elif value is not None and param.type == "number":
                try:
                    float(value)
                except (ValueError, TypeError):
                    errors.append(f"Parameter '{param.name}' must be a number")

        return errors


class ToolRegistry:
    """工具注册表"""

    def __init__(self):
        self._tools: dict[str, BaseTool] = {}
        self._manifests: dict[str, ToolManifest] = {}

    def register(self, tool: BaseTool):
        """注册工具"""
        name = tool.manifest.name if tool.manifest else tool.__class__.__name__.lower()
        self._tools[name] = tool
        if tool.manifest:
            self._manifests[name] = tool.manifest

    def unregister(self, name: str):
        """注销工具"""
        self._tools.pop(name, None)
        self._manifests.pop(name, None)

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def list_tools(self) -> dict[str, ToolManifest]:
        """列出所有已注册工具的清单"""
        return {
            name: tool.manifest or ToolManifest(name=name)
            for name, tool in self._tools.items()
        }

    async def execute(self, tool_name: str, **kwargs) -> ToolResult:
        """执行指定工具"""
        tool = self._tools.get(tool_name)
        if not tool:
            return ToolResult(content="", success=False, error=f"Tool '{tool_name}' not found")

        errors = tool.validate_params(**kwargs)
        if errors:
            return ToolResult(
                content="", success=False,
                error=f"Validation failed: {'; '.join(errors)}",
            )

        try:
            result = await tool.execute(**kwargs)
            logger.debug(f"Tool '{tool_name}' executed successfully")
            return result
        except Exception as e:
            logger.error(f"Tool '{tool_name}' execution failed: {e}")
            return ToolResult(content="", success=False, error=str(e))

    def to_openai_tools(self) -> list[dict[str, Any]]:
        """转换为 OpenAI function calling 格式"""
        tools = []
        for name, manifest in self._manifests.items():
            tool_def = {
                "type": "function",
                "function": {
                    "name": manifest.name,
                    "description": manifest.description,
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                },
            }
            for param in manifest.parameters:
                tool_def["function"]["parameters"]["properties"][param.name] = {
                    "type": param.type,
                    "description": param.description,
                }
                if param.default is not None:
                    tool_def["function"]["parameters"]["properties"][param.name]["default"] = param.default
                if param.required:
                    tool_def["function"]["parameters"]["required"].append(param.name)

            tools.append(tool_def)
        return tools


# 全局注册表
default_registry = ToolRegistry()
```

---

## 3. 内置工具实现 `plugins/builtin_tools.py`

```python
"""
内置工具集合。
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from plugins.tool_plugin import BaseTool, ToolManifest, ToolParameter, ToolResult

logger = logging.getLogger(__name__)


class WebSearchTool(BaseTool):
    """网页搜索工具"""

    manifest = ToolManifest(
        name="web_search",
        version="1.0.0",
        description="Search the web for information",
        parameters=[
            ToolParameter("query", "string", "Search query", required=True),
            ToolParameter("max_results", "number", "Maximum number of results", default=5),
        ],
    )

    async def execute(self, query: str, max_results: int = 5, **kwargs) -> ToolResult:
        # TODO: 接入真实搜索引擎 API（SerpAPI、Tavily 等）
        logger.info(f"Web search: '{query}' (max={max_results})")
        results = [
            f"[{i}] Example result for '{query}'",
            f"[{i+1}] Another result about '{query}'",
        ][:max_results]
        return ToolResult(
            content="\n".join(results),
            metadata={"query": query, "result_count": len(results)},
        )


class FileReadTool(BaseTool):
    """文件读取工具"""

    manifest = ToolManifest(
        name="file_read",
        version="1.0.0",
        description="Read content from a file",
        parameters=[
            ToolParameter("path", "string", "File path to read", required=True),
            ToolParameter("encoding", "string", "File encoding", default="utf-8"),
        ],
    )

    async def execute(self, path: str, encoding: str = "utf-8", **kwargs) -> ToolResult:
        file_path = Path(path)
        if not file_path.exists():
            return ToolResult(content="", success=False, error=f"File not found: {path}")
        try:
            content = file_path.read_text(encoding=encoding)
            return ToolResult(
                content=content[:10000],  # 限制返回大小
                metadata={"file_size": file_path.stat().st_size, "truncated": len(content) > 10000},
            )
        except Exception as e:
            return ToolResult(content="", success=False, error=str(e))


class DateTimeTool(BaseTool):
    """日期时间工具"""

    manifest = ToolManifest(
        name="datetime",
        version="1.0.0",
        description="Get current date and time or format a timestamp",
        parameters=[
            ToolParameter("format", "string", "Output format (default: ISO)", default="iso"),
        ],
    )

    async def execute(self, format: str = "iso", **kwargs) -> ToolResult:
        now = datetime.now(timezone.utc)
        if format == "iso":
            result = now.isoformat()
        elif format == "unix":
            result = str(now.timestamp())
        else:
            result = now.strftime(format)

        return ToolResult(content=result, metadata={"timezone": "UTC"})


class CalculatorTool(BaseTool):
    """计算器工具"""

    manifest = ToolManifest(
        name="calculator",
        version="1.0.0",
        description="Evaluate a mathematical expression",
        parameters=[
            ToolParameter("expression", "string", "Math expression to evaluate", required=True),
        ],
    )

    async def execute(self, expression: str, **kwargs) -> ToolResult:
        # 安全计算：只允许数字和运算符
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            return ToolResult(
                content="", success=False,
                error="Expression contains invalid characters",
            )

        try:
            result = eval(expression, {"__builtins__": {}}, {})  # noqa: S307
            return ToolResult(content=str(result))
        except Exception as e:
            return ToolResult(content="", success=False, error=str(e))


def register_builtin_tools(registry=None):
    """注册所有内置工具"""
    target = registry or default_registry
    for tool_class in [WebSearchTool, FileReadTool, DateTimeTool, CalculatorTool]:
        target.register(tool_class())
```

---

## 4. LLM 模型插件 `plugins/llm_plugin.py`

```python
"""
多 LLM 提供商适配。

每个提供商实现 BaseLLMProvider 接口，通过统一 API 调用。
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    """聊天消息"""
    role: str         # "system" | "user" | "assistant" | "tool"
    content: str
    name: str | None = None       # tool 调用时的工具名
    tool_call_id: str | None = None


@dataclass
class ChatResponse:
    """聊天响应"""
    content: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)   # prompt_tokens, completion_tokens
    finish_reason: str = "stop"
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class LLMConfig:
    """LLM 配置"""
    model_name: str
    api_key: str | None = None
    base_url: str | None = None
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 1.0


class BaseLLMProvider(ABC):
    """LLM 提供商基类"""

    @abstractmethod
    async def chat(
        self,
        messages: list[ChatMessage],
        system_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> ChatResponse:
        ...

    @abstractmethod
    async def count_tokens(self, text: str) -> int:
        """估算 token 数量"""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API 提供商"""

    def __init__(self, config: LLMConfig):
        self._config = config
        # TODO: from openai import AsyncOpenAI
        # self._client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)

    @property
    def model_name(self) -> str:
        return self._config.model_name

    async def chat(
        self, messages, system_prompt=None, tools=None,
        max_tokens=None, temperature=None,
    ) -> ChatResponse:
        logger.info(f"OpenAI call: model={self.model_name}, msgs={len(messages)}")
        # TODO: 真实 API 调用
        return ChatResponse(
            content="Simulated OpenAI response",
            model=self.model_name,
            usage={"prompt_tokens": 100, "completion_tokens": 50},
        )

    async def count_tokens(self, text: str) -> int:
        # TODO: tiktoken.encoding_for_model(self.model_name).encode(text)
        return len(text) // 4


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude API 提供商"""

    def __init__(self, config: LLMConfig):
        self._config = config

    @property
    def model_name(self) -> str:
        return self._config.model_name

    async def chat(
        self, messages, system_prompt=None, tools=None,
        max_tokens=None, temperature=None,
    ) -> ChatResponse:
        logger.info(f"Anthropic call: model={self.model_name}, msgs={len(messages)}")
        return ChatResponse(
            content="Simulated Claude response",
            model=self.model_name,
            usage={"prompt_tokens": 100, "completion_tokens": 50},
        )

    async def count_tokens(self, text: str) -> int:
        return len(text) // 4


class LocalModelProvider(BaseLLMProvider):
    """本地模型提供商（Ollama / vLLM）"""

    def __init__(self, config: LLMConfig):
        self._config = config

    @property
    def model_name(self) -> str:
        return self._config.model_name

    async def chat(
        self, messages, system_prompt=None, tools=None,
        max_tokens=None, temperature=None,
    ) -> ChatResponse:
        logger.info(f"Local model call: {self.model_name}")
        return ChatResponse(
            content="Simulated local model response",
            model=self.model_name,
        )

    async def count_tokens(self, text: str) -> int:
        return len(text) // 4


class LLMProviderRegistry:
    """LLM 提供商注册表"""

    def __init__(self):
        self._providers: dict[str, BaseLLMProvider] = {}

    def register(self, name: str, provider: BaseLLMProvider):
        self._providers[name] = provider

    def get(self, name: str) -> BaseLLMProvider | None:
        return self._providers.get(name)

    def list_providers(self) -> dict[str, str]:
        return {name: p.model_name for name, p in self._providers.items()}


# 全局注册表
llm_registry = LLMProviderRegistry()
```

---

## 5. 存储插件 `plugins/storage_plugin.py`

```python
"""
可插拔存储后端。

支持：SQLite、PostgreSQL、Redis（缓存）、ChromaDB（向量）
"""

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class BaseKeyValueStore(Protocol):
    """键值存储协议"""

    async def get(self, key: str) -> Any: ...
    async def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> bool: ...
    async def delete(self, key: str) -> bool: ...
    async def exists(self, key: str) -> bool: ...


class BaseVectorStore(Protocol):
    """向量存储协议"""

    async def add_documents(
        self, collection: str, documents: list[dict[str, Any]]
    ) -> int: ...
    async def search(
        self, collection: str, query_vector: list[float], top_k: int = 5
    ) -> list[dict[str, Any]]: ...


class JSONFileStore(BaseKeyValueStore):
    """JSON 文件存储（开发/测试用）"""

    def __init__(self, data_dir: str = "./data/kv"):
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)

    async def get(self, key: str) -> Any:
        file_path = self._data_dir / f"{key}.json"
        if not file_path.exists():
            return None
        data = json.loads(file_path.read_text())
        return data.get("value")

    async def set(self, key: str, value: Any, ttl_seconds=None) -> bool:
        file_path = self._data_dir / f"{key}.json"
        file_path.write_text(json.dumps({"value": value}, ensure_ascii=False))
        return True

    async def delete(self, key: str) -> bool:
        file_path = self._data_dir / f"{key}.json"
        if file_path.exists():
            file_path.unlink()
            return True
        return False

    async def exists(self, key: str) -> bool:
        return (self._data_dir / f"{key}.json").exists()


class InMemoryStore(BaseKeyValueStore):
    """内存存储（高性能，不持久化）"""

    def __init__(self):
        self._store: dict[str, Any] = {}

    async def get(self, key: str) -> Any:
        return self._store.get(key)

    async def set(self, key: str, value: Any, ttl_seconds=None) -> bool:
        self._store[key] = value
        return True

    async def delete(self, key: str) -> bool:
        return self._store.pop(key, None) is not None

    async def exists(self, key: str) -> bool:
        return key in self._store


class StorageRegistry:
    """存储后端注册表"""

    def __init__(self):
        self._kv_stores: dict[str, BaseKeyValueStore] = {}
        self._vector_stores: dict[str, BaseVectorStore] = {}

    def register_kv(self, name: str, store: BaseKeyValueStore):
        self._kv_stores[name] = store

    def register_vector(self, name: str, store: BaseVectorStore):
        self._vector_stores[name] = store

    def get_kv(self, name: str) -> BaseKeyValueStore | None:
        return self._kv_stores.get(name)

    def get_vector(self, name: str) -> BaseVectorStore | None:
        return self._vector_stores.get(name)


# 全局注册表
storage_registry = StorageRegistry()
```

---

## 6. YAML 配置 `config/plugins.yaml`

```yaml
plugins:
  tools:
    builtin: true              # 是否加载内置工具
    custom_dirs:               # 自定义插件目录
      - "./plugins/custom"

  llm_providers:
    openai:
      enabled: true
      model_name: "gpt-4o-mini"
      api_key: "${OPENAI_API_KEY}"
      max_tokens: 4096
      temperature: 0.7

    anthropic:
      enabled: false
      model_name: "claude-3-haiku"
      api_key: "${ANTHROPIC_API_KEY}"

    local:
      enabled: false
      model_name: "llama3:8b"
      base_url: "http://localhost:11434"

  storage:
    kv_store:
      type: "json_file"        # json_file | memory | redis
      data_dir: "./data/kv"

    vector_store:
      type: "chroma"           # chroma | faiss | qdrant
      persist_directory: "./data/vectors"
```

---

## 7. 架构总览

```
                    ┌─────────────────────┐
                    │   Plugin Registry    │
                    │                     │
                    │  ToolRegistry        │──→ BaseTool implementations
                    │  LLMProviderRegistry │──→ Provider implementations
                    │  StorageRegistry     │──→ Store implementations
                    └──────────┬──────────┘
                               │
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                     ▼
   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
   │  Tool Plugins │    │ LLM Plugins  │    │ Storage      │
   │              │    │              │    │ Plugins       │
   │ • web_search │    │ • OpenAI     │    │ • JSONFile    │
   │ • file_read  │    │ • Anthropic  │    │ • InMemory    │
   │ • datetime   │    │ • LocalModel │    │ • Redis       │
   │ • calculator │    │              │    │ • ChromaDB    │
   └──────────────┘    └──────────────┘    └──────────────┘
```

---

## 8. 设计总结

| 特性 | 实现方式 |
|------|---------|
| **工具插件** | BaseTool 抽象类 + ToolRegistry，支持 OpenAI function calling 格式导出 |
| **模型插件** | BaseLLMProvider 统一接口，多提供商适配（OpenAI/Anthropic/本地） |
| **存储插件** | Protocol 协议定义，KV Store / Vector Store 可插拔切换 |
| **内置工具** | WebSearch、FileRead、DateTime、Calculator 开箱即用 |
