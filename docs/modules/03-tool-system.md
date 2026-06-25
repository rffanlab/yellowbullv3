# 工具系统详细设计（含热加载）

## 1. 设计目标

| 目标 | 说明 |
|------|------|
| **统一协议** | 所有工具收敛到同一接口，Agent/LLM 侧零感知差异 |
| **即插即用** | 新增工具 = 实现基类 + 注册，不影响已有代码 |
| **热加载** | YAML 配置变更 → 动态启停/刷新工具，无需重启进程 |
| **JSON Schema 自动生成** | Python type hints → JSON Schema → LLM tool definition，零手动维护 |
| **错误隔离** | 单个工具失败不崩溃，返回结构化错误给 LLM 重试 |
| **超时控制** | 每个工具有独立 timeout，防止阻塞 Agent 主循环 |

---

## 2. 协议设计 `tools/protocol.py`

```python
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolParameter:
    """工具参数定义（对应 JSON Schema property）"""
    type: str                          # "string" | "number" | "integer" | "boolean" | "array" | "object"
    description: str = ""
    enum: list[Any] | None = None      # 枚举值约束
    default: Any | None = None


@dataclass(frozen=True)
class ToolSchema:
    """工具的 JSON Schema（用于 LLM function calling）"""
    name: str
    description: str
    parameters: dict[str, ToolParameter]   # property_name → ToolParameter

    def to_json_schema(self) -> dict[str, Any]:
        properties = {}
        required = []
        for pname, param in self.parameters.items():
            prop = {
                "type": param.type,
                "description": param.description,
            }
            if param.enum:
                prop["enum"] = param.enum
            if param.default is not None:
                prop["default"] = param.default
            properties[pname] = prop
        return {
            "type": "object",
            "properties": properties,
            # 所有参数都 required（LLM 必须提供值，缺省用 default）
            "required": list(self.parameters.keys()),
        }


@dataclass(frozen=True)
class ToolInput:
    """工具调用输入（来自 LLM 的 function call arguments）"""
    tool_name: str
    arguments: dict[str, Any]   # 已解析的 JSON dict


@dataclass
class ToolOutput:
    """工具执行结果"""
    success: bool
    content: str                          # 成功时的文本结果
    error: str | None = None              # 失败时的错误信息
    metadata: dict[str, Any] = field(default_factory=dict)   # 附加元数据（耗时等）

    @classmethod
    def ok(cls, content: str, **meta) -> "ToolOutput":
        return cls(success=True, content=content, metadata=meta)

    @classmethod
    def fail(cls, error: str, **meta) -> "ToolOutput":
        return cls(success=False, content="", error=error, metadata=meta)


# ToolDefinition 统一由 llm.protocol 定义，此处导入使用
from llm.protocol import ToolDefinition
```

---

## 3. 基类设计 `tools/base.py`

```python
from abc import ABC, abstractmethod
import asyncio
import time
import logging
from typing import Any

from tools.protocol import ToolSchema, ToolInput, ToolOutput

logger = logging.getLogger(__name__)


class BaseTool(ABC):
    """
    所有工具必须实现的接口。

    设计原则：
    - execute() 为 async，统一异步 IO
    - 输入输出使用内部协议对象
    - 每个实例可独立配置（settings dict）
    - 实现 _build_schema() 自动生成 JSON Schema
    """

    # ==================== 子类必须实现 ====================

    @property
    @abstractmethod
    def name(self) -> str:
        """工具唯一标识名"""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """工具描述（LLM 用来判断何时调用）"""
        ...

    @abstractmethod
    def _build_schema(self) -> ToolSchema:
        """构建工具的 JSON Schema（定义参数名、类型、描述）"""
        ...

    @abstractmethod
    async def execute(self, input: ToolInput) -> ToolOutput:
        """
        执行工具逻辑。

        Args:
            input: 包含 tool_name + arguments dict

        Returns:
            ToolOutput（success=True/False）

        Raises:
            不应抛出异常——所有错误应返回 ToolOutput.fail()
        """
        ...

    # ==================== 可选覆盖 ====================

    async def on_enable(self):
        """工具启用时调用（如建立连接池）"""
        pass

    async def on_disable(self):
        """工具禁用时调用（如关闭连接池）"""
        pass

    async def reload_config(self, settings: dict[str, Any]):
        """热更新配置时调用，子类覆盖以应用新配置"""
        pass

    # ==================== 公共方法 ====================

    @property
    def schema(self) -> ToolSchema:
        return self._build_schema()

    async def execute_with_guard(
        self,
        input: ToolInput,
        timeout: float | None = None,
    ) -> ToolOutput:
        """
        带防护的执行包装器：超时 + 异常捕获 + 耗时统计。

        Agent Core 应调用此方法而非直接调 execute()。
        """
        start = time.monotonic()
        try:
            if timeout:
                result = await asyncio.wait_for(self.execute(input), timeout=timeout)
            else:
                result = await self.execute(input)

            elapsed = time.monotonic() - start
            result.metadata["elapsed_ms"] = round(elapsed * 1000, 1)
            return result

        except asyncio.TimeoutError:
            elapsed = time.monotonic() - start
            logger.error(f"Tool {self.name} timed out after {timeout}s")
            return ToolOutput.fail(
                error=f"Tool execution timed out after {timeout}s",
                elapsed_ms=round(elapsed * 1000, 1),
            )

        except Exception as e:
            elapsed = time.monotonic() - start
            logger.exception(f"Tool {self.name} raised unexpected error")
            return ToolOutput.fail(
                error=f"Internal error: {type(e).__name__}: {e}",
                elapsed_ms=round(elapsed * 1000, 1),
            )


# ==================== 装饰器注册方式（可选）====================

_tool_registry_instance = None   # 延迟初始化


def tool(name: str, description: str):
    """
    装饰器方式注册工具。

    Usage:
        @tool("get_weather", "查询指定城市的天气")
        class WeatherTool(BaseTool):
            ...
    """
    def decorator(cls: type[BaseTool]):
        global _tool_registry_instance
        if _tool_registry_instance:
            _tool_registry_instance._register(name, cls)
        return cls
    return decorator
```

### 3.1 基类设计考量

| 决策 | 理由 |
|------|------|
| `execute_with_guard()` 而非直接调 `execute()` | 统一超时、异常捕获、耗时统计，子类只需关注业务逻辑 |
| `on_enable/on_disable` 生命周期钩子 | 热加载时自动调用，工具可管理连接池等资源 |
| `reload_config()` 热更新入口 | ConfigManager 回调触发，工具自行应用新配置 |
| `@tool` 装饰器可选 | 支持声明式注册和显式注册两种方式 |

---

## 4. 内置工具实现

### 4.1 时间查询 `tools/builtin/current_time.py`

```python
from datetime import datetime, timezone
from tools.base import BaseTool
from tools.protocol import ToolSchema, ToolParameter, ToolInput, ToolOutput


class CurrentTimeTool(BaseTool):
    """获取当前时间（支持时区）"""

    @property
    def name(self) -> str:
        return "current_time"

    @property
    def description(self) -> str:
        return "Get the current date and time. Supports timezone specification."

    def _build_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters={
                "timezone": ToolParameter(
                    type="string",
                    description="IANA timezone name (e.g. 'Asia/Shanghai'). Default: UTC.",
                    default="UTC",
                ),
                "format": ToolParameter(
                    type="string",
                    description="Output format string. Default: ISO 8601.",
                    default="%Y-%m-%d %H:%M:%S %Z",
                ),
            },
        )

    async def execute(self, input: ToolInput) -> ToolOutput:
        tz_name = input.arguments.get("timezone", "UTC")
        fmt = input.arguments.get("format", "%Y-%m-%d %H:%M:%S %Z")

        try:
            from zoneinfo import ZoneInfo
            tz = ZoneInfo(tz_name)
            now = datetime.now(tz)
            result = now.strftime(fmt)
            return ToolOutput.ok(result)
        except Exception as e:
            return ToolOutput.fail(f"Invalid timezone '{tz_name}': {e}")
```

### 4.2 计算器 `tools/builtin/calculator.py`

```python
import ast
import operator
from tools.base import BaseTool
from tools.protocol import ToolSchema, ToolParameter, ToolInput, ToolOutput


# 安全的运算符白名单（防止 eval 注入）
_SAFE_OPS = {
    "Add": operator.add,
    "Sub": operator.sub,
    "Mult": operator.mul,
    "Div": operator.truediv,
    "Pow": operator.pow,
    "Mod": operator.mod,
    "USub": operator.neg,
}


class CalculatorTool(BaseTool):
    """安全计算器——仅支持基础数学运算，无代码注入风险"""

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return "Evaluate a mathematical expression. Supports +, -, *, /, **, %, parentheses."

    def _build_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters={
                "expression": ToolParameter(
                    type="string",
                    description="Math expression to evaluate (e.g. '2 + 3 * 4', '(10 - 2) / 3')",
                ),
            },
        )

    async def execute(self, input: ToolInput) -> ToolOutput:
        expr = input.arguments.get("expression", "").strip()
        try:
            result = _safe_eval(expr)
            return ToolOutput.ok(str(result))
        except ZeroDivisionError:
            return ToolOutput.fail("Division by zero")
        except ValueError as e:
            return ToolOutput.fail(str(e))


def _safe_eval(expression: str):
    """
    基于 AST 的安全表达式求值。

    仅允许：数字、二元运算、一元负号、幂运算、取模
    禁止：函数调用、属性访问、变量引用
    """
    tree = ast.parse(expression, mode="eval")

    def _eval_node(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        elif isinstance(node, ast.BinOp):
            op_type = type(node.op).__name__
            if op_type not in _SAFE_OPS:
                raise ValueError(f"Operator '{op_type}' not allowed")
            left = _eval_node(node.left)
            right = _eval_node(node.right)
            return _SAFE_OPS[op_type](left, right)
        elif isinstance(node, ast.UnaryOp):
            op_type = type(node.op).__name__
            if op_type not in _SAFE_OPS:
                raise ValueError(f"Operator '{op_type}' not allowed")
            operand = _eval_node(node.operand)
            return _SAFE_OPS[op_type](operand)
        else:
            raise ValueError(f"Expression not allowed: {type(node).__name__}")

    return _eval_node(tree.body)
```

### 4.3 网页搜索 `tools/builtin/web_search.py`

```python
from tools.base import BaseTool
from tools.protocol import ToolSchema, ToolParameter, ToolInput, ToolOutput


class WebSearchTool(BaseTool):
    """
    网页搜索工具。

    支持引擎：
    - duckduckgo（免费，无需 API key）
    - google（需要 SERPAPI key）

    热更新支持：修改 engine / api_key / max_results → reload_config() 立即生效
    """

    def __init__(self, settings: dict | None = None):
        self._settings = settings or {}
        self._engine = self._settings.get("engine", "duckduckgo")
        self._api_key = self._settings.get("api_key", "")
        self._max_results = self._settings.get("max_results", 5)

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Search the web for information. Returns top results with title, URL, and snippet."

    def _build_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters={
                "query": ToolParameter(
                    type="string",
                    description="Search query (e.g. 'latest AI news 2026')",
                ),
                "max_results": ToolParameter(
                    type="integer",
                    description="Maximum number of results to return.",
                    default=5,
                ),
            },
        )

    async def execute(self, input: ToolInput) -> ToolOutput:
        query = input.arguments.get("query", "")
        max_results = min(input.arguments.get("max_results", self._max_results), 10)

        try:
            if self._engine == "google":
                results = await self._search_google(query, max_results)
            else:
                results = await self._search_duckduckgo(query, max_results)

            if not results:
                return ToolOutput.ok(f"No results found for '{query}'")

            lines = [f"Search results for '{query}':"]
            for i, r in enumerate(results, 1):
                lines.append(f"[{i}] {r['title']}")
                lines.append(f"    URL: {r['url']}")
                lines.append(f"    {r['snippet']}")
                lines.append("")

            return ToolOutput.ok("\n".join(lines))

        except Exception as e:
            return ToolOutput.fail(f"Search failed: {e}")

    async def _search_duckduckgo(self, query: str, max_results: int):
        import httpx
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json", "no_redirect": "1"},
            )
            data = resp.json()
            results = []
            # Abstract 结果
            for item in (data.get("AbstractRelatedTopics") or [])[:max_results]:
                if "Text" in item and "FirstURL" in item:
                    results.append({
                        "title": item.get("Text", "").split(".")[0],
                        "url": item.get("FirstURL", ""),
                        "snippet": item.get("Text", ""),
                    })
            return results

    async def _search_google(self, query: str, max_results: int):
        import httpx
        if not self._api_key:
            raise ValueError("Google SERP API key not configured")

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                "https://serpapi.com/search",
                params={
                    "q": query,
                    "api_key": self._api_key,
                    "num": max_results,
                },
            )
            data = resp.json()
            results = []
            for item in (data.get("organic_results") or [])[:max_results]:
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                })
            return results

    async def reload_config(self, settings: dict):
        """热更新：引擎、API key、最大结果数立即生效"""
        self._settings = settings
        self._engine = settings.get("engine", "duckduckgo")
        self._api_key = settings.get("api_key", "")
        self._max_results = settings.get("max_results", 5)
```

---

## 5. 工具注册表 `tools/registry.py`

```python
"""
ToolRegistry —— 全局单例，管理所有工具的注册、查询、热加载。

设计原则：
- 线程安全（asyncio lock）
- 支持动态启停（热更新入口）
- 按名称索引 O(1) 查找
- 与 ConfigManager 集成，配置变更自动刷新
"""

import asyncio
import logging
from typing import Any, Type

from llm.protocol import ToolDefinition
from tools.base import BaseTool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """工具注册表（全局单例）"""

    _instance: "ToolRegistry | None" = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self._tools: dict[str, BaseTool] = {}           # name → instance
        self._classes: dict[str, Type[BaseTool]] = {}   # name → class (用于重建)
        self._enabled: dict[str, bool] = {}             # name → enabled flag
        self._settings: dict[str, dict[str, Any]] = {}  # name → config settings
        self._lock = asyncio.Lock()

    @classmethod
    def reset(cls):
        """测试用：重置单例"""
        cls._instance = None

    # ==================== 注册 ====================

    async def register(self, tool: BaseTool, enabled: bool = True, settings: dict | None = None):
        """注册工具实例"""
        async with self._lock:
            name = tool.name
            self._tools[name] = tool
            self._enabled[name] = enabled
            self._settings[name] = settings or {}
            if enabled:
                await tool.on_enable()
            logger.info(f"Tool registered: {name} (enabled={enabled})")

    async def _register(self, name: str, cls: Type[BaseTool]):
        """装饰器内部调用：注册类（延迟实例化）"""
        self._classes[name] = cls

    # ==================== 查询 ====================

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def list_definitions(self) -> list[ToolDefinition]:
        """获取所有已启用工具的 LLM 定义"""
        definitions = []
        for name, tool in self._tools.items():
            if not self._enabled.get(name, True):
                continue
            schema_dict = tool.schema.to_json_schema()
            definitions.append(ToolDefinition(
                name=tool.name,
                description=tool.description,
                parameters=schema_dict,
            ))
        return definitions

    def list_names(self) -> list[str]:
        return [name for name, enabled in self._enabled.items() if enabled]

    # ==================== 热加载操作 ====================

    async def set_enabled(self, name: str, enabled: bool):
        """动态启用/禁用工具"""
        async with self._lock:
            tool = self._tools.get(name)
            if tool is None:
                logger.warning(f"Tool '{name}' not found for enable/disable")
                return

            old_enabled = self._enabled.get(name, True)
            if old_enabled == enabled:
                return   # 无变化

            self._enabled[name] = enabled
            if enabled:
                await tool.on_enable()
                logger.info(f"Tool enabled: {name}")
            else:
                await tool.on_disable()
                logger.info(f"Tool disabled: {name}")

    async def update_settings(self, name: str, settings: dict[str, Any]):
        """热更新工具配置"""
        async with self._lock:
            tool = self._tools.get(name)
            if tool is None:
                return

            old_settings = self._settings.get(name, {})
            # 检测变化
            new_merged = {**old_settings, **settings}
            if old_settings == new_merged:
                return   # 无变化

            self._settings[name] = new_merged
            await tool.reload_config(new_merged)
            logger.info(f"Tool settings updated: {name}")

    async def reload_all(self, config_tools: dict[str, dict]):
        """
        批量刷新所有工具配置（ConfigManager 回调入口）。

        Args:
            config_tools: tools.builtin from Settings (name → {enabled, settings})
        """
        async with self._lock:
            for name, cfg in config_tools.items():
                enabled = cfg.get("enabled", True)
                settings = cfg.get("settings", {})

                await self.set_enabled(name, enabled)
                if settings:
                    await self.update_settings(name, settings)

    # ==================== 执行 ====================

    async def execute(self, tool_name: str, arguments: dict[str, Any], timeout: float | None = None) -> "ToolOutput":
        """查找并执行工具（Agent Core 调用入口）"""
        from tools.protocol import ToolInput, ToolOutput

        tool = self._tools.get(tool_name)
        if tool is None:
            return ToolOutput.fail(f"Unknown tool: '{tool_name}'")

        if not self._enabled.get(tool_name, True):
            return ToolOutput.fail(f"Tool '{tool_name}' is disabled")

        input = ToolInput(tool_name=tool_name, arguments=arguments)
        return await tool.execute_with_guard(input, timeout=timeout)
```

---

## 6. 热加载集成 `tools/config_bridge.py`

```python
"""
ConfigManager ↔ ToolRegistry 桥接。

当 YAML 中 tools.builtin.* 配置变更时，自动刷新注册表。
"""

from config.manager import ConfigManager, get_manager


def setup_tool_config_watching():
    """在 ConfigManager 上注册工具配置监听回调"""
    manager = get_manager()

    @manager.on_change("tools")
    async def on_tools_config_changed(old_val, new_val):
        from tools.registry import ToolRegistry
        settings = manager.settings
        registry = ToolRegistry()
        await registry.reload_all(settings.tools.builtin)


# 在 main.py startup 中调用：
# setup_tool_config_watching()
```

### 6.1 热加载流程

```
YAML 变更 (tools.builtin.web_search.enabled: false)
    │
    ▼
ConfigManager._watch_loop() 检测到 hash 变化
    │
    ▼
ConfigManager.reload() → _notify_changes()
    │
    ▼
@manager.on_change("tools") 回调触发
    │
    ▼
ToolRegistry.reload_all(config_tools)
    │
    ├── set_enabled("web_search", False)
    │       └── tool.on_disable()   # 清理资源
    │
    └── update_settings(...)        # 其他工具配置刷新
```

---

## 7. 架构总览

```
                    ┌─────────────────────┐
                    │     Agent Core      │
                    │  registry.execute() │
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │    ToolRegistry      │
                    │  (全局单例)           │
                    │                      │
                    │  • register()        │
                    │  • list_definitions()│
                    │  • set_enabled()     │
                    │  • reload_all()      │
                    └──────────┬──────────┘
                               │ holds instances of
              ┌────────────────┼────────────────┐
              ▼                ▼                 ▼
    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
    │ CurrentTime │  │ Calculator  │  │ WebSearch   │
    │ Tool        │  │ Tool        │  │ Tool        │
    └─────────────┘  └─────────────┘  └─────────────┘
         ▲                ▲                 ▲
         │                │                 │
         └────────────────┼─────────────────┘
                          │ implements
                          ▼
               ┌─────────────────────┐
               │    BaseTool (ABC)   │
               │  execute()          │
               │  _build_schema()    │
               │  on_enable/disable()│
               │  reload_config()    │
               └─────────────────────┘

    ┌──────────────────────────────────────────┐
    │         ConfigManager Bridge             │
    │  YAML change → ToolRegistry.reload_all() │
    └──────────────────────────────────────────┘
```

---

## 8. 新增工具步骤清单

以添加 `EmailTool` 为例：

```python
# tools/builtin/email.py
from tools.base import BaseTool
from tools.protocol import ToolSchema, ToolParameter, ToolInput, ToolOutput


class EmailTool(BaseTool):
    @property
    def name(self) -> str:
        return "send_email"

    @property
    def description(self) -> str:
        return "Send an email to the specified recipient."

    def _build_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters={
                "to": ToolParameter(type="string", description="Recipient email address"),
                "subject": ToolParameter(type="string", description="Email subject line"),
                "body": ToolParameter(type="string", description="Email body text"),
            },
        )

    async def execute(self, input: ToolInput) -> ToolOutput:
        # SMTP 发送逻辑...
        return ToolOutput.ok("Email sent successfully")


# main.py startup
registry = ToolRegistry()
await registry.register(EmailTool(), enabled=True)
```

**无需修改：** Agent Core、LLM 层、Session Manager、API Server。

---

## 9. 设计总结

| 特性 | 实现方式 |
|------|---------|
| **统一协议** | `ToolInput` / `ToolOutput` / `ToolSchema` 内部标准对象 |
| **即插即用** | ABC + Registry，新增工具 = 1 个文件 |
| **JSON Schema 自动生成** | `_build_schema()` → `to_json_schema()` → LLM tool definition |
| **热加载** | ConfigManager 回调 → `reload_all()` → 动态启停/刷新配置 |
| **错误隔离** | `execute_with_guard()` 统一超时 + 异常捕获，返回结构化错误 |
| **生命周期管理** | `on_enable/on_disable` 钩子，工具可管理连接池等资源 |
| **安全执行** | Calculator 用 AST 白名单，WebSearch 用 httpx timeout |

---

## 10. 多模态工具支持

### 10.1. 设计目标

扩展工具系统以支持多模态输入/输出：
- **图像输入**：截图分析、OCR、视觉问答
- **音频输入**：语音转文字、音频分析
- **文件输入**：文档解析、代码审查

### 10.2. 协议扩展 `tools/multimodal_protocol.py`

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MediaType(str, Enum):
    IMAGE = "image"           # 图片（base64 / URL）
    AUDIO = "audio"           # 音频
    VIDEO = "video"           # 视频
    DOCUMENT = "document"     # 文档（PDF、Word 等）
    CODE_FILE = "code_file"   # 代码文件


@dataclass(frozen=True)
class MultimodalContent:
    """多模态内容"""
    media_type: MediaType       # 媒体类型
    data: str                   # base64 编码数据或 URL
    mime_type: str | None = None  # MIME 类型
    file_name: str | None = None  # 原始文件名


@dataclass(frozen=True)
class MultimodalToolInput:
    """多模态工具输入"""
    text: str                           # 文本描述（可选）
    contents: list[MultimodalContent] = field(default_factory=list)  # 媒体内容列表


@dataclass(frozen=True)
class MultimodalToolOutput:
    """多模态工具输出"""
    text: str = ""                      # 文本结果
    media: MultimodalContent | None = None  # 媒体结果（如生成的图片）
```

### 10.3. 图像分析工具 `tools/image_analysis.py`

```python
from tools.base import BaseTool, ToolInput, ToolOutput, ToolSchema


class ImageAnalysisTool(BaseTool):
    """
    图像分析工具。

    支持：
    - OCR 文字识别
    - 图像描述生成
    - 截图代码审查
    """

    name = "image_analysis"
    description = "分析图片内容，包括 OCR、描述生成等"

    def _build_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "image_url": {"type": "string", "description": "图片 URL"},
                    "operation": {
                        "type": "string",
                        "enum": ["ocr", "describe", "code_review"],
                        "description": "分析操作类型",
                    },
                },
                "required": ["image_url", "operation"],
            },
        )

    async def execute(self, arguments: dict[str, Any]) -> ToolOutput:
        image_url = arguments["image_url"]
        operation = arguments["operation"]

        if operation == "ocr":
            return await self._perform_ocr(image_url)
        elif operation == "describe":
            return await self._describe_image(image_url)
        elif operation == "code_review":
            return await self._review_code_screenshot(image_url)
        else:
            return ToolOutput(error=f"Unknown operation: {operation}")

    async def _perform_ocr(self, image_url: str) -> ToolOutput:
        """OCR 文字识别"""
        # TODO: 集成 OCR 引擎（如 PaddleOCR、Tesseract）
        return ToolOutput(text="[OCR result placeholder]")

    async def _describe_image(self, image_url: str) -> ToolOutput:
        """图像描述生成（使用多模态 LLM）"""
        # TODO: 调用支持视觉输入的 LLM
        return ToolOutput(text="[Image description placeholder]")

    async def _review_code_screenshot(self, image_url: str) -> ToolOutput:
        """截图代码审查"""
        # TODO: OCR + 代码分析
        return ToolOutput(text="[Code review from screenshot placeholder]")
```

---

## 11. 流式工具结果

### 11.1. 设计目标

对于耗时较长的工具调用（如大文件处理、复杂搜索），支持流式返回中间结果，提升用户体验。

### 11.2. 协议扩展 `tools/streaming.py`

```python
from dataclasses import dataclass, field
from enum import Enum


class StreamEventType(str, Enum):
    CHUNK = "chunk"             # 数据块
    PROGRESS = "progress"       # 进度更新
    COMPLETE = "complete"       # 完成信号
    ERROR = "error"             # 错误信号


@dataclass(frozen=True)
class StreamEvent:
    """流式事件"""
    event_type: StreamEventType   # 事件类型
    data: str = ""                # 数据内容
    progress_percent: float = 0.0  # 进度百分比 [0, 100]
    metadata: dict[str, Any] = field(default_factory=dict)  # 额外元数据


class StreamingToolOutput(ToolOutput):
    """流式工具输出"""

    def __init__(self, events=None, **kwargs):
        super().__init__(**kwargs)
        self._events = iter(events or [])

    async def stream(self):
        """异步迭代器，逐个产出事件"""
        for event in self._events:
            yield event
        yield StreamEvent(event_type=StreamEventType.COMPLETE)
```

### 11.3. Agent Core 集成

在 `agent/core.py` 的工具执行层增加流式支持：

```python
async def execute_tool_streaming(self, tool_call: ToolCall):
    """执行工具并流式返回结果"""
    result = await self.tool_registry.execute(tool_call)

    if isinstance(result, StreamingToolOutput):
        async for event in result.stream():
            yield event
    else:
        # 非流式结果包装为单次事件
        yield StreamEvent(
            event_type=StreamEventType.COMPLETE,
            data=result.text,
        )
```

---

## 12. 更新后的设计总结

| 特性 | 实现方式 |
|------|---------|
| **统一协议** | `ToolInput` / `ToolOutput` / `ToolSchema` 内部标准对象 |
| **即插即用** | ABC + Registry，新增工具 = 1 个文件 |
| **JSON Schema 自动生成** | `_build_schema()` → `to_json_schema()` → LLM tool definition |
| **热加载** | ConfigManager 回调 → `reload_all()` → 动态启停/刷新配置 |
| **错误隔离** | `execute_with_guard()` 统一超时 + 异常捕获，返回结构化错误 |
| **生命周期管理** | `on_enable/on_disable` 钩子，工具可管理连接池等资源 |
| **安全执行** | Calculator 用 AST 白名单，WebSearch 用 httpx timeout |
| **多模态支持** | `MultimodalContent` / `ImageAnalysisTool` 扩展协议 |
| **流式结果** | `StreamingToolOutput` + `StreamEvent` 异步迭代器 |

---

## 13. A2A 工具集成

### 13.1. 设计目标

在工具系统中原生支持 A2A（Agent-to-Agent）协议，使工具能够作为子代理被调度：
- **A2A 工具包装**：将远程 agent 暴露为本地可调用的工具
- **任务委托**：主 agent 通过工具调用将子任务委托给专业 agent
- **状态跟踪**：支持任务的创建、查询、取消等全生命周期管理

### 13.2. A2A 工具基类 `tools/a2a_tool.py`

```python
import asyncio
import logging
from typing import Any

import httpx

from tools.base import BaseTool
from tools.protocol import ToolSchema, ToolInput, ToolOutput

logger = logging.getLogger(__name__)


class A2ABaseTool(BaseTool):
    """
    A2A 工具基类。

    将远程 agent 的 A2A API 封装为标准工具接口，
    主 agent 可通过工具调用与子 agent 交互。
    """

    def __init__(self, agent_card_url: str, timeout: float = 120.0):
        self._agent_card_url = agent_card_url
        self._timeout = timeout
        self._agent_card: dict[str, Any] | None = None
        self._client = httpx.AsyncClient(timeout=timeout)

    async def on_enable(self):
        """启用时加载 Agent Card"""
        try:
            resp = await self._client.get(self._agent_card_url)
            self._agent_card = resp.json()
            logger.info(
                f"A2A agent loaded: {self._agent_card.get('name', 'unknown')} "
                f"({self._agent_card.get('version', '?')})"
            )
        except Exception as e:
            logger.error(f"Failed to load Agent Card from {self._agent_card_url}: {e}")

    async def on_disable(self):
        await self._client.aclose()

    @property
    def agent_capabilities(self) -> list[str]:
        if self._agent_card:
            return self._agent_card.get("capabilities", [])
        return []

    @property
    def default_timeout(self) -> float:
        card_timeout = (self._agent_card or {}).get("defaultTimeout", 60_000)
        return card_timeout / 1000.0  # ms → seconds
```

### 13.3. A2A 任务提交工具 `tools/a2a_submit_task.py`

```python
from tools.a2a_tool import A2ABaseTool


class A2ASubmitTaskTool(A2ABaseTool):
    """向远程 agent 提交任务"""

    name = "a2a_submit_task"
    description = (
        "Submit a task to a specialized remote agent. "
        "Returns a task ID for subsequent status polling."
    )

    def _build_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The task description or question for the remote agent.",
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Optional additional context (files, preferences, etc.).",
                    },
                },
                "required": ["query"],
            },
        )

    async def execute(self, input: ToolInput) -> ToolOutput:
        query = input.arguments.get("query", "")
        metadata = input.arguments.get("metadata", {})

        try:
            payload = {
                "kind": "submit",
                "message": {
                    "kind": "user",
                    "content": {"kind": "text", "text": query},
                    "metadata": metadata,
                },
            }

            card = self._agent_card or {}
            push_url = card.get("defaultPushUrl")
            if push_url:
                payload["pushNotification"] = {
                    "token": push_url,
                    "acceptedFormats": ["text"],
                }

            resp = await self._client.post(
                f"{self._get_base_url()}/tasks", json=payload
            )
            task_data = resp.json()

            return ToolOutput.ok(
                text=f"Task submitted successfully.",
                metadata={
                    "taskId": task_data.get("id"),
                    "status": task_data.get("status", "submitted"),
                },
            )
        except Exception as e:
            logger.error(f"A2A task submission failed: {e}")
            return ToolOutput.fail(str(e))

    def _get_base_url(self) -> str:
        url = self._agent_card_url.rstrip("/")
        if "/.well-known/agent.json" in url:
            url = url.replace("/.well-known/agent.json", "")
        return url
```

### 13.4. A2A 任务状态查询工具 `tools/a2a_task_status.py`

```python
from tools.a2a_tool import A2ABaseTool


class A2ATaskStatusTool(A2ABaseTool):
    """查询远程 agent 任务状态"""

    name = "a2a_task_status"
    description = (
        "Get the status and result of a previously submitted task. "
        "Returns completed results or intermediate status."
    )

    def _build_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "The task ID returned from a2a_submit_task.",
                    },
                },
                "required": ["task_id"],
            },
        )

    async def execute(self, input: ToolInput) -> ToolOutput:
        task_id = input.arguments.get("task_id", "")

        try:
            resp = await self._client.get(
                f"{self._get_base_url()}/tasks/{task_id}"
            )
            task_data = resp.json()
            status = task_data.get("status", "unknown")

            # 提取结果内容
            result_text = ""
            if status == "completed" and "result" in task_data:
                result_text = self._extract_result(task_data["result"])

            return ToolOutput.ok(
                text=result_text or f"Task status: {status}",
                metadata={
                    "taskId": task_id,
                    "status": status,
                    "taskData": task_data,
                },
            )
        except Exception as e:
            logger.error(f"A2A task status query failed: {e}")
            return ToolOutput.fail(str(e))

    @staticmethod
    def _extract_result(result: dict) -> str:
        """从 A2A 结果中提取文本内容"""
        if isinstance(result, list):
            parts = []
            for item in result:
                if isinstance(item, dict):
                    if item.get("kind") == "text":
                        parts.append(item.get("text", ""))
                    elif item.get("kind") == "data":
                        parts.append(str(item.get("data", "")))
            return "\n".join(parts)
        elif isinstance(result, dict):
            if result.get("kind") == "text":
                return result.get("text", "")
        return str(result)

    def _get_base_url(self) -> str:
        url = self._agent_card_url.rstrip("/")
        if "/.well-known/agent.json" in url:
            url = url.replace("/.well-known/agent.json", "")
        return url
```

### 13.5. A2A 工具注册示例 `tools/a2a_registry.py`

```python
from tools.a2a_submit_task import A2ASubmitTaskTool
from tools.a2a_task_status import A2ATaskStatusTool


async def register_a2a_tools(registry, agent_configs: list[dict]):
    """
    批量注册 A2A 工具。

    Args:
        registry: ToolRegistry 实例
        agent_configs: YAML 中 a2a.agents[] 配置列表

            Example YAML:
                a2a:
                  agents:
                    - name: code_reviewer
                      card_url: http://code-agent:8080/.well-known/agent.json
                      enabled: true
    """
    for cfg in agent_configs:
        agent_name = cfg.get("name", "unknown")
        card_url = cfg.get("card_url")

        if not card_url:
            continue

        # 为每个远程 agent 创建命名空间化的工具
        submit_tool = A2ASubmitTaskTool(
            agent_card_url=card_url,
            timeout=cfg.get("timeout", 120.0),
        )
        status_tool = A2ATaskStatusTool(
            agent_card_url=card_url,
            timeout=30.0,
        )

        # 命名空间化工具名，避免冲突
        submit_tool.name = f"a2a_{agent_name}_submit"
        submit_tool.description = (
            f"Submit a task to the '{agent_name}' remote agent."
        )
        status_tool.name = f"a2a_{agent_name}_status"
        status_tool.description = (
            f"Get status of a task submitted to '{agent_name}'."
        )

        await registry.register(submit_tool, enabled=cfg.get("enabled", True))
        await registry.register(status_tool, enabled=cfg.get("enabled", True))
```

### 13.6. A2A 交互时序图

```
主 Agent                    ToolRegistry              远程 Agent (A2A)
   │                            │                           │
   │── LLM 决定委托任务 ─────────>│                           │
   │                            │── POST /tasks ────────────>│
   │                            │   {query: "..."}           │
   │                            │                           │
   │                            │<── 200 {id, status} ──────│
   │<─ ToolOutput.ok(id) ──────│                           │
   │                            │                           │
   │         (异步处理中...)      │                           │
   │                            │                           │
   │── LLM 轮询结果 ────────────>│                           │
   │                            │── GET /tasks/{id} ───────>│
   │                            │                           │
   │                            │<── 200 {status, result} ──│
   │<─ ToolOutput.ok(result) ──│                           │
```

---

## 14. 更新后的设计总结

| 特性 | 实现方式 |
|------|---------|
| **统一协议** | `ToolInput` / `ToolOutput` / `ToolSchema` 内部标准对象 |
| **即插即用** | ABC + Registry，新增工具 = 1 个文件 |
| **JSON Schema 自动生成** | `_build_schema()` → `to_json_schema()` → LLM tool definition |
| **热加载** | ConfigManager 回调 → `reload_all()` → 动态启停/刷新配置 |
| **错误隔离** | `execute_with_guard()` 统一超时 + 异常捕获，返回结构化错误 |
| **生命周期管理** | `on_enable/on_disable` 钩子，工具可管理连接池等资源 |
| **安全执行** | Calculator 用 AST 白名单，WebSearch 用 httpx timeout |
| **多模态支持** | `MultimodalContent` / `ImageAnalysisTool` 扩展协议 |
| **流式结果** | `StreamingToolOutput` + `StreamEvent` 异步迭代器 |
| **A2A 集成** | `A2ABaseTool` 基类，命名空间化工具名，全生命周期管理 |
