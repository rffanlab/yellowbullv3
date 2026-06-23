# Tool Registry 模块详细设计

## 1. 概述

Tool Registry 提供工具的统一注册、查询和管理功能。采用装饰器模式实现工具的自动注册，支持运行时动态增删工具。

**对应源码:** `tools/registry.py`

### 职责
- 全局工具注册表（类级别单例）
- 装饰器驱动的自动注册
- LLM Function Calling Schema 生成
- 测试环境清理支持

## 2. 类设计

### ToolRegistry

```python
class ToolRegistry:
    _tools: dict[str, BaseTool] = {}  # 类变量，全局共享

    @classmethod
    def register(cls, tool: BaseTool) -> None
    @classmethod
    def unregister(cls, name: str) -> bool
    @classmethod
    def get(cls, name: str) -> BaseTool | None
    @classmethod
    def list_all(cls) -> list[BaseTool]
    @classmethod
    def to_function_definitions(cls) -> list[dict[str, Any]]
    @classmethod
    def clear(cls) -> None
```

#### 方法说明

| 方法 | 参数 | 返回值 | 说明 |
|---|---|---|---|
| `register` | tool: BaseTool | None | 注册工具到全局映射表 |
| `unregister` | name: str | bool | 移除指定名称的工具，返回是否成功 |
| `get` | name: str | BaseTool \| None | 按名称获取工具实例 |
| `list_all` | — | list[BaseTool] | 返回所有已注册工具的列表 |
| `to_function_definitions` | — | list[dict] | 转换为 LLM Function Calling 格式 |
| `clear` | — | None | 清空所有注册（测试用） |

### to_function_definitions 输出格式

```python
[
    {
        "name": "calculator",
        "description": "Evaluate a mathematical expression...",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", ...}
            },
            "required": ["expression"]
        }
    },
    ...
]
```

## 3. 装饰器注册模式

### register_tool

```python
def register_tool(
    name: str,
    description: str,
    parameters: dict[str, Any]
) -> Callable[[type[T]], type[T]]
```

类装饰器，用于自动注册工具。

**使用示例:**
```python
@register_tool(
    name="calculator",
    description="Evaluate a mathematical expression.",
    parameters={...},
)
class CalculatorTool(BaseTool):
    ...
```

**执行时机:** 模块导入时立即执行装饰器，创建工具实例并注册到 ToolRegistry。

**内部流程:**
1. 被装饰的类实例化 (`instance = cls()`)
2. 设置 `_info` 属性为 `ToolInfo(name, description, parameters)`
3. 调用 `ToolRegistry.register(instance)` 注册到全局映射表

## 4. 工具导入机制

内置工具的自动注册依赖 `tools/builtins/__init__.py`:

```python
from tools.builtins.calculator import CalculatorTool
from tools.builtins.current_time import CurrentTimeTool
from tools.builtins.web_search import WebSearchTool
```

当执行 `import tools.builtins` 时，所有子模块被导入，触发各自的 `@register_tool` 装饰器。

## 5. 与主文档的对应关系

| agent-design.md 章节 | 本模块覆盖内容 |
|---|---|
| 工具系统 - ToolRegistry | ✅ 注册、查询、列表功能 |
| 工具系统 - Function Calling Schema | ✅ to_function_definitions |
| 工具系统 - 装饰器自动注册 | ✅ register_tool 装饰器 |

## 6. 依赖关系

```
tools/registry
    └── tools.base.BaseTool, ToolInfo
```

## 7. 注意事项

- `_tools` 为类变量，所有实例共享同一注册表
- `clear()` 主要用于测试环境隔离，生产环境不应调用
- 装饰器在模块导入时执行，工具实例为单例
- 新增内置工具需在 `tools/builtins/__init__.py` 中添加 import
