# Context Builder 模块详细设计

## 1. 概述

Context Builder 负责将 Session 内部的消息转换为 LLM API 所需的请求格式。它是 Agent 核心循环中的关键组件，在每次 LLM 调用前构建上下文消息列表。

**对应源码:** `core/context_builder.py`

### 职责
- 注入系统提示词 (System Prompt)
- 应用滑动窗口截断历史消息
- 将内部消息模型转换为 LLM 适配的消息格式
- 保留工具调用的完整上下文链

## 2. 类设计

### ContextBuilder

```python
class ContextBuilder:
    def __init__(self, system_prompt: str)
    def build(self, session: Session, window_size: int = 48) -> list[LLMMessage]
```

#### 构造函数参数

| 参数 | 类型 | 说明 |
|---|---|---|
| `system_prompt` | `str` | 系统提示词，来自配置中的 `agent.system_prompt` |

#### build 方法

将 Session 消息转换为 LLM API 格式的消息列表。

**处理流程:**
1. 从 Session 获取滑动窗口内的上下文消息 (`session.get_context_messages(window_size)`)
2. 在消息列表头部插入 System Prompt 消息
3. 遍历上下文消息，按角色映射为 `LLMMessage`:
   - `USER` → `Role.USER`, 仅传递 content
   - `ASSISTANT` → `Role.ASSISTANT`, 传递 content + tool_calls
   - `TOOL` → `Role.TOOL`, 传递 content + tool_call_id

**返回格式:**
```python
[
    LLMMessage(role=Role.SYSTEM, content="系统提示词"),
    LLMMessage(role=Role.USER, content="用户消息"),
    LLMMessage(role=Role.ASSISTANT, content=None, tool_calls=[...]),
    LLMMessage(role=Role.TOOL, content="工具结果", tool_call_id="call-1"),
    ...
]
```

## 3. 滑动窗口机制

滑动窗口的实际截断逻辑在 `Session.get_context_messages()` 中实现：
- 过滤出所有非 SYSTEM 消息
- 取最后 N 条 (`window_size` 默认 48)
- 保留所有 SYSTEM 消息（如有）

这意味着 Context Builder 本身不直接控制窗口大小，而是依赖 Session 提供已截断的消息列表。

## 4. 与主文档的对应关系

| agent-design.md 章节 | 本模块覆盖内容 |
|---|---|
| 上下文构建器 - System Prompt 注入 | ✅ 构造函数接收并注入系统提示词 |
| 上下文构建器 - 滑动窗口截断 | ✅ 通过 Session.get_context_messages 实现 |
| 上下文构建器 - 消息格式转换 | ✅ USER/ASSISTANT/TOOL 角色映射 |

## 5. 依赖关系

```
ContextBuilder
    ├── models.session.Session       (获取上下文消息)
    ├── models.message.MessageRole   (角色判断)
    └── llm.base.LLMMessage, Role    (输出格式)
```

## 6. 注意事项

- Context Builder 是无状态的，每次 `build` 调用都是独立的
- 系统提示词在初始化时固定，运行时不可变（如需动态修改需重新实例化）
- 窗口大小默认 48 条消息，约对应 10-15 轮对话（含工具调用消息）
