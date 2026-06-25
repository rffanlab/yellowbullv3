# Agent Core 详细设计（主循环）

## 1. 职责边界

| 职责 | 说明 |
|------|------|
| **LLM 编排** | 构建 prompt → 调用 LLM → 解析响应 → 路由到工具或回复 |
| **工具执行** | 接收 tool call → 查找/执行工具 → 结果回注对话 |
| **流式输出** | SSE 逐 chunk 推送，用户实时看到 Agent 思考过程 |
| **最大轮次保护** | tool_call → execute → retry 循环有上限，防止死循环 |
| **错误恢复** | LLM 超时/工具失败 → 结构化错误回注 → LLM 自主决策重试或放弃 |

---

## 2. Agent Core `agent/core.py`

```python
"""
AgentCore —— Agent 主循环。

编排流程：
1. 接收用户消息 + session_id
2. 从 SessionManager 获取上下文窗口
3. 构建 prompt（system + history + user message）
4. 调用 LLM（流式或非流式）
5. 解析响应：
   a. 纯文本回复 → 保存到 session → SSE 推送给用户
   b. tool call(s) → 执行工具 → 结果回注 → 回到步骤 3
6. 达到 max_tool_rounds 或无 tool call → 结束

流式输出协议（SSE）：
- event: "chunk"       data: {"content": "..."}          # LLM 文本增量
- event: "tool_start"   data: {"name": "...", "args": {...}}
- event: "tool_end"     data: {"name": "...", "result": "..."}
- event: "done"         data: {"status": "completed"|"error"}
"""

import asyncio
import json
import logging
from typing import Any, AsyncIterator

from llm.base import BaseLLM
from llm.protocol import Message as LLMMessage, Role as LLMLRole
from tools.registry import ToolRegistry
from session.manager import SessionManager, get_session_manager
from session.models import Message, ContextWindow
from session.token_estimator import estimate_message_tokens

logger = logging.getLogger(__name__)


# ==================== 协议转换 ====================

def _to_llm_message(msg: Message) -> LLMMessage:
    """将 session.Message 转换为 llm.protocol.Message"""
    role_map = {"user": LLMLRole.USER, "assistant": LLMLRole.ASSISTANT, "system": LLMLRole.SYSTEM}
    return LLMMessage(
        role=role_map.get(msg.role, LLMLRole.USER),
        content=msg.content or None,
        tool_calls=msg.tool_calls or None,
    )


class AgentCore:
    """Agent 主循环"""

    def __init__(
        self,
        llm: BaseLLM,
        tool_registry: ToolRegistry,
        max_tool_rounds: int = 10,
        system_prompt: str | None = None,
    ):
        self._llm = llm
        self._tools = tool_registry
        self._max_tool_rounds = max_tool_rounds

        # System prompt（可来自 YAML，也可运行时覆盖）
        self._system_prompt = system_prompt or """\
You are a helpful AI assistant. You have access to tools that can help you answer questions.
Use tools when they are relevant to the user's request.
Always respond in the same language as the user's question."""

    async def run(
        self,
        user_message: str,
        session_id: str,
        user_id: str = "anonymous",
    ) -> AsyncIterator[dict[str, Any]]:
        """
        执行 Agent 主循环，通过 SSE yield 流式事件。

        Yields:
            {"event": "chunk", "data": {"content": "..."}}
            {"event": "tool_start", "data": {"name": "...", "args": {...}}}
            {"event": "tool_end", "data": {"name": "...", "result": "..."}}
            {"event": "done", "data": {"status": "completed"}}

        Raises:
            RuntimeError: session 不存在或不可用
        """
        sm = get_session_manager()

        # 1. 保存用户消息
        user_msg = Message(
            id=_gen_id(),
            session_id=session_id,
            role="user",
            content=user_message,
            token_count=estimate_message_tokens(Message(id="", session_id="", role="user", content=user_message)),
        )
        await sm.append_message(user_msg)

        # 2. 主循环（最多 max_tool_rounds 轮 tool call）
        conversation_history: list[LLMMessage] = [
            LLMMessage(role=LLMLRole.SYSTEM, content=self._system_prompt)
        ]
        tool_round = 0

        try:
            while True:
                # 3. 构建 prompt（从 session 加载历史 + 本轮新增）
                context = sm.build_context_window(session_id)
                if context and context.messages:
                    for m in context.messages:
                        conversation_history.append(_to_llm_message(m))

                # 4. 获取工具定义（仅已启用的，已是 llm.protocol.ToolDefinition）
                tools_for_llm = self._tools.list_definitions() or None

                # 5. 调用 LLM（流式）
                assistant_content = ""
                tool_calls_buffer = []

                async for chunk in self._llm.chat_stream(
                    messages=conversation_history,
                    tools=tools_for_llm,
                ):
                    # SSE: 推送文本增量
                    if chunk.content:
                        yield {"event": "chunk", "data": {"content": chunk.content}}
                        assistant_content += chunk.content

                    # 累积 tool calls（流式模式下可能在最后一个 chunk 中）
                    if chunk.tool_calls:
                        tool_calls_buffer = chunk.tool_calls

                # 6. 判断是否有 tool call
                if not tool_calls_buffer:
                    # 纯文本回复 → 保存并结束
                    assistant_msg = Message(
                        id=_gen_id(),
                        session_id=session_id,
                        role="assistant",
                        content=assistant_content,
                        token_count=estimate_message_tokens(
                            Message(id="", session_id="", role="assistant", content=assistant_content)
                        ),
                    )
                    await sm.append_message(assistant_msg)
                    yield {"event": "done", "data": {"status": "completed"}}
                    return

                # 7. 执行工具（可能有多个 parallel tool calls）
                tool_round += 1
                if tool_round > self._max_tool_rounds:
                    error_msg = f"Max tool rounds ({self._max_tool_rounds}) reached."
                    logger.warning(error_msg)
                    await sm.append_message(Message(
                        id=_gen_id(), session_id=session_id, role="assistant",
                        content=error_msg, token_count=0,
                    ))
                    yield {"event": "done", "data": {"status": "error", "error": error_msg}}
                    return

                # 保存 assistant message（含 tool_calls）
                assistant_msg = Message(
                    id=_gen_id(),
                    session_id=session_id,
                    role="assistant",
                    content=assistant_content or "",
                    tool_calls=tool_calls_buffer,
                    token_count=estimate_message_tokens(
                        Message(id="", session_id="", role="assistant", content=assistant_content)
                    ),
                )
                await sm.append_message(assistant_msg)

                # 追加到 conversation_history（供下一轮 LLM 调用）
                conversation_history.append(LLMMessage(
                    role=LLMLRole.ASSISTANT,
                    content=assistant_content or None,
                    tool_calls=tool_calls_buffer or None,
                ))

                # 8. 执行每个 tool call，结果回注
                for tc in tool_calls_buffer:
                    tool_name = tc.get("function", {}).get("name", "")
                    try:
                        args_json = tc.get("function", {}).get("arguments", "{}")
                        if isinstance(args_json, str):
                            args = json.loads(args_json)
                        else:
                            args = args_json
                    except json.JSONDecodeError:
                        args = {}

                    # SSE: tool_start
                    yield {"event": "tool_start", "data": {"name": tool_name, "args": args}}

                    # 执行工具
                    result = await self._tools.execute(tool_name, args)

                    # SSE: tool_end
                    yield {
                        "event": "tool_end",
                        "data": {
                            "name": tool_name,
                            "success": result.success,
                            "content": result.content if result.success else "",
                            "error": result.error if not result.success else None,
                        },
                    }

                    # 保存工具结果到 session（作为 tool role message）
                    tool_result_msg = Message(
                        id=_gen_id(),
                        session_id=session_id,
                        role="tool",
                        content=result.content if result.success else f"Error: {result.error}",
                        token_count=0,
                    )
                    await sm.append_message(tool_result_msg)

                    # 追加到 conversation_history
                    conversation_history.append(LLMMessage(
                        role=LLMLRole.TOOL,
                        content=tool_result_msg.content or None,
                        tool_call_id=tc.get("id", ""),
                    ))

                # 回到步骤 3，用更新后的 history 重新调用 LLM


def _gen_id() -> str:
    import uuid
    return str(uuid.uuid4())
```

---

## 3. SSE 流式协议

### 3.1 事件类型

| Event | Data Schema | 说明 |
|-------|------------|------|
| `chunk` | `{"content": "str"}` | LLM 文本增量（逐 token/word） |
| `tool_start` | `{"name": "str", "args": {}}` | 工具开始执行 |
| `tool_end` | `{"name": "str", "success": bool, "content": "str", "error": "str?"}` | 工具执行完成 |
| `done` | `{"status": "completed"|"error", "error": "str?"}` | Agent 循环结束 |

### 3.2 前端消费示例

```javascript
const eventSource = new EventSource("/api/v1/agent/run?session_id=xxx");

eventSource.addEventListener("chunk", (e) => {
    const data = JSON.parse(e.data);
    appendToChat(data.content);   // 追加文本增量
});

eventSource.addEventListener("tool_start", (e) => {
    const data = JSON.parse(e.data);
    showToolIndicator(data.name); // 显示"正在使用 xxx 工具..."
});

eventSource.addEventListener("tool_end", (e) => {
    const data = JSON.parse(e.data);
    hideToolIndicator();          // 隐藏工具指示器
    if (!data.success) logError(data.error);
});

eventSource.addEventListener("done", (e) => {
    const data = JSON.parse(e.data);
    setInputEnabled(true);        // 恢复输入框
    eventSource.close();
});
```

---

## 4. Agent 主循环状态机

```
                    ┌─────────────┐
                    │   Start     │
                    │ (user msg)  │
                    └──────┬──────┘
                           ▼
                    ┌─────────────┐
              ┌────▶│ Call LLM    │
              │     │ (streaming) │
              │     └──────┬──────┘
              │            ▼
              │     ┌─────────────┐
              │     │ Has tools?  │
              │     └──┬──────┬───┘
              │        │yes   │no
              │        ▼      ▼
              │  ┌──────────┐  ┌──────────┐
              │  │ Execute  │  │ Save &   │
              │  │ tools    │  │ Return   │
              │  └────┬─────┘  └──────────┘
              │       │            ▲
              │       ▼            │
              │  round < max? ────┤
              │       │no          │
              │       ▼            │
              │  ┌──────────┐      │
              └──│ Error:   │──────┘
                 │ Max rounds│
                 └──────────┘
```

---

## 5. 非流式模式（批量/后台任务）

```python
async def run_sync(
    self,
    user_message: str,
    session_id: str,
    user_id: str = "anonymous",
) -> dict[str, Any]:
    """
    非流式执行（用于 API 同步端点或后台任务）。

    Returns:
        {"status": "completed", "content": "...", "tool_calls": [...]}
    """
    results = []
    async for event in self.run(user_message, session_id, user_id):
        results.append(event)

    # 聚合所有 chunk 为完整回复
    chunks = [r["data"]["content"] for r in results if r["event"] == "chunk"]
    done_event = next((r for r in results if r["event"] == "done"), None)

    return {
        "status": done_event["data"]["status"] if done_event else "error",
        "content": "".join(chunks),
        "events": results,   # 完整事件链（用于调试）
    }
```

---

## 6. 错误处理策略

| 场景 | 处理方式 |
|------|---------|
| **LLM 超时** | `run()` catch → yield `done(status="error")` → session 保存错误消息 |
| **工具执行失败** | ToolOutput.fail() → 错误文本回注 conversation_history → LLM 自主决策重试/换方案 |
| **Max rounds 超限** | 保存错误消息 → yield `done(status="error", error="...")` |
| **Session 不存在** | raise RuntimeError → API layer 返回 404 |
| **LLM 返回无效 JSON** | tool call arguments parse fail → args={} → 工具自行处理缺省参数 |

---

## 7. 设计总结

| 特性 | 实现方式 |
|------|---------|
| **主循环** | while True: LLM → parse → tools → retry（max rounds 保护） |
| **流式输出** | SSE event stream，前端实时渲染 |
| **工具编排** | parallel tool calls → sequential execution → results back to LLM |
| **错误恢复** | 结构化错误回注 conversation_history，LLM 自主决策 |
| **Session 集成** | 每步操作都 append_message()，完整审计链 |

---

## 8. A2A 代理间通信集成

### 8.1. 设计目标

在 Agent Core 中集成 A2A（Agent-to-Agent）协议支持：
- **任务委托**：主 agent 可将子任务委托给专业远程 agent
- **结果聚合**：收集多个子 agent 的结果并综合处理
- **降级策略**：A2A 不可用时自动回退到本地工具

### 8.2. A2A 决策逻辑

```python
async def _should_delegate_a2a(self, tool_call: ToolCall) -> bool:
    """判断是否应将任务委托给远程 agent"""
    return tool_call.name.startswith("a2a_")
```

### 8.3. A2A 任务执行流程

在主循环的工具执行阶段增加 A2A 分支：

```python
async def _execute_tool(self, tool_call: ToolCall) -> ToolOutput:
    result = await self.tool_registry.execute(tool_call.name, tool_call.arguments)

    # A2A 工具返回异步任务 ID，需要轮询等待结果
    if tool_call.name.startswith("a2a_") and result.metadata:
        task_id = result.metadata.get("taskId")
        if task_id and result.metadata.get("status") in ("submitted", "working"):
            result = await self._poll_a2a_task(
                tool_call.name.replace("_submit", "_status"), task_id
            )

    return result


async def _poll_a2a_task(self, status_tool_name: str, task_id: str, max_polls: int = 30) -> ToolOutput:
    """轮询 A2A 任务结果"""
    for attempt in range(max_polls):
        result = await self.tool_registry.execute(status_tool_name, {"task_id": task_id})

        status = (result.metadata or {}).get("status", "unknown")
        if status == "completed":
            return result
        elif status == "failed" or status == "canceled":
            return ToolOutput.fail(f"A2A task {task_id} ended with status: {status}")

        await asyncio.sleep(1.0)  # 轮询间隔

    return ToolOutput.fail(f"A2A task {task_id} timed out after {max_polls} polls")
```

### 8.4. A2A 降级策略

当远程 agent 不可用时，Agent Core 自动回退：

```python
async def _execute_tool_with_fallback(self, tool_call: ToolCall) -> ToolOutput:
    result = await self._execute_tool(tool_call)

    if not result.is_ok and tool_call.name.startswith("a2a_"):
        logger.warning(f"A2A tool failed: {tool_call.name}, attempting fallback")
        # 尝试使用本地等效工具或生成提示性回复
        return await self._local_fallback(tool_call)
```

---

## 9. 结构化输出集成

### 9.1. 设计目标

Agent Core 能够利用 LLM 的结构化输出来优化决策流程：
- **解析增强**：使用 `generate_structured()` 获取强类型响应
- **工具选择优化**：通过 schema 约束提高 tool call 准确率
- **中间格式标准化**：统一内部数据交换格式

### 9.2. 结构化 LLM 调用

```python
async def _structured_llm_call(self, messages: list[Message], schema: dict) -> LLMResponse:
    """使用结构化输出约束 LLM 响应"""
    if hasattr(self.llm, "generate_structured"):
        return await self.llm.generate_structured(
            messages=[self._to_openai_msg(m) for m in messages],
            response_format=schema,
        )
    # 降级到普通调用
    return await self.chat(messages)
```

---

## 10. 更新后的设计总结

| 特性 | 实现方式 |
|------|---------|
| **主循环** | while True: LLM → parse → tools → retry（max rounds 保护） |
| **流式输出** | SSE event stream，前端实时渲染 |
| **工具编排** | parallel tool calls → sequential execution → results back to LLM |
| **错误恢复** | 结构化错误回注 conversation_history，LLM 自主决策 |
| **Session 集成** | 每步操作都 append_message()，完整审计链 |
| **A2A 通信** | `a2a_*` 工具自动识别 → 轮询等待 → 降级本地处理 |
| **结构化输出** | `generate_structured()` 优先调用，无支持时降级普通调用 |
