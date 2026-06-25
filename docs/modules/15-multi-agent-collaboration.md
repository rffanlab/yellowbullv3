# 多智能体协作详细设计

## 1. 职责边界

| 职责 | 说明 |
|------|------|
| **Agent 注册** | 定义多个专用 Agent（代码、搜索、分析等），统一管理生命周期 |
| **路由分发** | Supervisor Agent 根据用户意图将任务分发给合适的子 Agent |
| **协作编排** | 支持串行链式调用、并行执行、条件分支等多 Agent 协作模式 |
| **结果聚合** | 汇总多个子 Agent 的输出，生成最终回复 |
| **上下文共享** | 子 Agent 间可共享部分上下文信息 |

---

## 2. Agent 注册表 `multi_agent/registry.py`

```python
"""
多智能体注册与路由。

Agent 类型：
- Supervisor: 总控 Agent，负责意图识别和任务分发
- Specialist: 专用 Agent（代码、搜索、数据分析等）
- Orchestrator: 编排 Agent，协调多个子 Agent 协作
"""

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Callable, Optional

from llm.base import BaseLLM
from tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class AgentType(str, Enum):
    SUPERVISOR = "supervisor"     # 总控路由
    SPECIALIST = "specialist"     # 专用领域
    ORCHESTRATOR = "orchestrator" # 编排协调


@dataclass
class AgentDefinition:
    """Agent 定义"""
    name: str                          # Agent 唯一标识
    agent_type: AgentType              # Agent 类型
    description: str                   # Agent 能力描述（用于路由决策）
    system_prompt: str                 # System prompt
    llm: Optional[BaseLLM] = None     # 专用 LLM（可不同模型）
    tools: list[str] = field(default_factory=list)  # 可用工具名列表
    enabled: bool = True               # 是否启用

    @property
    def routing_keywords(self) -> list[str]:
        """从描述中提取路由关键词"""
        return self.description.lower().split()


class AgentRegistry:
    """多智能体注册表"""

    def __init__(self):
        self._agents: dict[str, AgentDefinition] = {}
        self._supervisor: Optional[AgentDefinition] = None

    def register(self, agent_def: AgentDefinition) -> "AgentRegistry":
        """注册 Agent"""
        if agent_def.agent_type == AgentType.SUPERVISOR:
            self._supervisor = agent_def
        self._agents[agent_def.name] = agent_def
        logger.info(f"Registered agent: {agent_def.name} ({agent_def.agent_type})")
        return self

    def get(self, name: str) -> Optional[AgentDefinition]:
        """获取 Agent 定义"""
        return self._agents.get(name)

    @property
    def supervisor(self) -> Optional[AgentDefinition]:
        return self._supervisor

    @property
    def specialists(self) -> list[AgentDefinition]:
        """获取所有启用的专用 Agent"""
        return [
            a for a in self._agents.values()
            if a.agent_type == AgentType.SPECIALIST and a.enabled
        ]

    def list_for_routing(self) -> str:
        """生成用于路由决策的 Agent 列表文本"""
        lines = []
        for agent in self.specialists:
            lines.append(f"- {agent.name}: {agent.description}")
        return "\n".join(lines)


def get_agent_registry() -> AgentRegistry:
    """获取全局 Agent 注册表"""
    import config.manager as cm
    manager = cm.get_manager()
    if not hasattr(manager, "_agent_registry"):
        manager._agent_registry = _build_default_registry()
    return manager._agent_registry


def _build_default_registry() -> AgentRegistry:
    """构建默认 Agent 注册表"""
    registry = AgentRegistry()

    # Supervisor Agent
    registry.register(AgentDefinition(
        name="supervisor",
        agent_type=AgentType.SUPERVISOR,
        description="总控路由，分析用户意图并分发给合适的子 Agent",
        system_prompt="""你是一个任务调度器。根据用户请求，从以下 Agent 中选择最合适的来处理：

{agent_list}

输出格式（JSON）：
{{"agent": "agent_name", "reason": "...", "task": "具体任务描述"}}

如果无法匹配任何 Agent，使用 "general" Agent 处理。""",
    ))

    # Code Agent
    registry.register(AgentDefinition(
        name="code_agent",
        agent_type=AgentType.SPECIALIST,
        description="代码生成、调试、解释和重构。支持 Python、JavaScript、Go 等语言。",
        system_prompt="""你是一个资深程序员。负责：
1. 编写高质量代码，包含注释和错误处理
2. 调试代码问题，定位 bug 原因
3. 解释代码逻辑，帮助理解复杂实现
4. 重构代码，提升可读性和性能

始终提供可运行的完整代码示例。""",
        tools=["execute_code", "analyze_data"],
    ))

    # Search Agent
    registry.register(AgentDefinition(
        name="search_agent",
        agent_type=AgentType.SPECIALIST,
        description="网络搜索、信息检索、事实核查和最新信息查询。",
        system_prompt="""你是一个研究助手。负责：
1. 使用搜索引擎查找相关信息
2. 综合多个来源，提取关键事实
3. 标注信息来源和时间
4. 对不确定信息明确说明

始终引用可靠来源，区分事实和观点。""",
        tools=["web_search", "fetch_url"],
    ))

    # Data Analysis Agent
    registry.register(AgentDefinition(
        name="data_agent",
        agent_type=AgentType.SPECIALIST,
        description="数据分析、统计计算、可视化描述和报告生成。",
        system_prompt="""你是一个数据分析师。负责：
1. 解析和处理结构化/非结构化数据
2. 执行统计分析（均值、分布、相关性等）
3. 生成数据洞察和建议
4. 用自然语言描述分析结果

使用 Python (pandas/numpy) 进行计算，清晰呈现结论。""",
        tools=["execute_code", "analyze_data"],
    ))

    # General Agent（兜底）
    registry.register(AgentDefinition(
        name="general",
        agent_type=AgentType.SPECIALIST,
        description="通用对话、知识问答和日常任务处理。",
        system_prompt="""你是一个友好的 AI 助手。负责：
1. 回答各类问题，提供有用信息
2. 进行自然流畅的对话
3. 处理日常任务和请求

保持简洁、准确、有帮助。""",
    ))

    return registry
```

---

## 3. Supervisor Router `multi_agent/supervisor.py`

```python
"""
Supervisor Agent —— 多智能体路由中枢。

工作流程：
1. 接收用户消息
2. 调用 LLM 分析意图，选择目标 Agent
3. 将任务分发给子 Agent 执行
4. 聚合结果返回给用户

支持协作模式：
- single: 单个 Agent 处理（默认）
- chain: 串行链式调用 A → B → C
- parallel: 并行执行多个 Agent
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Optional

from agent.core import AgentCore
from llm.base import BaseLLM
from multi_agent.registry import AgentRegistry, AgentDefinition
from session.manager import SessionManager, get_session_manager
from tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class RoutingMode(str, Enum):
    SINGLE = "single"       # 单个 Agent
    CHAIN = "chain"         # 串行链式
    PARALLEL = "parallel"   # 并行执行


@dataclass
class RoutingDecision:
    """路由决策"""
    mode: RoutingMode
    agents: list[str]              # Agent 名称列表
    task_description: str          # 任务描述
    reason: str                    # 选择理由


class SupervisorAgent:
    """Supervisor 总控 Agent"""

    def __init__(
        self,
        llm: BaseLLM,
        agent_registry: AgentRegistry,
        tool_registry: ToolRegistry,
    ):
        self._llm = llm
        self._registry = agent_registry
        self._tools = tool_registry

    async def route(
        self,
        user_message: str,
        session_id: str,
    ) -> RoutingDecision:
        """分析用户意图，做出路由决策"""
        supervisor_def = self._registry.supervisor
        if not supervisor_def:
            # 没有 Supervisor，默认使用 general Agent
            return RoutingDecision(
                mode=RoutingMode.SINGLE,
                agents=["general"],
                task_description=user_message,
                reason="No supervisor configured",
            )

        # 构建路由 prompt
        agent_list = self._registry.list_for_routing()
        routing_prompt = supervisor_def.system_prompt.format(agent_list=agent_list)

        messages = [
            {"role": "system", "content": routing_prompt},
            {"role": "user", "content": user_message},
        ]

        # 调用 LLM 做路由决策
        response = await self._llm.chat(messages, temperature=0.1)
        decision_text = response.content or "{}"

        try:
            decision = json.loads(decision_text)
            agent_name = decision.get("agent", "general")

            # 验证 Agent 是否存在
            if not self._registry.get(agent_name):
                logger.warning(f"Agent '{agent_name}' not found, falling back to 'general'")
                agent_name = "general"

            return RoutingDecision(
                mode=RoutingMode.SINGLE,
                agents=[agent_name],
                task_description=decision.get("task", user_message),
                reason=decision.get("reason", ""),
            )
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse routing decision: {decision_text}")
            return RoutingDecision(
                mode=RoutingMode.SINGLE,
                agents=["general"],
                task_description=user_message,
                reason="Parse error, fallback",
            )

    async def run(
        self,
        user_message: str,
        session_id: str,
        user_id: str = "anonymous",
    ) -> AsyncIterator[dict[str, Any]]:
        """执行多 Agent 协作流程"""
        # Step 1: 路由决策
        decision = await self.route(user_message, session_id)

        yield {
            "event": "routing",
            "data": {
                "mode": decision.mode.value,
                "agents": decision.agents,
                "reason": decision.reason,
            },
        }

        # Step 2: 根据模式执行
        if decision.mode == RoutingMode.SINGLE:
            async for event in self._execute_single(
                decision.agents[0], decision.task_description, session_id, user_id
            ):
                yield event

        elif decision.mode == RoutingMode.CHAIN:
            context = ""
            for agent_name in decision.agents:
                task = f"{decision.task_description}\n\n前序结果:\n{context}" if context else decision.task_description
                async for event in self._execute_single(
                    agent_name, task, session_id, user_id
                ):
                    yield event
                    if event["event"] == "done" and event["data"]["status"] == "completed":
                        # 提取结果作为下一个 Agent 的上下文
                        chunks = [e["data"]["content"] for e in [] if e.get("event") == "chunk"]
                        context += f"\n[{agent_name}]: {''.join(chunks)}"

        elif decision.mode == RoutingMode.PARALLEL:
            results = await self._execute_parallel(
                decision.agents, decision.task_description, session_id, user_id
            )
            async for event in self._aggregate_results(results, decision):
                yield event

    async def _execute_single(
        self,
        agent_name: str,
        task: str,
        session_id: str,
        user_id: str,
    ) -> AsyncIterator[dict[str, Any]]:
        """执行单个 Agent"""
        agent_def = self._registry.get(agent_name)
        if not agent_def:
            yield {"event": "error", "data": {"message": f"Agent '{agent_name}' not found"}}
            return

        # 构建专用 ToolRegistry（仅该 Agent 可用的工具）
        sub_tools = ToolRegistry()
        for tool_name in agent_def.tools:
            tool = self._tools.get(tool_name)
            if tool:
                sub_tools.register(tool)

        # 创建子 Agent Core
        llm = agent_def.llm or self._llm
        core = AgentCore(
            llm=llm,
            tool_registry=sub_tools,
            system_prompt=agent_def.system_prompt,
        )

        yield {"event": "agent_start", "data": {"name": agent_name}}

        async for event in core.run(task, session_id, user_id):
            # 添加 Agent 名称标记
            event["data"]["agent"] = agent_name
            yield event

        yield {"event": "agent_end", "data": {"name": agent_name}}

    async def _execute_parallel(
        self,
        agent_names: list[str],
        task: str,
        session_id: str,
        user_id: str,
    ) -> dict[str, list[dict]]:
        """并行执行多个 Agent"""
        tasks = []
        for agent_name in agent_names:
            tasks.append(
                self._collect_events(agent_name, task, session_id, user_id)
            )

        results = await asyncio.gather(*tasks, return_exceptions=True)

        collected = {}
        for agent_name, result in zip(agent_names, results):
            if isinstance(result, Exception):
                collected[agent_name] = [{"event": "error", "data": {"message": str(result)}}]
            else:
                collected[agent_name] = result

        return collected

    async def _collect_events(
        self, agent_name: str, task: str, session_id: str, user_id: str,
    ) -> list[dict]:
        """收集 Agent 的所有事件"""
        events = []
        async for event in self._execute_single(agent_name, task, session_id, user_id):
            events.append(event)
        return events

    async def _aggregate_results(
        self, results: dict[str, list[dict]], decision: RoutingDecision,
    ) -> AsyncIterator[dict[str, Any]]:
        """聚合并行结果"""
        # 提取每个 Agent 的文本输出
        summaries = {}
        for agent_name, events in results.items():
            chunks = [e["data"].get("content", "") for e in events if e.get("event") == "chunk"]
            summaries[agent_name] = "".join(chunks)

        # 推送各 Agent 结果
        for agent_name, summary in summaries.items():
            yield {
                "event": "agent_result",
                "data": {"name": agent_name, "content": summary},
            }

        # 使用 Supervisor LLM 做最终汇总
        aggregate_prompt = f"""以下是多个专家对同一问题的回答，请综合整理为一个完整的回复：

{chr(10).join(f'### {name}:\\n{summary}' for name, summary in summaries.items())}

请用自然语言整合以上信息，给出最佳答案。保留各来源的独特观点，消除重复内容。"""

        response = await self._llm.chat([
            {"role": "system", "content": "你是一个信息整合专家。"},
            {"role": "user", "content": aggregate_prompt},
        ])

        yield {
            "event": "chunk",
            "data": {"content": response.content or "", "agent": "supervisor"},
        }
        yield {"event": "done", "data": {"status": "completed"}}


def get_supervisor() -> SupervisorAgent:
    """获取全局 Supervisor Agent"""
    import config.manager as cm
    from llm.factory import create_llm

    manager = cm.get_manager()
    if not hasattr(manager, "_supervisor_agent"):
        registry = get_agent_registry()
        tool_reg = ToolRegistry()  # 或从全局获取
        llm_settings = manager.settings.llm or {}
        llm = create_llm(llm_settings)

        manager._supervisor_agent = SupervisorAgent(
            llm=llm,
            agent_registry=registry,
            tool_registry=tool_reg,
        )
    return manager._supervisor_agent
```

---

## 4. API 路由 `api/multi_agent.py`

```python
"""
多智能体协作 API。
"""

from fastapi import APIRouter, Query
from typing import Optional

router = APIRouter(prefix="/multi-agent", tags=["multi-agent"])


@router.post("/run")
async def run_multi_agent(
    message: str,
    session_id: str = Query(...),
    user_id: str = Query("anonymous"),
):
    """通过 Supervisor 执行多 Agent 协作"""
    from multi_agent.supervisor import get_supervisor

    supervisor = get_supervisor()

    async def event_stream():
        async for event in supervisor.run(message, session_id, user_id):
            yield f"event: {event['event']}\ndata: {__import__('json').dumps(event['data'])}\n\n"

    from fastapi.responses import StreamingResponse
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


@router.get("/agents")
async def list_agents():
    """列出所有已注册的 Agent"""
    from multi_agent.registry import get_agent_registry

    registry = get_agent_registry()
    agents = []
    for name, agent in registry._agents.items():
        agents.append({
            "name": name,
            "type": agent.agent_type.value,
            "description": agent.description,
            "tools": agent.tools,
            "enabled": agent.enabled,
        })
    return {"agents": agents}


@router.post("/route")
async def test_routing(message: str, session_id: str = Query("test")):
    """测试路由决策（不执行）"""
    from multi_agent.supervisor import get_supervisor

    supervisor = get_supervisor()
    decision = await supervisor.route(message, session_id)
    return {
        "mode": decision.mode.value,
        "agents": decision.agents,
        "task_description": decision.task_description,
        "reason": decision.reason,
    }


@router.post("/direct/{agent_name}")
async def run_direct_agent(agent_name: str, message: str, session_id: str = Query(...)):
    """直接调用指定 Agent（绕过 Supervisor）"""
    from multi_agent.registry import get_agent_registry
    from agent.core import AgentCore
    from llm.factory import create_llm
    from tools.registry import ToolRegistry

    registry = get_agent_registry()
    agent_def = registry.get(agent_name)

    if not agent_def:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")

    llm = agent_def.llm or create_llm({})
    tools = ToolRegistry()

    core = AgentCore(
        llm=llm,
        tool_registry=tools,
        system_prompt=agent_def.system_prompt,
    )

    async def event_stream():
        async for event in core.run(message, session_id):
            event["data"]["agent"] = agent_name
            yield f"event: {event['event']}\ndata: {__import__('json').dumps(event['data'])}\n\n"

    from fastapi.responses import StreamingResponse
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )
```

---

## 5. YAML 配置 `multi_agent` section

```yaml
# config/settings.yaml (新增)
multi_agent:
  enabled: false              # 是否启用多 Agent 模式（默认使用单 Agent Core）

  supervisor:
    llm_provider: "openai"    # Supervisor 使用的 LLM
    model: "gpt-4o"           # 路由决策需要更强的推理能力
    temperature: 0.1          # 低温度保证稳定路由

  agents:
    code_agent:
      enabled: true
      llm_provider: "openai"
      model: "gpt-4o"         # 代码任务用强模型
      tools: ["execute_code", "analyze_data"]

    search_agent:
      enabled: true
      llm_provider: "openai"
      model: "gpt-4o-mini"    # 搜索可以用轻量模型
      tools: ["web_search", "fetch_url"]

    data_agent:
      enabled: true
      llm_provider: "openai"
      model: "gpt-4o"
      tools: ["execute_code", "analyze_data"]

    general:
      enabled: true
      llm_provider: "openai"
      model: "gpt-4o-mini"    # 通用对话用轻量模型
```

---

## 6. SSE 事件扩展

| Event | Data Schema | 说明 |
|-------|------------|------|
| `routing` | `{"mode": "...", "agents": [...], "reason": "..."}` | 路由决策结果 |
| `agent_start` | `{"name": "str"}` | 子 Agent 开始执行 |
| `agent_end` | `{"name": "str"}` | 子 Agent 执行结束 |
| `agent_result` | `{"name": "str", "content": "..."}` | 并行模式下单个 Agent 结果 |

---

## 7. 协作模式图

```
                    ┌──────────────┐
                    │   User Input │
                    └──────┬───────┘
                           ▼
                  ┌────────────────┐
                  │  Supervisor    │
                  │ (Intent Route) │
                  └──┬─────┬───┬───┘
                     │     │   │
          ┌──────────┘     │   └──────────┐
          ▼                ▼              ▼
   ┌──────────┐    ┌──────────┐    ┌──────────┐
   │ Code     │    │ Search   │    │ Data     │
   │ Agent    │    │ Agent    │    │ Agent    │
   └────┬─────┘    └────┬─────┘    └────┬─────┘
        │               │               │
        └───────────────┼───────────────┘
                        │
                  ┌─────▼──────┐
                  │ Aggregate  │
                  │ & Summarize│
                  └─────┬──────┘
                        ▼
                   User Response
```

---

## 8. 设计总结

| 特性 | 实现方式 |
|------|---------|
| **Agent 注册** | AgentRegistry，支持动态注册/启用/禁用 |
| **智能路由** | Supervisor LLM 分析意图 → JSON 决策 |
| **协作模式** | single / chain / parallel 三种编排 |
| **专用模型** | 每个 Agent 可配置独立 LLM（成本优化） |
| **工具隔离** | 每个 Agent 仅拥有指定工具集 |
| **结果聚合** | Supervisor LLM 汇总多 Agent 输出 |
| **A2A 协议** | `AgentCommunicationBus` 支持请求/响应、委托、协商三种模式 |

---

## 9. Agent-to-Agent 通信协议

### 9.1. 设计目标

定义 Agent 之间通信的标准协议：
- **请求/响应模式**：同步调用另一个 Agent 的能力
- **委托模式**：将子任务完全委托给专家 Agent
- **协商模式**：多个 Agent 就某个决策进行协商

### 9.2. 协议设计 `multiagent/a2a_protocol.py`

```python
from dataclasses import dataclass, field
from enum import Enum


class A2AMessageType(str, Enum):
    REQUEST = "request"           # 请求
    RESPONSE = "response"         # 响应
    DELEGATE = "delegate"         # 委托
    RESULT = "result"             # 结果返回
    NEGOTIATE = "negotiate"       # 协商提议
    COUNTER_PROPOSAL = "counter_proposal"  # 反提议


@dataclass(frozen=True)
class A2AMessage:
    """Agent-to-Agent 消息"""
    message_type: A2AMessageType   # 消息类型
    sender_id: str                 # 发送方 Agent ID
    receiver_id: str              # 接收方 Agent ID
    correlation_id: str           # 关联 ID（用于追踪请求链）
    payload: dict[str, Any]       # 负载数据
    timeout_ms: int = 30000       # 超时时间（毫秒）
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class A2ARequest(A2AMessage):
    """Agent 请求"""
    task_description: str         # 任务描述
    context: dict[str, Any] = field(default_factory=dict)  # 上下文信息


@dataclass(frozen=True)
class A2AResponse(A2AMessage):
    """Agent 响应"""
    success: bool                 # 是否成功
    result_data: Any = None       # 结果数据
    error_message: str | None = None  # 错误信息


@dataclass(frozen=True)
class A2ANegotiation(A2AMessage):
    """协商消息"""
    proposal: dict[str, Any]      # 提议内容
    reasoning: str                # 理由说明
    round_number: int = 1         # 协商轮次


class AgentCommunicationBus:
    """Agent 通信总线"""

    def __init__(self):
        self._agents: dict[str, Any] = {}  # agent_id -> agent instance
        self._pending_requests: dict[str, asyncio.Future] = {}

    def register_agent(self, agent_id: str, agent) -> None:
        """注册 Agent"""
        self._agents[agent_id] = agent

    async def send_request(
        self,
        sender_id: str,
        receiver_id: str,
        task_description: str,
        context: dict[str, Any] | None = None,
        timeout_ms: int = 30000,
    ) -> A2AResponse:
        """发送请求并等待响应"""
        import asyncio
        import uuid

        correlation_id = str(uuid.uuid4())
        request = A2ARequest(
            message_type=A2AMessageType.REQUEST,
            sender_id=sender_id,
            receiver_id=receiver_id,
            correlation_id=correlation_id,
            payload={"context": context or {}},
            timeout_ms=timeout_ms,
            task_description=task_description,
        )

        # 创建 Future 等待响应
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        self._pending_requests[correlation_id] = future

        try:
            await self._agents[receiver_id].handle_a2a_message(request)
            response = await asyncio.wait_for(future, timeout=timeout_ms / 1000)
            return response

        except asyncio.TimeoutError:
            return A2AResponse(
                message_type=A2AMessageType.RESPONSE,
                sender_id=receiver_id,
                receiver_id=sender_id,
                correlation_id=correlation_id,
                payload={},
                success=False,
                error_message=f"Request timed out after {timeout_ms}ms",
            )

    async def deliver_response(self, response: A2AResponse) -> None:
        """投递响应到等待的 Future"""
        future = self._pending_requests.pop(response.correlation_id, None)
        if future and not future.done():
            future.set_result(response)
```

### 9.3. Agent 侧集成 `multiagent/a2a_handler.py`

```python
class A2AHandler:
    """Agent 侧的 A2A 消息处理器"""

    def __init__(self, agent_id: str, comm_bus: AgentCommunicationBus):
        self._agent_id = agent_id
        self._comm_bus = comm_bus

    async def handle_a2a_message(self, message: A2AMessage) -> None:
        """处理来自其他 Agent 的消息"""
        if message.message_type == A2AMessageType.REQUEST:
            result = await self._process_request(message)
            response = A2AResponse(
                message_type=A2AMessageType.RESPONSE,
                sender_id=self._agent_id,
                receiver_id=message.sender_id,
                correlation_id=message.correlation_id,
                payload={"result_data": result},
                success=True,
            )
            await self._comm_bus.deliver_response(response)

        elif message.message_type == A2AMessageType.NEGOTIATE:
            decision = await self._evaluate_proposal(message)
            counter = A2ANegotiation(
                message_type=A2AMessageType.COUNTER_PROPOSAL,
                sender_id=self._agent_id,
                receiver_id=message.sender_id,
                correlation_id=message.correlation_id,
                payload={"decision": decision},
                proposal=decision.get("counter_proposal", {}),
                reasoning=decision.get("reasoning", ""),
                round_number=message.round_number + 1,
            )
            await self._comm_bus.send_message(counter)

    async def _process_request(self, message: A2AMessage) -> Any:
        """处理请求（由子类实现具体逻辑）"""
        raise NotImplementedError

    async def _evaluate_proposal(self, message: A2AMessage) -> dict[str, Any]:
        """评估协商提议（由子类实现）"""
        raise NotImplementedError
```

---

## 10. 更新后的设计总结

| 特性 | 实现方式 |
|------|---------|
| **Agent 注册** | AgentRegistry，支持动态注册/启用/禁用 |
| **智能路由** | Supervisor LLM 分析意图 → JSON 决策 |
| **协作模式** | single / chain / parallel 三种编排 |
| **专用模型** | 每个 Agent 可配置独立 LLM（成本优化） |
| **工具隔离** | 每个 Agent 仅拥有指定工具集 |
| **结果聚合** | Supervisor LLM 汇总多 Agent 输出 |
| **A2A 协议** | `AgentCommunicationBus` 支持请求/响应、委托、协商三种模式 |
