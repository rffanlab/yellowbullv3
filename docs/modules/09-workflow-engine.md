# 自动化工作流引擎详细设计（Workflow Engine）

## 1. 职责边界

| 职责 | 说明 |
|------|------|
| **DAG 编排** | 定义节点间的依赖关系，支持并行、串行、条件分支 |
| **调度执行** | 定时触发（Cron）、事件触发、手动触发 |
| **状态管理** | 工作流实例的生命周期：pending → running → completed/failed/cancelled |
| **重试机制** | 节点失败自动重试，支持指数退避 |
| **人工审批** | 关键节点可插入人工确认环节 |

---

## 2. 工作流定义 `workflow/models.py`

```python
"""
工作流数据模型。

YAML 配置示例：
    workflow:
      name: "daily_report"
      trigger:
        type: "cron"
        schedule: "0 9 * * 1-5"       # 工作日早 9 点
      nodes:
        - id: fetch_data
          type: tool_call
          config:
            tool_name: web_search
            arguments:
              query: "今日行业新闻"
          retry: { max_attempts: 3, backoff: exponential }

        - id: summarize
          type: llm_call
          depends_on: [fetch_data]
          config:
            prompt_template: |
              请总结以下新闻：{{ fetch_data.result }}
            model: gpt-4o-mini

        - id: send_report
          type: tool_call
          depends_on: [summarize]
          config:
            tool_name: send_email
            arguments:
              to: "team@example.com"
              body: "{{ summarize.result }}"
"""

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class WorkflowStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class NodeStatus(str, Enum):
    PENDING = "pending"
    WAITING_DEPS = "waiting_deps"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


class TriggerType(str, Enum):
    MANUAL = "manual"           # 手动触发
    CRON = "cron"               # 定时调度
    EVENT = "event"             # 事件驱动（如：新消息到达）
    WEBHOOK = "webhook"         # HTTP Webhook


class NodeType(str, Enum):
    TOOL_CALL = "tool_call"     # 调用工具
    LLM_CALL = "llm_call"       # 调用 LLM
    CONDITION = "condition"      # 条件分支
    APPROVAL = "approval"        # 人工审批
    SCRIPT = "script"            # 执行 Python 脚本
    WAIT = "wait"               # 等待指定时间


@dataclass
class RetryPolicy:
    """重试策略"""
    max_attempts: int = 3
    backoff: str = "exponential"   # fixed | exponential | linear
    base_delay: float = 1.0        # 秒
    max_delay: float = 60.0


@dataclass
class NodeConfig:
    """节点配置"""
    node_id: str
    node_type: NodeType
    depends_on: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    retry_policy: RetryPolicy | None = None
    timeout: float | None = None   # 秒


@dataclass
class TriggerConfig:
    """触发器配置"""
    trigger_type: TriggerType
    schedule: str | None = None     # Cron expression（定时触发）
    event_name: str | None = None   # 事件名称（事件触发）
    webhook_url: str | None = None  # Webhook URL


@dataclass
class WorkflowDefinition:
    """工作流定义"""
    workflow_id: str
    name: str
    description: str = ""
    trigger: TriggerConfig | None = None
    nodes: list[NodeConfig] = field(default_factory=list)
    variables: dict[str, Any] = field(default_factory=dict)  # 全局变量


@dataclass
class NodeExecutionResult:
    """节点执行结果"""
    node_id: str
    status: NodeStatus
    output: Any = None
    error: str | None = None
    started_at: float = 0.0
    completed_at: float = 0.0
    attempt: int = 1


@dataclass
class WorkflowInstance:
    """工作流实例（一次执行）"""
    instance_id: str
    workflow_id: str
    status: WorkflowStatus = WorkflowStatus.PENDING
    node_results: dict[str, NodeExecutionResult] = field(default_factory=dict)
    variables: dict[str, Any] = field(default_factory=dict)  # 运行时变量
    created_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    completed_at: float = 0.0
    error: str | None = None


class WorkflowDefinitionParser:
    """从 YAML/Dict 解析工作流定义"""

    @staticmethod
    def from_dict(data: dict[str, Any]) -> WorkflowDefinition:
        trigger_data = data.get("trigger", {})
        trigger = TriggerConfig(
            trigger_type=TriggerType(trigger_data.get("type", "manual")),
            schedule=trigger_data.get("schedule"),
            event_name=trigger_data.get("event_name"),
            webhook_url=trigger_data.get("webhook_url"),
        ) if trigger_data else None

        nodes = []
        for nd in data.get("nodes", []):
            retry_data = nd.get("retry")
            retry_policy = RetryPolicy(**retry_data) if retry_data else None

            nodes.append(NodeConfig(
                node_id=nd["id"],
                node_type=NodeType(nd["type"]),
                depends_on=nd.get("depends_on", []),
                config=nd.get("config", {}),
                retry_policy=retry_policy,
                timeout=nd.get("timeout"),
            ))

        return WorkflowDefinition(
            workflow_id=data.get("workflow_id", uuid.uuid4().hex[:12]),
            name=data["name"],
            description=data.get("description", ""),
            trigger=trigger,
            nodes=nodes,
            variables=data.get("variables", {}),
        )

    @staticmethod
    def from_yaml(yaml_content: str) -> WorkflowDefinition:
        import yaml
        data = yaml.safe_load(yaml_content)
        return WorkflowDefinitionParser.from_dict(data)
```

---

## 3. DAG 执行引擎 `workflow/engine.py`

```python
"""
DAG 工作流执行引擎。

核心逻辑：
1. 拓扑排序确定执行顺序
2. 并行执行无依赖的节点
3. 条件分支根据上游结果决定路径
4. 失败重试 + 指数退避
"""

import asyncio
import logging
import time
from collections import deque
from typing import Any, Protocol

from workflow.models import (
    WorkflowDefinition, WorkflowInstance, WorkflowStatus,
    NodeConfig, NodeType, NodeStatus, NodeExecutionResult, RetryPolicy,
)

logger = logging.getLogger(__name__)


class NodeExecutor(Protocol):
    """节点执行器协议"""

    async def execute(self, node: NodeConfig, variables: dict[str, Any]) -> Any: ...


class TemplateRenderer:
    """简单的模板渲染（支持 {{ variable }} 语法）"""

    @staticmethod
    def render(template: str, context: dict[str, Any]) -> str:
        """替换模板中的变量引用"""
        import re

        def replacer(match):
            var_path = match.group(1).strip()
            parts = var_path.split(".")
            value = context
            for part in parts:
                if isinstance(value, dict):
                    value = value.get(part, "")
                else:
                    value = getattr(value, part, "") if hasattr(value, part) else ""
            return str(value)

        return re.sub(r"\{\{\s*([^}]+)\s*\}\}", replacer, template)

    @staticmethod
    def render_dict(data: Any, context: dict[str, Any]) -> Any:
        """递归渲染字典/列表中的模板"""
        if isinstance(data, str):
            return TemplateRenderer.render(data, context)
        elif isinstance(data, dict):
            return {k: TemplateRenderer.render_dict(v, context) for k, v in data.items()}
        elif isinstance(data, list):
            return [TemplateRenderer.render_dict(item, context) for item in data]
        return data


class WorkflowEngine:
    """
    DAG 工作流执行引擎。

    Usage:
        engine = WorkflowEngine(node_executor=DefaultNodeExecutor())
        instance = await engine.execute(workflow_def, input_variables={"query": "..."})
    """

    def __init__(self, node_executor: NodeExecutor):
        self._executor = node_executor
        self._renderer = TemplateRenderer()

    async def execute(
        self,
        workflow: WorkflowDefinition,
        input_variables: dict[str, Any] | None = None,
    ) -> WorkflowInstance:
        """执行工作流"""
        instance = WorkflowInstance(
            instance_id=f"inst_{workflow.workflow_id}_{int(time.time())}",
            workflow_id=workflow.workflow_id,
            status=WorkflowStatus.RUNNING,
            variables={**workflow.variables, **(input_variables or {})},
            started_at=time.time(),
        )

        logger.info(f"Starting workflow '{workflow.name}' instance {instance.instance_id}")

        try:
            await self._execute_dag(workflow, instance)

            # 检查是否有节点失败
            has_failed = any(
                r.status == NodeStatus.FAILED
                for r in instance.node_results.values()
            )

            if has_failed:
                instance.status = WorkflowStatus.FAILED
                instance.error = "One or more nodes failed"
            else:
                instance.status = WorkflowStatus.COMPLETED

        except Exception as e:
            instance.status = WorkflowStatus.FAILED
            instance.error = str(e)
            logger.error(f"Workflow '{workflow.name}' failed: {e}")

        instance.completed_at = time.time()
        elapsed = instance.completed_at - instance.started_at
        logger.info(
            f"Workflow '{workflow.name}' finished with status={instance.status}, "
            f"elapsed={elapsed:.1f}s"
        )
        return instance

    async def _execute_dag(
        self, workflow: WorkflowDefinition, instance: WorkflowInstance
    ):
        """基于拓扑排序的 DAG 执行"""
        # 构建依赖图
        graph = {node.node_id: set(node.depends_on) for node in workflow.nodes}
        node_map = {node.node_id: node for node in workflow.nodes}

        # 拓扑排序（Kahn's algorithm）
        execution_order = self._topological_sort(graph)
        if execution_order is None:
            raise ValueError("Workflow contains circular dependency")

        logger.debug(f"Execution order: {execution_order}")

        # 按层执行（同一层的节点可并行）
        executed: set[str] = set()

        for layer in self._get_execution_layers(graph, execution_order):
            # 检查条件分支：跳过被上游条件排除的节点
            active_nodes = []
            for node_id in layer:
                node = node_map[node_id]
                if self._should_skip(node, instance):
                    instance.node_results[node_id] = NodeExecutionResult(
                        node_id=node_id, status=NodeStatus.SKIPPED,
                        started_at=time.time(), completed_at=time.time(),
                    )
                    executed.add(node_id)
                    continue
                active_nodes.append(node_id)

            # 并行执行同一层的节点
            if active_nodes:
                tasks = [
                    self._execute_node(node_map[nid], instance)
                    for nid in active_nodes
                ]
                await asyncio.gather(*tasks, return_exceptions=True)

            executed.update(active_nodes)

    def _topological_sort(self, graph: dict[str, set[str]]) -> list[str] | None:
        """Kahn's 拓扑排序，返回执行顺序；有环则返回 None"""
        in_degree = {node: 0 for node in graph}
        for node, deps in graph.items():
            for dep in deps:
                if dep in in_degree:
                    in_degree[node] += 1

        queue = deque([n for n, d in in_degree.items() if d == 0])
        order = []

        while queue:
            node = queue.popleft()
            order.append(node)
            for other, deps in graph.items():
                if node in deps:
                    in_degree[other] -= 1
                    if in_degree[other] == 0:
                        queue.append(other)

        return order if len(order) == len(graph) else None

    def _get_execution_layers(
        self, graph: dict[str, set[str]], order: list[str]
    ) -> list[list[str]]:
        """将拓扑排序分组为执行层（同一层的节点可并行）"""
        layers = []
        remaining = set(order)
        executed: set[str] = set()

        while remaining:
            layer = []
            for node in order:
                if node not in remaining:
                    continue
                deps = graph[node]
                if deps.issubset(executed):
                    layer.append(node)

            if not layer:
                break  # safety check

            layers.append(layer)
            executed.update(layer)
            remaining -= set(layer)

        return layers

    def _should_skip(self, node: NodeConfig, instance: WorkflowInstance) -> bool:
        """检查节点是否应被跳过（条件分支逻辑）"""
        for dep_id in node.depends_on:
            if dep_id not in instance.node_results:
                continue
            dep_result = instance.node_results[dep_id]
            # 如果依赖节点被跳过，当前节点也跳过
            if dep_result.status == NodeStatus.SKIPPED:
                return True

        # 检查条件配置
        condition = node.config.get("condition")
        if not condition:
            return False

        # 评估条件表达式
        expr = condition.get("expression", "")
        context = self._build_context(instance)
        try:
            result = self._renderer.render(str(expr), context)
            return result.lower() in ("false", "0", "no", "skip")
        except Exception:
            return False

    async def _execute_node(
        self, node: NodeConfig, instance: WorkflowInstance
    ):
        """执行单个节点（含重试逻辑）"""
        start_time = time.time()
        max_attempts = (node.retry_policy or RetryPolicy()).max_attempts
        backoff_type = (node.retry_policy or RetryPolicy()).backoff
        base_delay = (node.retry_policy or RetryPolicy()).base_delay

        last_error = None

        for attempt in range(1, max_attempts + 1):
            try:
                # 渲染配置中的模板变量
                context = self._build_context(instance)
                rendered_config = self._renderer.render_dict(node.config, context)

                logger.debug(
                    f"Executing node '{node.node_id}' (attempt {attempt}/{max_attempts})"
                )

                result = await asyncio.wait_for(
                    self._executor.execute(node, rendered_config),
                    timeout=node.timeout or 300.0,
                )

                instance.node_results[node.node_id] = NodeExecutionResult(
                    node_id=node.node_id,
                    status=NodeStatus.COMPLETED,
                    output=result,
                    started_at=start_time,
                    completed_at=time.time(),
                    attempt=attempt,
                )
                return

            except asyncio.TimeoutError:
                last_error = f"Node '{node.node_id}' timed out after {node.timeout}s"
                logger.warning(last_error)

            except Exception as e:
                last_error = str(e)
                logger.warning(
                    f"Node '{node.node_id}' attempt {attempt} failed: {e}"
                )

            # 重试前等待
            if attempt < max_attempts:
                delay = self._calculate_delay(backoff_type, base_delay, attempt)
                logger.info(f"Retrying node '{node.node_id}' in {delay:.1f}s...")
                await asyncio.sleep(delay)

        # 所有重试失败
        instance.node_results[node.node_id] = NodeExecutionResult(
            node_id=node.node_id,
            status=NodeStatus.FAILED,
            error=last_error,
            started_at=start_time,
            completed_at=time.time(),
            attempt=max_attempts,
        )

    def _calculate_delay(self, backoff_type: str, base_delay: float, attempt: int) -> float:
        """计算重试延迟"""
        if backoff_type == "exponential":
            return min(base_delay * (2 ** (attempt - 1)), 60.0)
        elif backoff_type == "linear":
            return min(base_delay * attempt, 60.0)
        else:  # fixed
            return base_delay

    def _build_context(self, instance: WorkflowInstance) -> dict[str, Any]:
        """构建变量上下文（上游节点结果 + 全局变量）"""
        context = dict(instance.variables)
        for node_id, result in instance.node_results.items():
            if result.status == NodeStatus.COMPLETED and result.output is not None:
                context[node_id] = {"result": result.output, "status": "completed"}
            elif result.status == NodeStatus.FAILED:
                context[node_id] = {"result": None, "status": "failed", "error": result.error}
        return context

    async def cancel(self, instance: WorkflowInstance) -> bool:
        """取消正在运行的工作流"""
        if instance.status in (WorkflowStatus.RUNNING,):
            instance.status = WorkflowStatus.CANCELLED
            instance.completed_at = time.time()
            logger.info(f"Cancelled workflow instance {instance.instance_id}")
            return True
        return False
```

---

## 4. 默认节点执行器 `workflow/executors.py`

```python
"""
内置节点执行器。

支持：tool_call, llm_call, condition, approval, script, wait
"""

import asyncio
import logging
from typing import Any

from workflow.models import NodeConfig, NodeType

logger = logging.getLogger(__name__)


class DefaultNodeExecutor:
    """默认节点执行器，注册各类节点的执行逻辑"""

    def __init__(self):
        self._handlers: dict[NodeType, Any] = {
            NodeType.TOOL_CALL: self._execute_tool_call,
            NodeType.LLM_CALL: self._execute_llm_call,
            NodeType.CONDITION: self._execute_condition,
            NodeType.APPROVAL: self._execute_approval,
            NodeType.SCRIPT: self._execute_script,
            NodeType.WAIT: self._execute_wait,
        }

    async def execute(self, node: NodeConfig, variables: dict[str, Any]) -> Any:
        handler = self._handlers.get(node.node_type)
        if not handler:
            raise ValueError(f"Unknown node type: {node.node_type}")
        return await handler(node, variables)

    async def _execute_tool_call(self, node: NodeConfig, variables: dict[str, Any]) -> Any:
        """执行工具调用"""
        tool_name = node.config.get("tool_name")
        arguments = node.config.get("arguments", {})

        # TODO: 从 ToolRegistry 获取工具实例
        logger.info(f"Tool call: {tool_name} with args={arguments}")
        return {"tool": tool_name, "output": f"Executed {tool_name}"}

    async def _execute_llm_call(self, node: NodeConfig, variables: dict[str, Any]) -> Any:
        """执行 LLM 调用"""
        prompt = node.config.get("prompt_template", "")
        model = node.config.get("model", "gpt-4o-mini")

        # TODO: 从 LLMProviderRegistry 获取模型实例
        logger.info(f"LLM call: model={model}, prompt='{prompt[:50]}...'")
        return {"model": model, "output": f"LLM response for '{prompt[:30]}'"}

    async def _execute_condition(self, node: NodeConfig, variables: dict[str, Any]) -> Any:
        """条件分支节点"""
        expression = node.config.get("expression", "true")
        return {"condition_met": expression.lower() not in ("false", "0")}

    async def _execute_approval(self, node: NodeConfig, variables: dict[str, Any]) -> Any:
        """人工审批节点（阻塞等待）"""
        approvers = node.config.get("approvers", [])
        timeout_hours = node.config.get("timeout_hours", 24)

        logger.info(f"Approval required from {approvers}, timeout={timeout_hours}h")
        # TODO: 实现审批队列 + WebSocket 通知
        return {"approved": True, "approver": approvers[0] if approvers else None}

    async def _execute_script(self, node: NodeConfig, variables: dict[str, Any]) -> Any:
        """执行 Python 脚本"""
        script = node.config.get("script", "")
        # TODO: 在沙箱中执行脚本
        logger.info(f"Executing script (length={len(script)})")
        return {"output": "Script executed"}

    async def _execute_wait(self, node: NodeConfig, variables: dict[str, Any]) -> Any:
        """等待指定时间"""
        seconds = float(node.config.get("seconds", 0))
        logger.info(f"Waiting {seconds}s...")
        await asyncio.sleep(seconds)
        return {"waited_seconds": seconds}


class ToolCallExecutor(DefaultNodeExecutor):
    """带真实工具调用的执行器"""

    def __init__(self, tool_registry: dict[str, Any]):
        super().__init__()
        self._tools = tool_registry

    async def _execute_tool_call(self, node: NodeConfig, variables: dict[str, Any]) -> Any:
        tool_name = node.config.get("tool_name")
        arguments = node.config.get("arguments", {})

        if tool_name not in self._tools:
            raise ValueError(f"Tool '{tool_name}' not found in registry")

        tool = self._tools[tool_name]
        result = await tool.execute(arguments)
        return {"tool": tool_name, "output": result.content}


class LLMCallExecutor(DefaultNodeExecutor):
    """带真实 LLM 调用的执行器"""

    def __init__(self, llm_provider: Any):
        super().__init__()
        self._llm = llm_provider

    async def _execute_llm_call(self, node: NodeConfig, variables: dict[str, Any]) -> Any:
        prompt = node.config.get("prompt_template", "")
        model = node.config.get("model")
        system_prompt = node.config.get("system_prompt", "")

        response = await self._llm.chat(
            system_prompt=system_prompt or None,
            user_message=prompt,
            max_tokens=node.config.get("max_tokens", 2000),
            temperature=node.config.get("temperature", 0.7),
        )
        return {"model": model or self._llm.model_name, "output": response.content}
```

---

## 5. 调度器 `workflow/scheduler.py`

```python
"""
工作流调度器。

支持：Cron 定时触发、事件驱动触发、Webhook 触发。
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Callable

from workflow.models import WorkflowDefinition, TriggerConfig, TriggerType
from workflow.engine import WorkflowEngine, WorkflowInstance

logger = logging.getLogger(__name__)


class CronParser:
    """简化版 Cron 解析器（支持分钟、小时、日、月、星期）"""

    @staticmethod
    def next_run_time(cron_expr: str, after: datetime | None = None) -> datetime:
        """计算下次执行时间"""
        from croniter import croniter
        base = after or datetime.now()
        iter_ = croniter(cron_expr, base)
        return iter_.get_next(datetime)


class WorkflowScheduler:
    """
    工作流调度器。

    Usage:
        scheduler = WorkflowScheduler(engine=engine)
        scheduler.register(workflow_def)
        await scheduler.start()   # 启动后台调度循环
    """

    def __init__(self, engine: WorkflowEngine):
        self._engine = engine
        self._workflows: dict[str, WorkflowDefinition] = {}
        self._event_handlers: dict[str, list[WorkflowDefinition]] = {}
        self._running = False
        self._task: asyncio.Task | None = None

    def register(self, workflow: WorkflowDefinition):
        """注册工作流"""
        self._workflows[workflow.workflow_id] = workflow

        if workflow.trigger and workflow.trigger.trigger_type == TriggerType.EVENT:
            event_name = workflow.trigger.event_name or "default"
            if event_name not in self._event_handlers:
                self._event_handlers[event_name] = []
            self._event_handlers[event_name].append(workflow)

    async def start(self):
        """启动调度器"""
        self._running = True
        self._task = asyncio.create_task(self._scheduler_loop())
        logger.info("Workflow scheduler started")

    async def stop(self):
        """停止调度器"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Workflow scheduler stopped")

    async def trigger_event(self, event_name: str, payload: dict[str, Any] | None = None):
        """触发事件驱动的工作流"""
        workflows = self._event_handlers.get(event_name, [])
        tasks = []
        for wf in workflows:
            task = asyncio.create_task(
                self._engine.execute(wf, input_variables=payload)
            )
            tasks.append(task)

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Event-triggered workflow '{workflows[i].name}' failed: {result}")

    async def _scheduler_loop(self):
        """后台调度循环"""
        while self._running:
            now = datetime.now()
            for wf_id, workflow in self._workflows.items():
                if not workflow.trigger:
                    continue

                if workflow.trigger.trigger_type == TriggerType.CRON:
                    next_run = CronParser.next_run_time(workflow.trigger.schedule or "")
                    if next_run <= now:
                        logger.info(f"Cron trigger: executing '{workflow.name}'")
                        asyncio.create_task(self._engine.execute(workflow))

            await asyncio.sleep(10)  # 每 10 秒检查一次
```

---

## 6. YAML 配置 `config/workflow.yaml`

```yaml
workflow:
  enabled: true

  scheduler:
    poll_interval_seconds: 10     # Cron 检查间隔
    max_concurrent_instances: 5   # 最大并发实例数

  storage:
    type: "sqlite"                # json_file | sqlite
    db_path: "./data/workflows.db"

  default_retry:
    max_attempts: 3
    backoff: exponential
    base_delay: 1.0
    max_delay: 60.0

  templates:
    - name: "daily_report"
      trigger:
        type: cron
        schedule: "0 9 * * 1-5"
      nodes:
        - id: fetch_news
          type: tool_call
          config: { tool_name: web_search }

        - id: summarize
          type: llm_call
          depends_on: [fetch_news]
          config:
            prompt_template: "总结以下新闻：{{ fetch_news.result }}"

        - id: send_email
          type: tool_call
          depends_on: [summarize]
          config: { tool_name: send_email }
```

---

## 7. 架构总览

```
                    ┌─────────────────────┐
                    │    Workflow YAML     │
                    │   (定义 + 配置)       │
                    └──────────┬──────────┘
                               │ parse
                               ▼
              ┌──────────────────────────────────┐
              │      WorkflowDefinition           │
              │  name, trigger, nodes[DAG]        │
              └──────────┬───────────────────────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼               ▼
   ┌────────────┐ ┌────────────┐ ┌────────────┐
   │ Cron       │ │ Event      │ │ Webhook    │
   │ Scheduler  │ │ Trigger    │ │ Endpoint   │
   └─────┬──────┘ └─────┬──────┘ └─────┬──────┘
         │              │               │
         ▼              ▼               ▼
              ┌─────────────────────┐
              │    WorkflowEngine   │
              │                     │
              │  Topological Sort   │
              │  Layer-by-Layer     │
              │  Parallel Execution │
              │  Retry + Backoff    │
              └──────────┬──────────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼               ▼
   ┌────────────┐ ┌────────────┐ ┌────────────┐
   │ Tool Call  │ │ LLM Call   │ │ Condition  │
   │ Executor   │ │ Executor   │ │ / Approval │
   └────────────┘ └────────────┘ └────────────┘
```

---

## 8. 设计总结

| 特性 | 实现方式 |
|------|---------|
| **DAG 编排** | YAML 定义 → 拓扑排序 → 分层并行执行 |
| **多触发器** | Cron / Event / Webhook / Manual，统一调度入口 |
| **重试机制** | 指数退避 / 线性退避 / 固定间隔，可配置 max_attempts |
| **模板变量** | `{{ node_id.result }}` 语法，节点间数据传递 |
| **条件分支** | 表达式评估，动态跳过不满足条件的节点 |
| **人工审批** | 阻塞等待 + WebSocket 通知（待实现） |
