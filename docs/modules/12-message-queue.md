# 消息队列与异步任务（Message Queue & Async Tasks）详细设计

## 1. 职责边界

| 场景 | 说明 | 实现方式 |
|------|------|---------|
| **长时工具执行** | 超过 HTTP timeout 的工具调用（如批量数据处理） | async task + polling |
| **后台索引构建** | RAG 文档解析和向量化，不阻塞对话 | asyncio queue |
| **事件驱动任务** | 定时清理、定期摘要生成、健康检查 | APScheduler |
| **跨服务通信** | 多实例部署时的任务分发（可选） | Redis Pub/Sub / RabbitMQ |

---

## 2. 异步任务管理器 `tasks/task_manager.py`

```python
"""
Async task manager。

核心能力：
- 提交后台任务，返回 task_id
- 轮询任务状态和结果
- 任务超时控制
- 失败重试机制
"""

import asyncio
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AsyncTask:
    """异步任务实体"""

    def __init__(self, task_id: str, name: str, coroutine_fn: Callable):
        self.task_id = task_id
        self.name = name
        self.coroutine_fn = coroutine_fn
        self.status = TaskStatus.PENDING
        self.result: Optional[Any] = None
        self.error: Optional[str] = None
        self.created_at = datetime.now(timezone.utc)
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


class TaskManager:
    """异步任务管理器"""

    def __init__(self, max_concurrent: int = 10, default_timeout: float = 300):
        self._tasks: dict[str, AsyncTask] = {}
        self._max_concurrent = max_concurrent
        self._default_timeout = default_timeout
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def submit(
        self, name: str, coroutine_fn: Callable, timeout: float = None,
    ) -> str:
        """提交异步任务，返回 task_id"""
        task_id = str(uuid.uuid4())
        task = AsyncTask(task_id, name, coroutine_fn)
        self._tasks[task_id] = task

        # 在后台启动任务（不阻塞）
        asyncio.create_task(self._execute(task, timeout or self._default_timeout))
        return task_id

    async def _execute(self, task: AsyncTask, timeout: float):
        """执行异步任务"""
        async with self._semaphore:
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now(timezone.utc)

            try:
                task.result = await asyncio.wait_for(
                    task.coroutine_fn(), timeout=timeout
                )
                task.status = TaskStatus.COMPLETED
            except asyncio.TimeoutError:
                task.error = f"Task timed out after {timeout}s"
                task.status = TaskStatus.FAILED
            except Exception as e:
                task.error = str(e)
                task.status = TaskStatus.FAILED
            finally:
                task.completed_at = datetime.now(timezone.utc)

    def get_status(self, task_id: str) -> Optional[dict]:
        """获取任务状态"""
        task = self._tasks.get(task_id)
        if not task:
            return None

        return {
            "task_id": task.task_id,
            "name": task.name,
            "status": task.status.value,
            "result": task.result,
            "error": task.error,
            "created_at": task.created_at.isoformat(),
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "duration_seconds": task.duration_seconds,
        }

    async def cancel(self, task_id: str) -> bool:
        """取消任务（仅能取消 pending 状态）"""
        task = self._tasks.get(task_id)
        if not task or task.status != TaskStatus.PENDING:
            return False

        task.status = TaskStatus.CANCELLED
        task.completed_at = datetime.now(timezone.utc)
        return True

    def cleanup_completed(self, max_age_hours: int = 24):
        """清理过期任务"""
        from datetime import timedelta
        cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)

        to_delete = [
            tid for tid, t in self._tasks.items()
            if t.completed_at and t.completed_at < cutoff
        ]
        for tid in to_delete:
            del self._tasks[tid]
        return len(to_delete)


def get_task_manager() -> TaskManager:
    """获取全局任务管理器"""
    import config.manager as cm
    manager = cm.get_manager()
    if not hasattr(manager, "_task_manager"):
        settings = (manager.settings.tasks or {})
        manager._task_manager = TaskManager(
            max_concurrent=settings.get("max_concurrent", 10),
            default_timeout=settings.get("default_timeout", 300),
        )
    return manager._task_manager
```

---

## 3. 定时任务调度器 `tasks/scheduler.py`

```python
"""
Scheduled task scheduler。

使用 APScheduler 管理定时任务：
- 定期清理过期会话
- 定期清理过期记忆
- 定期生成对话摘要
- 健康检查
"""

from apscheduler.schedulers.asyncio import AsyncIOScheduler


class TaskScheduler:
    """定时任务调度器"""

    def __init__(self):
        self._scheduler = AsyncIOScheduler()

    async def start(self):
        """启动调度器"""
        if not self._scheduler.running:
            self._scheduler.start()

    async def stop(self):
        """停止调度器"""
        if self._scheduler.running:
            await self._scheduler.shutdown(wait=False)

    def add_job(
        self, func, trigger: str = "interval", name: str = None, **kwargs
    ):
        """添加定时任务"""
        self._scheduler.add_job(
            func,
            trigger=trigger,  # interval | cron
            id=name or func.__name__,
            name=name,
            replace_existing=True,
            **kwargs,
        )

    def remove_job(self, job_id: str):
        """移除定时任务"""
        self._scheduler.remove_job(job_id)


# 预定义的定时任务
async def cleanup_expired_sessions():
    """清理过期会话（每天凌晨 2 点）"""
    from session.manager import get_session_manager

    sm = get_session_manager()
    cleaned = await sm.cleanup_expired(ttl_hours=168)  # 7天
    __import__("logging").getLogger("scheduler").info(
        f"Cleaned up {cleaned} expired sessions"
    )


async def cleanup_task_results():
    """清理过期任务结果（每小时）"""
    from tasks.task_manager import get_task_manager

    tm = get_task_manager()
    cleaned = tm.cleanup_completed(max_age_hours=24)
    __import__("logging").getLogger("scheduler").info(
        f"Cleaned up {cleaned} completed tasks"
    )


async def cleanup_expired_memories():
    """清理过期记忆（每天凌晨 3 点）"""
    from memory.long_term import get_long_term_memory

    ltm = get_long_term_memory()
    await ltm.cleanup_expired()


def setup_default_jobs(scheduler: TaskScheduler):
    """注册默认定时任务"""
    scheduler.add_job(
        cleanup_expired_sessions,
        trigger="cron",
        name="cleanup_sessions",
        hour=2, minute=0,  # 每天凌晨 2 点
    )

    scheduler.add_job(
        cleanup_task_results,
        trigger="interval",
        name="cleanup_tasks",
        hours=1,  # 每小时
    )

    scheduler.add_job(
        cleanup_expired_memories,
        trigger="cron",
        name="cleanup_memories",
        hour=3, minute=0,  # 每天凌晨 3 点
    )


def get_scheduler() -> TaskScheduler:
    """获取全局调度器"""
    import config.manager as cm
    manager = cm.get_manager()
    if not hasattr(manager, "_scheduler"):
        scheduler = TaskScheduler()
        setup_default_jobs(scheduler)
        manager._scheduler = scheduler
    return manager._scheduler
```

---

## 4. Agent Core 集成 `tasks/integration.py`

```python
"""
异步任务与 Agent Core 的集成。

长时工具执行流程：
1. LLM 返回 tool call → 检测到工具标记为 long_running
2. Agent Core 提交后台任务，返回 task_id 给用户
3. SSE stream 推送 "task_started" event
4. 用户通过 GET /tasks/{task_id} 轮询状态
5. 任务完成后，SSE 推送 "task_completed" event + 结果
"""


class LongRunningToolHandler:
    """长时工具处理器"""

    def __init__(self, task_manager: "TaskManager"):
        self._task_manager = task_manager

    async def execute(
        self, tool_name: str, args: dict, timeout: float = 300,
    ) -> dict:
        """提交长时工具执行任务"""
        from tools.registry import get_tool_registry

        registry = get_tool_registry()
        tool = registry.get(tool_name)

        async def _run():
            return await tool.execute(**args)

        task_id = await self._task_manager.submit(
            name=f"tool:{tool_name}",
            coroutine_fn=_run,
            timeout=timeout,
        )

        return {
            "type": "long_running_task",
            "task_id": task_id,
            "tool_name": tool_name,
            "message": f"工具 '{tool_name}' 已提交后台执行，请使用 task_id 查询进度。",
        }


class RAGIndexTask:
    """RAG 文档索引构建任务"""

    def __init__(self, task_manager: "TaskManager"):
        self._task_manager = task_manager

    async def index_document(self, file_path: str, collection: str = "default") -> dict:
        """提交文档索引任务"""
        from rag.chunker import parse_document, SlidingWindowChunker
        from rag.vector_store import get_vector_store

        async def _run():
            text = parse_document(file_path)
            chunker = SlidingWindowChunker()
            chunks = chunker.chunk(text, source=file_path)
            vs = get_vector_store()
            await vs.add_documents(chunks, collection=collection)
            return {"indexed_chunks": len(chunks), "source": file_path}

        task_id = await self._task_manager.submit(
            name=f"rag_index:{file_path}",
            coroutine_fn=_run,
            timeout=600,  # 10分钟超时
        )

        return {
            "type": "index_task",
            "task_id": task_id,
            "source": file_path,
            "message": f"文档 '{file_path}' 已开始索引，请使用 task_id 查询进度。",
        }
```

---

## 5. API 路由 `api/tasks.py`

```python
"""
异步任务相关 API 端点。
"""

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.post("/submit")
async def submit_task(name: str, coroutine_type: str, params: dict):
    """提交异步任务（内部使用，通常由 Agent Core 触发）"""
    from tasks.task_manager import get_task_manager

    tm = get_task_manager()
    task_id = await tm.submit(name=name, coroutine_fn=lambda: None)
    return {"task_id": task_id}


@router.get("/{task_id}")
async def get_task_status(task_id: str):
    """查询任务状态"""
    from tasks.task_manager import get_task_manager

    tm = get_task_manager()
    status = tm.get_status(task_id)

    if not status:
        raise HTTPException(status_code=404, detail="Task not found")

    return status


@router.delete("/{task_id}")
async def cancel_task(task_id: str):
    """取消任务"""
    from tasks.task_manager import get_task_manager

    tm = get_task_manager()
    cancelled = await tm.cancel(task_id)

    if not cancelled:
        raise HTTPException(status_code=400, detail="Task cannot be cancelled")

    return {"status": "cancelled"}


@router.get("/")
async def list_tasks(status_filter: str = None):
    """列出所有任务"""
    from tasks.task_manager import get_task_manager

    tm = get_task_manager()
    tasks = []
    for tid in tm._tasks:
        s = tm.get_status(tid)
        if status_filter and s["status"] != status_filter:
            continue
        tasks.append(s)
    return {"tasks": tasks, "total": len(tasks)}
```

---

## 6. SSE 事件推送 `tasks/sse_events.py`

```python
"""
异步任务相关的 SSE 事件。

在 Agent Core 的 SSE stream 中注入任务状态更新：
- task_started: 后台任务已启动
- task_progress: 进度更新（可选）
- task_completed: 任务完成，附带结果
- task_failed: 任务失败，附带错误信息
"""


def format_task_event(event_type: str, task_id: str, data: dict = None) -> str:
    """格式化 SSE 事件"""
    import json

    event_data = {
        "type": event_type,
        "task_id": task_id,
        "data": data or {},
    }

    return f"event: task_update\ndata: {json.dumps(event_data)}\n\n"


# 示例：Agent Core 中的使用方式
async def handle_long_running_tool(tool_name: str, args: dict):
    handler = LongRunningToolHandler(get_task_manager())
    result = await handler.execute(tool_name, args)

    # SSE stream 推送任务启动事件
    sse_event = format_task_event("task_started", result["task_id"])
    yield sse_event

    # 等待任务完成（可选：也可以让用户轮询）
    tm = get_task_manager()
    while True:
        status = tm.get_status(result["task_id"])
        if status["status"] in ("completed", "failed"):
            event_type = "task_completed" if status["status"] == "completed" else "task_failed"
            sse_event = format_task_event(event_type, result["task_id"], {
                "result": status.get("result"),
                "error": status.get("error"),
            })
            yield sse_event
            break

        await asyncio.sleep(1)  # 轮询间隔
```

---

## 7. YAML 配置 `tasks` section

```yaml
# config/settings.yaml (新增)
tasks:
  max_concurrent: 10          # 最大并发任务数
  default_timeout: 300        # 默认超时（秒）

  scheduler:
    enabled: true             # 是否启用定时任务调度器

  cleanup:
    sessions_ttl_hours: 168   # 会话过期时间（7天）
    tasks_retention_hours: 24 # 任务结果保留时间
```

---

## 8. 数据流图

```
用户请求 → Agent Core
              │
              ├─ [短工具] ─→ ToolRegistry.execute() → SSE stream 直接返回
              │
              └─ [长时工具/索引构建]
                   │
                   ▼
             TaskManager.submit()
                   │
                   ├─ 返回 task_id → SSE "task_started" event
                   │
                   ├─ asyncio.create_task(_execute)
                   │       │
                   │       ├─ Semaphore 控制并发
                   │       ├─ wait_for(timeout)
                   │       └─ 更新任务状态
                   │
                   ▼
             用户轮询 GET /tasks/{task_id}
                   │
                   ├─ status: completed → SSE "task_completed" + result
                   └─ status: failed    → SSE "task_failed" + error

定时任务 (APScheduler)
   ├── cleanup_sessions (每天 02:00)
   ├── cleanup_tasks (每小时)
   └── cleanup_memories (每天 03:00)
```

---

## 9. 设计总结

| 特性 | 实现方式 |
|------|---------|
| **异步任务** | asyncio + Semaphore 并发控制，超时保护 |
| **定时调度** | APScheduler，支持 cron / interval trigger |
| **状态查询** | REST API 轮询 + SSE 事件推送双通道 |
| **自动清理** | 过期会话、任务结果、记忆条目定期清理 |
| **长时工具** | 后台执行 + task_id 追踪，不阻塞 HTTP 连接 |
