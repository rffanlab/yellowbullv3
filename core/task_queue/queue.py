"""Async task queue for background processing."""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)


class TaskStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TaskResult:
    task_id: str
    status: TaskStatus
    result: Any = None
    error: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None


TaskFn = Callable[..., Coroutine[Any, Any, Any]]


class TaskQueue:
    """Async task queue with worker pool.

    Submits coroutines for background execution and tracks their status.
    """

    def __init__(self, max_workers: int = 4) -> None:
        self._max_workers = max_workers
        self._queue: asyncio.Queue[tuple[str, TaskFn, tuple, dict]] = asyncio.Queue()
        self._results: dict[str, TaskResult] = {}
        self._workers: list[asyncio.Task] = []
        self._running = False

    async def start(self) -> None:
        """Start worker pool."""
        if self._running:
            return
        self._running = True
        for i in range(self._max_workers):
            worker = asyncio.create_task(self._worker_loop(i))
            self._workers.append(worker)
        logger.info("TaskQueue started with %d workers", self._max_workers)

    async def stop(self, wait: bool = True) -> None:
        """Stop all workers."""
        self._running = False
        if wait:
            await asyncio.gather(*self._workers, return_exceptions=True)
        logger.info("TaskQueue stopped")

    def submit(
        self,
        fn: TaskFn,
        *args: Any,
        task_id: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Submit a coroutine for background execution.

        Args:
            fn: Async callable to execute.
            *args: Positional arguments for the callable.
            task_id: Optional custom task ID (auto-generated if not provided).
            **kwargs: Keyword arguments for the callable.

        Returns:
            Task ID for status tracking.
        """
        tid = task_id or str(uuid.uuid4())
        self._results[tid] = TaskResult(
            task_id=tid, status=TaskStatus.PENDING
        )
        self._queue.put_nowait((tid, fn, args, kwargs))
        logger.debug("Task submitted: id=%s", tid)
        return tid

    def get_status(self, task_id: str) -> TaskResult | None:
        """Query task execution status."""
        return self._results.get(task_id)

    async def wait_for(self, task_id: str, timeout: float | None = None) -> TaskResult | None:
        """Wait for a task to complete. Returns result or None on timeout."""
        start = datetime.now()
        while True:
            result = self.get_status(task_id)
            if result and result.status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                return result
            if timeout and (datetime.now() - start).total_seconds() > timeout:
                return None
            await asyncio.sleep(0.1)

    # ── Internal worker loop ───────────────────────────────────────

    async def _worker_loop(self, worker_id: int) -> None:
        """Process tasks from the queue."""
        logger.debug("TaskQueue worker %d started", worker_id)
        while self._running:
            try:
                tid, fn, args, kwargs = await asyncio.wait_for(
                    self._queue.get(), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue

            result = self._results[tid]
            result.status = TaskStatus.RUNNING
            result.started_at = datetime.now()

            try:
                coro = fn(*args, **kwargs)
                result.result = await coro
                result.status = TaskStatus.COMPLETED
                logger.debug("Task completed: id=%s", tid)
            except Exception as e:
                result.status = TaskStatus.FAILED
                result.error = str(e)
                logger.warning("Task failed: id=%s error=%s", tid, e)
            finally:
                result.completed_at = datetime.now()
                self._queue.task_done()
