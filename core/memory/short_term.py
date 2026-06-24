"""Short-term memory: in-memory message buffer with capacity limits."""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

from models.message import Message

logger = logging.getLogger(__name__)


class ShortTermMemory:
    """In-memory short-term context store.

    Keeps a bounded list of recent messages for the current session.
    When capacity is exceeded, oldest messages are evicted (FIFO).
    """

    def __init__(self, capacity: int = 100) -> None:
        self._capacity = max(capacity, 1)
        self._messages: deque[Message] = deque(maxlen=self._capacity)

    # ── Core operations ────────────────────────────────────────────

    def append(self, message: Message) -> None:
        """Append a message to memory. Evicts oldest if at capacity."""
        self._messages.append(message)
        logger.debug(
            "ShortTermMemory.append role=%s len=%d/%d",
            message.role,
            len(self._messages),
            self._capacity,
        )

    def get_recent(self, limit: int = 50) -> list[Message]:
        """Return the most recent *limit* messages (newest last)."""
        return list(self._messages)[-limit:]

    # ── Utility ────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        """Current number of stored messages."""
        return len(self._messages)

    @property
    def capacity(self) -> int:
        """Maximum message capacity."""
        return self._capacity

    def clear(self) -> None:
        """Wipe all memory contents."""
        self._messages.clear()
        logger.info("ShortTermMemory cleared")

    def to_dict_list(self) -> list[dict[str, Any]]:
        """Serialize messages for API responses."""
        return [
            {
                "id": m.id,
                "role": m.role.value if hasattr(m.role, "value") else str(m.role),
                "content": m.content,
                "tool_calls": m.tool_calls,
                "tool_call_id": m.tool_call_id,
                "created_at": m.created_at.isoformat(),
            }
            for m in self._messages
        ]
