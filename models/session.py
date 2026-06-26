"""Session models for conversation state management."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from models.message import Message, MessageRole


@dataclass
class SessionState:
    """Multi-step task state tracking."""

    step: int = 0
    chain_depth: int = 0
    tool_retry_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class Session:
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    messages: list[Message] = field(default_factory=list)
    state: SessionState = field(default_factory=SessionState)
    work_dir: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def add_message(self, msg: Message):
        self.messages.append(msg)
        self.updated_at = datetime.now()

    def get_context_messages(self, window_size: int) -> list[Message]:
        """Sliding window: keep system messages + last N non-system messages."""
        non_system = [m for m in self.messages if m.role != MessageRole.SYSTEM]
        system_msgs = [m for m in self.messages if m.role == MessageRole.SYSTEM]
        return system_msgs + non_system[-window_size:]
