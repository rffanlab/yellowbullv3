"""Base LLM abstraction with streaming support."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    role: Role
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


@dataclass
class LLMResponse:
    content: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, int] | None = None


@dataclass
class StreamChunk:
    """A single chunk from streaming response."""

    delta: str = ""
    done: bool = False
    tool_call: dict[str, Any] | None = None


class BaseLLM(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
    ) -> LLMResponse | AsyncIterator[StreamChunk]:
        """Send a chat request. If stream=True, returns an async iterator."""
        ...

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text for context window management."""
        ...
