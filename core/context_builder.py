"""Context builder converts session messages to LLM request format."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from llm.base import Message as LLMMessage
from llm.base import Role
from models.message import MessageRole
from models.session import Session

if TYPE_CHECKING:
    from llm.base import BaseLLM


# ── Token counting helpers ─────────────────────────────────────────────


def _count_tokens(text: str) -> int:
    """Estimate token count using a simple heuristic (works without tiktoken)."""
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        # Fallback: ~4 chars per token for English, ~2 for CJK-heavy text
        has_cjk = any("\u4e00" <= c <= "\u9fff" for c in text)
        divisor = 2 if has_cjk else 4
        return max(1, len(text) // divisor)


def _message_tokens(msg: LLMMessage) -> int:
    """Estimate tokens consumed by a single message (content + overhead)."""
    base = 4  # role tag + structural overhead
    content_len = _count_tokens(msg.content or "")

    if msg.tool_calls:
        for tc in msg.tool_calls:
            name_len = len(tc.get("function", {}).get("name", "")) // 4
            args_len = len(json.dumps(tc.get("function", {}).get("arguments", {}))) // 4
            content_len += max(1, name_len + args_len)

    return base + max(1, content_len)


# ── Compression strategy ───────────────────────────────────────────────


@dataclass
class CompressionConfig:
    """Controls how older turns are summarized."""

    enabled: bool = True
    max_summary_tokens: int = 300
    min_turns_to_keep: int = 6  # keep last N turns uncompressed


def _summarize_turns(turns: list[list[LLMMessage]], budget: int) -> str:
    """Summarize a batch of conversation turns into a compact summary."""
    parts: list[str] = []
    for turn in turns:
        user_text = ""
        assistant_text = ""
        for msg in turn:
            if msg.role == Role.USER:
                user_text = msg.content or ""
            elif msg.role == Role.ASSISTANT:
                assistant_text = msg.content or ""

        # Truncate long messages before summarizing
        if _count_tokens(user_text) > 100:
            tokens = user_text.split()
            user_text = " ".join(tokens[:50]) + " ..."

        parts.append(f"User: {user_text}\nAssistant: {assistant_text}")

    summary = f"[Earlier conversation ({len(turns)} turns):\n" + "\n---\n".join(parts) + "\n]"
    # Hard truncate if summary itself is too long
    while _count_tokens(summary) > budget and len(summary) > 50:
        summary = summary[:-20]
    return summary


# ── Context Builder ────────────────────────────────────────────────────


class ContextBuilder:
    """Convert internal session messages to LLM API format with token-aware truncation."""

    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt

    def build(
        self,
        session: Session,
        window_size: int = 48,
        compression: CompressionConfig | None = None,
        llm: BaseLLM | None = None,
        work_dir: str | None = None,
    ) -> list[LLMMessage]:
        """Build context messages with token-aware truncation.

        Args:
            session: Active session with message history.
            window_size: Max tokens for the context window (not message count).
            compression: Optional config for summarizing older turns.
            llm: Optional LLM instance for generating summaries via API.
            work_dir: Working directory to inject into system prompt.
        """
        if compression is None:
            compression = CompressionConfig()

        raw_msgs = session.get_context_messages(window_size)

        # Build effective system prompt with work_dir injection
        effective_prompt = self.system_prompt
        if work_dir:
            effective_prompt += (
                f"\n\n## Working Directory\n"
                f"- Your working directory is: `{work_dir}`\n"
                f"- All file and folder operations MUST be performed within this directory.\n"
                f"- When creating files or folders, use absolute paths under `{work_dir}`.\n"
                f"- Example: to create a file named `test.py`, the path should be `{work_dir}\\test.py` (Windows) or `{work_dir}/test.py` (Linux/Mac).\n"
            )

        messages = [LLMMessage(role=Role.SYSTEM, content=effective_prompt)]

        # Convert to LLM format first
        llm_msgs: list[LLMMessage] = []
        for msg in raw_msgs:
            if msg.role == MessageRole.USER:
                llm_msgs.append(LLMMessage(role=Role.USER, content=msg.content))
            elif msg.role == MessageRole.ASSISTANT:
                llm_msgs.append(
                    LLMMessage(
                        role=Role.ASSISTANT,
                        content=msg.content,
                        tool_calls=msg.tool_calls,
                    )
                )
            elif msg.role == MessageRole.TOOL:
                llm_msgs.append(
                    LLMMessage(
                        role=Role.TOOL,
                        content=msg.content,
                        tool_call_id=msg.tool_call_id,
                    )
                )

        # Calculate system prompt overhead
        system_tokens = _count_tokens(self.system_prompt) + 4
        available = max(window_size - system_tokens, 100)

        if not compression.enabled or sum(_message_tokens(m) for m in llm_msgs) <= available:
            messages.extend(llm_msgs)
            return messages

        # ── Token-aware truncation with turn grouping ────────────────

        # Group into turns (user + assistant/tool pairs)
        turns: list[list[LLMMessage]] = []
        current_turn: list[LLMMessage] = []
        for msg in llm_msgs:
            if msg.role == Role.USER and current_turn:
                turns.append(current_turn)
                current_turn = []
            current_turn.append(msg)
        if current_turn:
            turns.append(current_turn)

        # Keep last N turns uncompressed, compress the rest
        min_keep = compression.min_turns_to_keep
        if len(turns) <= min_keep:
            messages.extend(llm_msgs)
            return messages

        old_turns = turns[:-min_keep]
        new_turns = turns[-min_keep:]

        # Calculate budget for summary
        new_tokens = sum(_message_tokens(m) for turn in new_turns for m in turn)
        summary_budget = min(compression.max_summary_tokens, available - new_tokens)

        if summary_budget > 50:
            summary_text = _summarize_turns(old_turns, summary_budget)
            messages.append(LLMMessage(role=Role.SYSTEM, content=summary_text))

        # Append recent turns
        for turn in new_turns:
            messages.extend(turn)

        return messages
