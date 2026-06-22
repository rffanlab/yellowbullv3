"""Context builder converts session messages to LLM request format."""

from llm.base import Message as LLMMessage, Role
from models.message import MessageRole
from models.session import Session


class ContextBuilder:
    """Convert internal session messages to LLM API format."""

    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt

    def build(self, session: Session, window_size: int = 48) -> list[LLMMessage]:
        context_msgs = session.get_context_messages(window_size)
        messages = [LLMMessage(role=Role.SYSTEM, content=self.system_prompt)]

        for msg in context_msgs:
            if msg.role == MessageRole.USER:
                messages.append(LLMMessage(role=Role.USER, content=msg.content))
            elif msg.role == MessageRole.ASSISTANT:
                messages.append(
                    LLMMessage(
                        role=Role.ASSISTANT,
                        content=msg.content,
                        tool_calls=msg.tool_calls,
                    )
                )
            elif msg.role == MessageRole.TOOL:
                messages.append(
                    LLMMessage(
                        role=Role.TOOL,
                        content=msg.content,
                        tool_call_id=msg.tool_call_id,
                    )
                )

        return messages
