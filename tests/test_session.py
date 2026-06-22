"""Tests for session management."""

from datetime import datetime, timedelta

import pytest


def test_create_session():
    from core.session_manager import SessionManager

    mgr = SessionManager()
    session = mgr.create("user-1")
    assert session.user_id == "user-1"
    assert len(session.session_id) > 0
    assert session in [mgr.get(session.session_id)]


def test_get_session():
    from core.session_manager import SessionManager

    mgr = SessionManager()
    s = mgr.create("u1")
    retrieved = mgr.get(s.session_id)
    assert retrieved is not None
    assert retrieved.user_id == "u1"


def test_delete_session():
    from core.session_manager import SessionManager

    mgr = SessionManager()
    s = mgr.create("u1")
    assert mgr.delete(s.session_id) is True
    assert mgr.get(s.session_id) is None


def test_cleanup_expired():
    from core.session_manager import SessionManager
    from models.session import Session

    mgr = SessionManager()
    s = mgr.create("u1")
    # Make it old
    s.updated_at = datetime.now() - timedelta(hours=2)
    mgr.cleanup_expired(max_age_seconds=3600)
    assert mgr.get(s.session_id) is None


def test_context_window():
    """Test sliding window context."""
    from models.message import Message, MessageRole
    from models.session import Session

    session = Session()
    for i in range(100):
        session.add_message(Message(
            id=f"msg-{i}",
            role=MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT,
            content=f"message {i}",
        ))

    context = session.get_context_messages(window_size=10)
    # Should have last 10 non-system messages (no system in this test)
    assert len(context) == 10


async def test_agent_chat_creates_session():
    """Test agent creates new session when none provided."""
    from config.settings import Settings
    from core.agent import Agent, ChatRequest

    class MockLLM:
        async def chat(self, messages, tools=None):
            from llm.base import LLMResponse
            return LLMResponse(content="Hello!")

        def count_tokens(self, text):
            return len(text) // 4

    settings = Settings()
    agent = Agent(settings, MockLLM())

    result = await agent.chat(ChatRequest(
        session_id=None,
        user_id="test-user",
        message="Hi",
    ))
    assert result.content == "Hello!"
    assert len(result.session_id) > 0


async def test_agent_reuses_session():
    """Test agent reuses existing session."""
    from config.settings import Settings
    from core.agent import Agent, ChatRequest

    class MockLLM:
        async def chat(self, messages, tools=None):
            from llm.base import LLMResponse
            return LLMResponse(content="OK")

        def count_tokens(self, text):
            return len(text) // 4

    settings = Settings()
    agent = Agent(settings, MockLLM())

    r1 = await agent.chat(ChatRequest(session_id=None, user_id="u", message="Hi"))
    r2 = await agent.chat(ChatRequest(
        session_id=r1.session_id,
        user_id="u",
        message="Again",
    ))
    assert r1.session_id == r2.session_id


async def test_agent_max_chain_depth():
    """Test agent stops at max chain depth."""
    from config.settings import AgentConfig, Settings
    from core.agent import Agent, ChatRequest
    from llm.base import LLMResponse

    # LLM that always returns tool calls (infinite loop without limit)
    class LoopLLM:
        async def chat(self, messages, tools=None):
            return LLMResponse(
                content=None,
                tool_calls=[{
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "nonexistent", "arguments": "{}"},
                }],
            )

        def count_tokens(self, text):
            return 0

    settings = Settings(agent=AgentConfig(max_chain_depth=2))
    agent = Agent(settings, LoopLLM())

    result = await agent.chat(ChatRequest(
        session_id=None,
        user_id="u",
        message="loop test",
    ))
    # Should hit max depth and return fallback
    assert "抱歉" in result.content or len(result.content) > 0


async def test_context_builder():
    """Test context builder converts messages correctly."""
    from config.settings import Settings
    from core.context_builder import ContextBuilder
    from models.message import Message, MessageRole
    from models.session import Session

    session = Session()
    session.add_message(Message(
        id="1", role=MessageRole.USER, content="Hello"
    ))
    session.add_message(Message(
        id="2", role=MessageRole.ASSISTANT, content="Hi there!"
    ))

    builder = ContextBuilder("You are helpful")
    messages = builder.build(session)

    assert len(messages) == 3  # system + 2 user/assistant
    assert messages[0].role.value == "system"
    assert messages[1].content == "Hello"


async def test_context_builder_with_tool_calls():
    """Test context builder handles tool call messages."""
    from config.settings import Settings
    from core.context_builder import ContextBuilder
    from models.message import Message, MessageRole
    from models.session import Session

    session = Session()
    session.add_message(Message(
        id="1", role=MessageRole.USER, content="What time is it?"
    ))
    session.add_message(Message(
        id="2",
        role=MessageRole.ASSISTANT,
        tool_calls=[{"id": "call-1", "type": "function"}],
    ))
    session.add_message(Message(
        id="3",
        role=MessageRole.TOOL,
        content="12:00 PM",
        tool_call_id="call-1",
    ))

    builder = ContextBuilder("You are helpful")
    messages = builder.build(session)

    assert len(messages) == 4  # system + user + assistant + tool
    assert messages[3].role.value == "tool"
    assert messages[3].tool_call_id == "call-1"
