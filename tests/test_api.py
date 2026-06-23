"""Tests for FastAPI endpoints."""

from unittest.mock import patch


async def test_health_endpoint():
    """Test /api/health returns ok."""
    from unittest.mock import AsyncMock

    from fastapi.testclient import TestClient

    mock_llm = AsyncMock()
    from llm.base import LLMResponse
    mock_llm.chat.return_value = LLMResponse(content="OK")
    mock_llm.count_tokens.return_value = 10

    with patch("llm.factory.create_llm", return_value=mock_llm):
        from api.server import create_app

        app = create_app()
        client = TestClient(app)

        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


async def test_chat_endpoint():
    """Test /api/chat returns valid response."""
    # Patch the LLM to return a simple response
    from unittest.mock import AsyncMock, patch

    from fastapi.testclient import TestClient

    mock_llm = AsyncMock()
    from llm.base import LLMResponse

    mock_llm.chat.return_value = LLMResponse(content="Test response")
    mock_llm.count_tokens.return_value = 10

    with patch("llm.factory.create_llm", return_value=mock_llm):
        from api.server import create_app

        app = create_app()
        client = TestClient(app)

        response = client.post(
            "/api/chat",
            json={"user_id": "test", "message": "Hello"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "content" in data
        assert "session_id" in data


async def test_delete_session():
    """Test session deletion."""
    from unittest.mock import AsyncMock

    from fastapi.testclient import TestClient

    mock_llm = AsyncMock()
    from llm.base import LLMResponse

    mock_llm.chat.return_value = LLMResponse(content="OK")
    mock_llm.count_tokens.return_value = 10

    with patch("llm.factory.create_llm", return_value=mock_llm):
        from api.server import create_app

        app = create_app()
        client = TestClient(app)

        # Create a session via chat
        r = client.post(
            "/api/chat",
            json={"user_id": "test", "message": "Hi"},
        )
        sid = r.json()["session_id"]

        # Delete it
        delete_resp = client.delete(f"/api/sessions/{sid}")
        assert delete_resp.status_code == 204


async def test_get_history():
    """Test session history endpoint."""
    from unittest.mock import AsyncMock

    from fastapi.testclient import TestClient

    mock_llm = AsyncMock()
    from llm.base import LLMResponse

    mock_llm.chat.return_value = LLMResponse(content="OK")
    mock_llm.count_tokens.return_value = 10

    with patch("llm.factory.create_llm", return_value=mock_llm):
        from api.server import create_app

        app = create_app()
        client = TestClient(app)

        # Create session
        r = client.post(
            "/api/chat",
            json={"user_id": "test", "message": "Hi"},
        )
        sid = r.json()["session_id"]

        # Get history
        hist = client.get(f"/api/sessions/{sid}/history")
        assert hist.status_code == 200
        data = hist.json()
        assert len(data["messages"]) >= 1


async def test_delete_nonexistent_session():
    """Test deleting a session that doesn't exist returns 404."""
    from unittest.mock import AsyncMock

    from fastapi.testclient import TestClient

    mock_llm = AsyncMock()
    from llm.base import LLMResponse

    mock_llm.chat.return_value = LLMResponse(content="OK")
    mock_llm.count_tokens.return_value = 10

    with patch("llm.factory.create_llm", return_value=mock_llm):
        from api.server import create_app

        app = create_app()
        client = TestClient(app)

        resp = client.delete("/api/sessions/nonexistent-id")
        assert resp.status_code == 404


async def test_chat_with_tool_calls():
    """Test chat endpoint with tool calling flow."""
    from unittest.mock import AsyncMock

    from fastapi.testclient import TestClient

    # First call returns tool call, second returns final answer
    mock_llm = AsyncMock()
    from llm.base import LLMResponse

    call_count = 0

    async def side_effect(messages, tools=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return LLMResponse(
                content=None,
                tool_calls=[{
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "current_time", "arguments": "{}"},
                }],
            )
        return LLMResponse(content="The current time is 12:00 PM")

    mock_llm.chat.side_effect = side_effect
    mock_llm.count_tokens.return_value = 10

    # Import builtins to register tools
    from tools.builtins import __init__ as _  # noqa: F401

    with patch("llm.factory.create_llm", return_value=mock_llm):
        from api.server import create_app

        app = create_app()
        client = TestClient(app)

        response = client.post(
            "/api/chat",
            json={"user_id": "test", "message": "What time is it?"},
        )
        assert response.status_code == 200
