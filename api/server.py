"""FastAPI application with chat, session management endpoints."""

import time
from collections import defaultdict
from collections.abc import AsyncIterator

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Global agent instance — set by create_app()
_agent = None


# ── Pydantic V2 request/response models ────────────────────────────────


class ChatRequest(BaseModel):
    session_id: str | None = None
    user_id: str
    message: str


class ChatResponse(BaseModel):
    content: str
    session_id: str
    tool_results: list[dict] = []
    needs_clarification: bool = False
    usage: dict | None = None


class SessionInfo(BaseModel):
    """Session metadata for listing."""

    session_id: str
    user_id: str
    message_count: int
    created_at: str
    updated_at: str


# ── Rate Limiter Middleware ─────────────────────────────────────────────


class RateLimiter:
    """Simple in-memory sliding-window rate limiter."""

    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, key: str) -> bool:
        now = time.time()
        cutoff = now - self.window_seconds
        # Prune old entries
        self._requests[key] = [t for t in self._requests[key] if t > cutoff]
        if len(self._requests[key]) >= self.max_requests:
            return False
        self._requests[key].append(now)
        return True


_rate_limiter = RateLimiter()


async def rate_limit_middleware(request, call_next):
    """FastAPI middleware for per-IP rate limiting."""
    client_ip = request.client.host if request.client else "unknown"
    if not _rate_limiter.is_allowed(client_ip):
        return HTTPException(429, "Rate limit exceeded")
    return await call_next(request)


# ── SSE Streaming Helper ───────────────────────────────────────────────


async def sse_stream(generator: AsyncIterator[str]) -> AsyncIterator[str]:
    """Wrap an async text generator into SSE format."""
    async for chunk in generator:
        yield f"data: {chunk}\n\n"
    yield "data: [DONE]\n\n"


# ── Streaming chat endpoint helper ─────────────────────────────────────


async def stream_chat_response(request: ChatRequest):
    """Generate SSE chunks by calling LLM with streaming."""
    from core.agent import Agent

    if not _agent:
        yield "data: [ERROR] Agent not initialized\n\n"
        return

    # Reuse agent's session logic but stream LLM response
    session = _agent._get_or_create_session(
        Agent.ChatRequest(session_id=request.session_id, user_id=request.user_id, message="")
    )

    import uuid

    from models.message import Message, MessageRole

    user_msg = Message(
        id=str(uuid.uuid4()),
        role=MessageRole.USER,
        content=request.message,
    )
    session.add_message(user_msg)

    context = _agent.context_builder.build(session, _agent.settings.agent.context_window)
    from tools.registry import ToolRegistry

    tools = ToolRegistry.to_function_definitions()

    try:
        response = await _agent.llm.chat(messages=context, tools=tools, stream=True)
    except Exception as e:
        yield f"data: [ERROR] {e}\n\n"
        return

    full_content = ""
    if hasattr(response, "__aiter__"):
        from llm.base import StreamChunk

        async for chunk in response:
            if isinstance(chunk, StreamChunk):
                if chunk.done:
                    # Save to session
                    assistant_msg = Message(
                        id=str(uuid.uuid4()),
                        role=MessageRole.ASSISTANT,
                        content=full_content or "(empty response)",
                    )
                    session.add_message(assistant_msg)
                    yield f"data: {{\"session_id\": \"{session.session_id}\"}}\n\n"
                elif chunk.delta:
                    full_content += chunk.delta
                    yield f"data: {chunk.delta}\n\n"
    else:
        # Non-streaming fallback
        content = response.content or ""
        assistant_msg = Message(
            id=str(uuid.uuid4()),
            role=MessageRole.ASSISTANT,
            content=content,
        )
        session.add_message(assistant_msg)
        yield f"data: {content}\n\n"
        yield f"data: {{\"session_id\": \"{session.session_id}\"}}\n\n"


# ── App Factory ────────────────────────────────────────────────────────


def create_app() -> FastAPI:
    """Application factory for uvicorn --factory mode."""
    from config.settings import load_settings
    from core.agent import Agent
    from core.agent import ChatRequest as AgentChatRequest
    from llm.factory import create_llm

    app = FastAPI(title="YellowBull Agent API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.middleware("http")(rate_limit_middleware)

    # Initialize agent
    settings = load_settings()
    llm_config = settings.llm.providers.get(settings.llm.active, {})
    llm = create_llm(settings.llm.active, llm_config)
    global _agent
    _agent = Agent(settings, llm)

    # ── Chat endpoints ───────────────────────────────────────────────

    @app.post("/api/chat")
    async def chat(request: ChatRequest) -> ChatResponse:
        result = await _agent.chat(AgentChatRequest(**request.model_dump()))  # type: ignore[arg-type]
        return ChatResponse(
            content=result.content,
            session_id=result.session_id,
            tool_results=[],
            needs_clarification=result.needs_clarification,
            usage=result.usage,
        )

    @app.post("/api/chat/stream")
    async def chat_stream(request: ChatRequest) -> StreamingResponse:
        """SSE streaming endpoint for real-time token delivery."""
        return StreamingResponse(
            sse_stream(stream_chat_response(request)),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ── WebSocket endpoint ───────────────────────────────────────────

    @app.websocket("/ws/chat")
    async def websocket_chat(websocket: WebSocket):
        await websocket.accept()
        session_id = None
        user_id = ""

        try:
            while True:
                data = await websocket.receive_json()
                message = data.get("message", "")
                if not user_id and data.get("user_id"):
                    user_id = data["user_id"]

                if not _agent:
                    await websocket.send_json({"error": "Agent not initialized"})
                    continue

                from core.agent import ChatRequest as AgentChatRequest
                chat_req = AgentChatRequest(
                    session_id=session_id,
                    user_id=user_id or "ws",
                    message=message,
                )
                result = await _agent.chat(chat_req)  # type: ignore[arg-type]
                session_id = result.session_id

                # Send incremental tokens via WebSocket
                await websocket.send_json({
                    "type": "content",
                    "content": result.content,
                    "session_id": result.session_id,
                })
                if result.tool_results:
                    await websocket.send_json({
                        "type": "tool_results",
                        "results": result.tool_results,
                    })
                await websocket.send_json({"type": "done"})

        except WebSocketDisconnect:
            pass

    # ── Session management endpoints ─────────────────────────────────

    @app.delete("/api/sessions/{session_id}", status_code=204)
    async def delete_session(session_id: str):
        if _agent and _agent.session_manager.delete(session_id):
            return None
        raise HTTPException(404, "Session not found")

    @app.get("/api/sessions/{session_id}/history")
    async def get_history(session_id: str) -> dict:
        if not _agent:
            raise HTTPException(500, "Agent not initialized")
        session = _agent.session_manager.get(session_id)
        if not session:
            raise HTTPException(404, "Session not found")
        return {
            "session_id": session.session_id,
            "messages": [
                {
                    "role": m.role.value,
                    "content": m.content,
                    "created_at": str(m.created_at),
                }
                for m in session.messages
            ],
        }

    @app.get("/api/sessions")
    async def list_sessions(user_id: str | None = None) -> list[SessionInfo]:
        """List all sessions, optionally filtered by user_id."""
        if not _agent:
            raise HTTPException(500, "Agent not initialized")
        results = []
        for _sid, session in _agent.session_manager._sessions.items():
            if user_id and session.user_id != user_id:
                continue
            results.append(SessionInfo(
                session_id=session.session_id,
                user_id=session.user_id,
                message_count=len(session.messages),
                created_at=str(session.created_at),
                updated_at=str(session.updated_at),
            ))
        return results

    # ── Health check ─────────────────────────────────────────────────

    @app.get("/api/health")
    async def health() -> dict:
        return {"status": "ok"}

    return app


# Module-level app for direct uvicorn usage (no --factory flag needed)
app = FastAPI(title="YellowBull Agent API", version="0.1.0")


@app.get("/api/health")
async def health():
    return {"status": "ok"}
