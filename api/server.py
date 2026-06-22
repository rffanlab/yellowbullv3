"""FastAPI application with chat, session management endpoints."""

from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Global agent instance — set by create_app()
_agent = None


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    user_id: str
    message: str


class ChatResponse(BaseModel):
    content: str
    session_id: str
    tool_results: list[dict] = []
    needs_clarification: bool = False
    usage: Optional[dict] = None


def create_app() -> FastAPI:
    """Application factory for uvicorn --factory mode."""
    from config.settings import load_settings
    from core.agent import Agent, ChatRequest as AgentChatRequest
    from llm.factory import create_llm

    app = FastAPI(title="YellowBull Agent API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize agent
    settings = load_settings()
    llm_config = settings.llm.providers.get(settings.llm.active, {})
    llm = create_llm(settings.llm.active, llm_config)
    global _agent
    _agent = Agent(settings, llm)

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

    @app.get("/api/health")
    async def health() -> dict:
        return {"status": "ok"}

    return app


# Module-level app for direct uvicorn usage (no --factory flag needed)
app = FastAPI(title="YellowBull Agent API", version="0.1.0")


@app.get("/api/health")
async def health():
    return {"status": "ok"}
