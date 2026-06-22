# API Server 详细设计（FastAPI）

## 1. 职责边界

| 职责 | 说明 |
|------|------|
| **HTTP REST API** | Session CRUD、Agent 同步调用、配置查询 |
| **SSE 流式端点** | Agent 流式输出，前端实时渲染 |
| **WebSocket** | 可选的双向通信通道（未来扩展） |
| **中间件** | CORS、认证、请求日志、错误处理 |
| **健康检查** | `/health` 端点，K8s/负载均衡器探针 |

---

## 2. API 路由设计 `api/router.py`

```python
"""
FastAPI router —— Agent API v1。

Endpoints:
- POST   /api/v1/sessions              创建会话
- GET    /api/v1/sessions              列出用户会话
- GET    /api/v1/sessions/{id}         获取会话详情（含消息历史）
- DELETE /api/v1/sessions/{id}         删除会话
- POST   /api/v1/agent/run             Agent 同步执行（非流式）
- POST   /api/v1/agent/stream          Agent 流式执行（SSE）
- GET    /api/v1/config                查询当前配置（脱敏）
- POST   /api/v1/config/reload         手动触发配置重载
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Any

from agent.core import AgentCore
from session.manager import get_session_manager
from config.manager import get_manager

router = APIRouter(prefix="/api/v1", tags=["Agent"])


# ==================== Request/Response Models ====================

class CreateSessionRequest(BaseModel):
    user_id: str = Field(..., description="用户标识")
    title: str = Field("", description="会话标题（可选，自动生成）")


class SendMessageRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000, description="用户消息")
    stream: bool = Field(False, description="是否流式响应")


class SessionResponse(BaseModel):
    id: str
    user_id: str
    title: str
    status: str
    message_count: int
    created_at: str
    updated_at: str


class MessageResponse(BaseModel):
    id: str
    role: str
    content: str
    tool_calls: list[dict] = []
    token_count: int
    created_at: str


# ==================== Session Endpoints ====================

@router.post("/sessions", response_model=SessionResponse, status_code=201)
async def create_session(req: CreateSessionRequest):
    """创建新会话"""
    sm = get_session_manager()
    session = await sm.create_session(user_id=req.user_id, title=req.title)
    return SessionResponse(
        id=session.id, user_id=session.user_id, title=session.title,
        status=session.status.value, message_count=session.message_count,
        created_at=session.created_at.isoformat(), updated_at=session.updated_at.isoformat(),
    )


@router.get("/sessions", response_model=list[SessionResponse])
async def list_sessions(user_id: str = Query(...), limit: int = Query(50, ge=1, le=200)):
    """列出用户会话"""
    sm = get_session_manager()
    sessions = await sm.list_user_sessions(user_id)
    return [
        SessionResponse(
            id=s.id, user_id=s.user_id, title=s.title,
            status=s.status.value, message_count=s.message_count,
            created_at=s.created_at.isoformat(), updated_at=s.updated_at.isoformat(),
        )
        for s in sessions
    ]


@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """获取会话详情（含消息历史）"""
    sm = get_session_manager()
    session = await sm.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    messages = await sm.get_messages(session_id)
    return {
        "session": SessionResponse(
            id=session.id, user_id=session.user_id, title=session.title,
            status=session.status.value, message_count=session.message_count,
            created_at=session.created_at.isoformat(), updated_at=session.updated_at.isoformat(),
        ),
        "messages": [
            MessageResponse(
                id=m.id, role=m.role, content=m.content,
                tool_calls=m.tool_calls, token_count=m.token_count,
                created_at=m.created_at.isoformat(),
            )
            for m in messages
        ],
    }


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_session(session_id: str):
    """删除会话"""
    sm = get_session_manager()
    session = await sm.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    await sm.delete_session(session_id)


# ==================== Agent Endpoints ====================

_agent_core: AgentCore | None = None   # 由 main.py 注入


def set_agent_core(core: AgentCore):
    global _agent_core
    _agent_core = core


@router.post("/agent/run")
async def agent_run(req: SendMessageRequest, session_id: str = Query(...)):
    """
    Agent 同步执行（非流式）。

    适用于：移动端、后台任务、不需要实时输出的场景。
    """
    if not _agent_core:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    result = await _agent_core.run_sync(req.message, session_id)

    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result.get("content", "Agent error"))

    return {
        "session_id": session_id,
        "content": result["content"],
        "events": result.get("events", []),
    }


@router.post("/agent/stream")
async def agent_stream(req: SendMessageRequest, session_id: str = Query(...)):
    """
    Agent 流式执行（SSE）。

    Response type: text/event-stream
    Client 通过 EventSource API 消费。
    """
    from fastapi.responses import StreamingResponse

    if not _agent_core:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    async def event_generator():
        try:
            async for event in _agent_core.run(req.message, session_id):
                yield format_sse_event(event["event"], event["data"])
        except Exception as e:
            yield format_sse_event("error", {"message": str(e)})
        finally:
            yield format_sse_event("done", {"status": "completed"})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",   # 禁用 Nginx buffering
        },
    )


def format_sse_event(event: str, data: dict[str, Any]) -> str:
    """格式化为 SSE event"""
    return f"event: {event}\ndata: {data.model_dump_json() if hasattr(data, 'model_dump_json') else json.dumps(data, ensure_ascii=False)}\n\n"


# ==================== Config Endpoints ====================

@router.get("/config")
async def get_config():
    """查询当前配置（脱敏——API key 等敏感字段被遮蔽）"""
    manager = get_manager()
    return manager.settings.masked_dict()


@router.post("/config/reload", status_code=204)
async def reload_config():
    """手动触发配置重载"""
    manager = get_manager()
    await manager.reload()


# ==================== Health Check ====================

@router.get("/health")
async def health_check():
    return {"status": "ok"}
```

---

## 3. FastAPI 应用入口 `api/app.py`

```python
"""
FastAPI application factory。

职责：
- 创建 app 实例
- 注册中间件（CORS、请求日志）
- 注册路由
- lifespan 管理（startup / shutdown）
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

from api.router import router, set_agent_core
from config.manager import get_manager
from llm.provider_factory import create_provider
from tools.registry import ToolRegistry
from session.manager import get_session_manager
from agent.core import AgentCore

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifecycle manager。

    Startup order:
    1. ConfigManager (load YAML + start watch loop)
    2. SessionManager (connect SQLite)
    3. LLM Provider (initialize client)
    4. ToolRegistry (register builtin tools)
    5. AgentCore (wire everything together)

    Shutdown order: reverse of startup
    """
    # ---- Startup ----
    logger.info("Starting application...")

    # 1. Config
    config_mgr = get_manager()
    await config_mgr.start()

    # 2. Session Manager
    sm = get_session_manager()
    await sm.start()

    # 3. LLM Provider
    settings = config_mgr.settings
    llm_provider = create_provider(settings.llm)
    await llm_provider.initialize()

    # 4. Tool Registry + builtin tools
    registry = ToolRegistry()
    _register_builtin_tools(registry, settings.tools.builtin)

    # 5. Agent Core
    agent = AgentCore(
        llm_provider=llm_provider,
        tool_registry=registry,
        max_tool_rounds=settings.agent.max_tool_rounds,
        system_prompt=settings.agent.system_prompt,
    )
    set_agent_core(agent)

    # 6. Config watching bridges
    from tools.config_bridge import setup_tool_config_watching
    from session.config_bridge import setup_session_config_watching
    setup_tool_config_watching()
    setup_session_config_watching()

    logger.info("Application started successfully")
    yield

    # ---- Shutdown ----
    logger.info("Shutting down application...")
    await sm.stop()
    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    app = FastAPI(
        title="YellowBull Agent",
        version="1.0.0",
        lifespan=lifespan,
    )

    # ---- Middleware ----
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],           # 生产环境改为具体域名
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"],           # 生产环境改为具体域名
    )

    # ---- Routers ----
    app.include_router(router)

    return app


async def _register_builtin_tools(registry: ToolRegistry, config: dict):
    """根据配置注册内置工具"""
    from tools.builtin.current_time import CurrentTimeTool
    from tools.builtin.calculator import CalculatorTool
    from tools.builtin.web_search import WebSearchTool

    tool_map = {
        "current_time": CurrentTimeTool,
        "calculator": CalculatorTool,
        "web_search": lambda: WebSearchTool(settings=config.get("web_search", {}).get("settings", {})),
    }

    for name, cfg in config.items():
        if name in tool_map:
            enabled = cfg.get("enabled", True)
            settings = cfg.get("settings", {})
            tool_instance = tool_map[name]()
            await registry.register(tool_instance, enabled=enabled, settings=settings)


# ---- Uvicorn entry point ----
if __name__ == "__main__":
    import uvicorn
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## 4. 中间件设计 `api/middleware.py`

```python
"""
Request logging middleware。

记录：method, path, status_code, duration_ms, client_ip
不记录：request body（可能含敏感信息）
"""

import time
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.monotonic()
        client_ip = request.client.host if request.client else "unknown"

        response = await call_next(request)

        elapsed_ms = round((time.monotonic() - start) * 1000, 1)
        logger.info(
            f"{request.method} {request.url.path} → {response.status_code} "
            f"({elapsed_ms}ms) [client={client_ip}]"
        )
        return response
```

---

## 5. 认证中间件（可选）`api/auth.py`

```python
"""
API Key 认证中间件。

从请求头 X-API-Key 或查询参数 api_key 获取密钥。
与 YAML 配置 security.api_keys 比对。
"""

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware
from config.manager import get_manager


class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Health check 端点免认证
        if request.url.path in ("/api/v1/health", "/docs", "/openapi.json"):
            return await call_next(request)

        # 获取 API key
        api_key = (
            request.headers.get("X-API-Key")
            or request.query_params.get("api_key")
        )

        if not api_key:
            raise HTTPException(status_code=401, detail="API key required")

        settings = get_manager().settings.security
        if api_key not in settings.api_keys:
            raise HTTPException(status_code=403, detail="Invalid API key")

        return await call_next(request)
```

---

## 6. OpenAPI / Swagger

FastAPI 自动生成：
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI JSON**: `http://localhost:8000/openapi.json`

---

## 7. API 调用示例

### 创建会话 + 流式对话

```bash
# 1. Create session
curl -X POST http://localhost:8000/api/v1/sessions \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user_001", "title": "AI Discussion"}'

# Response: {"id": "uuid...", "user_id": "user_001", ...}

# 2. Stream chat (SSE)
curl -N http://localhost:8000/api/v1/agent/stream?session_id=uuid... \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the weather in Beijing?", "stream": true}'

# SSE output:
# event: chunk
# data: {"content": "Let"}
#
# event: chunk
# data: {"content": " me check"}
#
# event: tool_start
# data: {"name": "web_search", "args": {"query": "Beijing weather"}}
#
# event: tool_end
# data: {"name": "web_search", "success": true, "content": "..."}
#
# event: chunk
# data: {"content": "The current weather in Beijing is..."}
#
# event: done
# data: {"status": "completed"}
```

---

## 8. 设计总结

| 特性 | 实现方式 |
|------|---------|
| **REST API** | FastAPI + Pydantic models，自动生成 OpenAPI spec |
| **SSE 流式** | `StreamingResponse(text/event-stream)` + async generator |
| **Lifespan** | `@asynccontextmanager` 管理 startup/shutdown 顺序 |
| **中间件** | CORS、请求日志、可选 API Key 认证 |
| **健康检查** | `/api/v1/health` → K8s liveness/readiness probe |
| **配置查询** | `/api/v1/config`（脱敏）+ `/api/v1/config/reload` |
