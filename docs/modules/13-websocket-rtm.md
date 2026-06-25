# WebSocket 实时通信详细设计（WebSocket RTM）

## 1. 职责边界

| 领域 | 说明 |
|------|------|
| **连接管理** | WebSocket 握手、心跳保活、断线重连 |
| **消息路由** | 基于 session_id / channel 的消息分发 |
| **流式输出** | Agent 思考过程实时推送（SSE fallback） |
| **广播机制** | 多客户端同步同一会话状态 |

---

## 2. WebSocket 连接管理 `websocket/connection.py`

```python
"""
WebSocket 连接管理器。

职责：
- 维护活跃连接池
- 心跳检测与超时清理
- 断线重连支持
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class WebSocketConnection:
    """WebSocket 连接信息"""
    connection_id: str
    session_id: str
    user_id: str
    websocket: Any                     # FastAPI WebSocket 对象
    connected_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    is_active: bool = True

    def update_heartbeat(self):
        self.last_heartbeat = time.time()

    @property
    def idle_seconds(self) -> float:
        return time.time() - self.last_heartbeat


class ConnectionManager:
    """连接管理器"""

    def __init__(self, heartbeat_interval: int = 30, timeout_seconds: int = 90):
        self._connections: dict[str, WebSocketConnection] = {}   # connection_id → conn
        self._session_map: dict[str, list[str]] = {}             # session_id → [connection_ids]
        self._heartbeat_interval = heartbeat_interval
        self._timeout_seconds = timeout_seconds
        self._heartbeat_task: asyncio.Task | None = None

    async def start_heartbeat(self):
        """启动心跳检测任务"""
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def stop_heartbeat(self):
        """停止心跳检测"""
        if self._heartbeat_task:
            self._heartbeat_task.cancel()

    async def _heartbeat_loop(self):
        """定期发送心跳并清理超时连接"""
        while True:
            try:
                await asyncio.sleep(self._heartbeat_interval)
                await self._check_timeouts()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat loop error: {e}")

    async def _check_timeouts(self):
        """清理超时连接"""
        to_remove = []
        for conn_id, conn in self._connections.items():
            if conn.idle_seconds > self._timeout_seconds and conn.is_active:
                logger.warning(f"Connection timeout: {conn_id} (idle={conn.idle_seconds:.0f}s)")
                await self.disconnect(conn_id)
                to_remove.append(conn_id)

        for conn_id in to_remove:
            self._connections.pop(conn_id, None)

    async def connect(
        self, connection_id: str, session_id: str, user_id: str, websocket: Any
    ) -> WebSocketConnection:
        """注册新连接"""
        conn = WebSocketConnection(
            connection_id=connection_id,
            session_id=session_id,
            user_id=user_id,
            websocket=websocket,
        )
        self._connections[connection_id] = conn

        if session_id not in self._session_map:
            self._session_map[session_id] = []
        self._session_map[session_id].append(connection_id)

        logger.info(
            f"Connected: id={connection_id} session={session_id} user={user_id} "
            f"(total={len(self._connections)})"
        )
        return conn

    async def disconnect(self, connection_id: str):
        """断开连接"""
        conn = self._connections.pop(connection_id, None)
        if not conn:
            return

        conn.is_active = False

        # 从 session map 中移除
        if conn.session_id in self._session_map:
            if connection_id in self._session_map[conn.session_id]:
                self._session_map[conn.session_id].remove(connection_id)
            if not self._session_map[conn.session_id]:
                del self._session_map[conn.session_id]

        # 关闭 WebSocket
        try:
            await conn.websocket.close(code=1000, reason="Disconnected")
        except Exception as e:
            logger.warning(f"Error closing websocket {connection_id}: {e}")

    def get_by_session(self, session_id: str) -> list[WebSocketConnection]:
        """获取某个会话的所有连接"""
        conn_ids = self._session_map.get(session_id, [])
        return [
            self._connections[cid] for cid in conn_ids
            if cid in self._connections and self._connections[cid].is_active
        ]

    def get_by_user(self, user_id: str) -> list[WebSocketConnection]:
        """获取某个用户的所有连接"""
        return [c for c in self._connections.values() if c.user_id == user_id and c.is_active]

    @property
    def active_count(self) -> int:
        return sum(1 for c in self._connections.values() if c.is_active)
```

---

## 3. 消息协议 `websocket/protocol.py`

```python
"""
WebSocket 消息协议。

所有消息遵循统一的 JSON 格式：
    {
        "type": "<message_type>",
        "id": "<unique_message_id>",
        "timestamp": <unix_timestamp>,
        "payload": { ... },
        "error": null | { "code": "...", "message": "..." }
    }

消息类型：
┌─────────────────────┬──────────┬──────────────────────────┐
│ Type                │ Direction│ Description              │
├─────────────────────┼──────────┼──────────────────────────┤
│ client_message      │ → Server │ 用户发送的消息            │
│ server_message      │ ← Client │ Agent 回复（完整）        │
│ server_chunk        │ ← Client │ Agent 流式输出片段       │
│ tool_call           │ ← Client │ 工具调用开始             │
│ tool_result         │ ← Client │ 工具调用结果             │
│ session_start       │ ← Client │ 会话创建确认            │
│ session_end         │ ← Client │ 会话结束                │
│ heartbeat           │ ↔ Both   │ 心跳保活                 │
│ error               │ ← Client │ 错误通知                 │
└─────────────────────┴──────────┴──────────────────────────┘
"""

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class MessageType(str, Enum):
    # 客户端 → 服务端
    CLIENT_MESSAGE = "client_message"
    HEARTBEAT_PING = "heartbeat_ping"

    # 服务端 → 客户端
    SERVER_MESSAGE = "server_message"       # 完整回复
    SERVER_CHUNK = "server_chunk"           # 流式片段
    TOOL_CALL = "tool_call"                 # 工具调用开始
    TOOL_RESULT = "tool_result"             # 工具调用结果
    SESSION_START = "session_start"         # 会话创建确认
    SESSION_END = "session_end"             # 会话结束
    HEARTBEAT_PONG = "heartbeat_pong"       # 心跳响应
    ERROR = "error"                         # 错误通知


@dataclass
class WSError:
    """WebSocket 错误"""
    code: str
    message: str


@dataclass
class WSMessage:
    """WebSocket 消息"""

    type: MessageType
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: float = field(default_factory=time.time)
    payload: dict[str, Any] = field(default_factory=dict)
    error: WSError | None = None

    def to_json(self) -> str:
        data = {
            "type": self.type.value,
            "id": self.id,
            "timestamp": self.timestamp,
            "payload": self.payload,
            "error": asdict(self.error) if self.error else None,
        }
        return json.dumps(data, ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> "WSMessage":
        data = json.loads(raw)
        error_data = data.get("error")
        return cls(
            type=MessageType(data["type"]),
            id=data["id"],
            timestamp=data["timestamp"],
            payload=data.get("payload", {}),
            error=WSError(**error_data) if error_data else None,
        )

    @classmethod
    def client_message(cls, session_id: str, content: str) -> "WSMessage":
        return cls(
            type=MessageType.CLIENT_MESSAGE,
            payload={"session_id": session_id, "content": content},
        )

    @classmethod
    def server_chunk(cls, session_id: str, chunk: str, is_final: bool = False) -> "WSMessage":
        return cls(
            type=MessageType.SERVER_CHUNK,
            payload={
                "session_id": session_id,
                "chunk": chunk,
                "is_final": is_final,
            },
        )

    @classmethod
    def server_message(cls, session_id: str, content: str) -> "WSMessage":
        return cls(
            type=MessageType.SERVER_MESSAGE,
            payload={"session_id": session_id, "content": content},
        )

    @classmethod
    def tool_call_msg(cls, session_id: str, tool_name: str, params: dict) -> "WSMessage":
        return cls(
            type=MessageType.TOOL_CALL,
            payload={
                "session_id": session_id,
                "tool_name": tool_name,
                "params": params,
            },
        )

    @classmethod
    def tool_result_msg(cls, session_id: str, tool_name: str, result: str) -> "WSMessage":
        return cls(
            type=MessageType.TOOL_RESULT,
            payload={
                "session_id": session_id,
                "tool_name": tool_name,
                "result": result,
            },
        )

    @classmethod
    def error_msg(cls, code: str, message: str) -> "WSMessage":
        return cls(
            type=MessageType.ERROR,
            error=WSError(code=code, message=message),
        )
```

---

## 4. WebSocket Handler `websocket/handler.py`

```python
"""
WebSocket 消息处理器。

FastAPI WebSocket endpoint，处理客户端连接和消息路由。
"""

import asyncio
import logging
import uuid

from websocket.connection import ConnectionManager
from websocket.protocol import MessageType, WSMessage

logger = logging.getLogger(__name__)


class WebSocketHandler:
    """WebSocket 消息处理器"""

    def __init__(self, connection_manager: ConnectionManager):
        self._manager = connection_manager

    async def handle_connection(self, websocket, session_id: str, user_id: str):
        """处理 WebSocket 连接生命周期"""
        await websocket.accept()
        connection_id = uuid.uuid4().hex[:12]

        conn = await self._manager.connect(
            connection_id=connection_id,
            session_id=session_id,
            user_id=user_id,
            websocket=websocket,
        )

        # 发送会话开始确认
        start_msg = WSMessage(
            type=MessageType.SESSION_START,
            payload={"session_id": session_id, "connection_id": connection_id},
        )
        await websocket.send_text(start_msg.to_json())

        logger.info(f"WebSocket connected: {connection_id} → session={session_id}")

        try:
            async for raw_message in websocket.iter_text():
                await self._handle_message(connection_id, raw_message)
        except Exception as e:
            logger.error(f"WebSocket error on {connection_id}: {e}")
        finally:
            await self._manager.disconnect(connection_id)

    async def _handle_message(self, connection_id: str, raw: str):
        """处理收到的消息"""
        try:
            message = WSMessage.from_json(raw)
        except Exception as e:
            logger.warning(f"Invalid message from {connection_id}: {e}")
            return

        if message.type == MessageType.HEARTBEAT_PING:
            pong = WSMessage(type=MessageType.HEARTBEAT_PONG, payload={"time": message.timestamp})
            await self._broadcast_to_session(
                message.payload.get("session_id", ""), pong
            )
            return

        if message.type == MessageType.CLIENT_MESSAGE:
            session_id = message.payload.get("session_id")
            content = message.payload.get("content", "")

            # 转发到 Agent 引擎（通过事件总线）
            from core.event_bus import get_event_bus
            bus = get_event_bus()
            await bus.emit(
                "agent.message",
                session_id=session_id,
                user_message=content,
                connection_id=connection_id,
            )

    async def send_to_session(self, session_id: str, message: WSMessage):
        """向指定会话的所有连接发送消息"""
        connections = self._manager.get_by_session(session_id)
        for conn in connections:
            try:
                await conn.websocket.send_text(message.to_json())
            except Exception as e:
                logger.warning(f"Send failed to {conn.connection_id}: {e}")

    async def broadcast_to_user(self, user_id: str, message: WSMessage):
        """向指定用户的所有连接广播消息"""
        connections = self._manager.get_by_user(user_id)
        for conn in connections:
            try:
                await conn.websocket.send_text(message.to_json())
            except Exception as e:
                logger.warning(f"Broadcast failed to {conn.connection_id}: {e}")

    async def _broadcast_to_session(self, session_id: str, message: WSMessage):
        """内部广播方法"""
        await self.send_to_session(session_id, message)


# 全局 handler
default_handler = WebSocketHandler(ConnectionManager())
```

---

## 5. SSE Fallback `websocket/sse.py`

```python
"""
Server-Sent Events (SSE) fallback。

当 WebSocket 不可用时，使用 SSE 提供单向流式推送。
"""

import asyncio
import json
import logging
import time
from typing import Any, AsyncGenerator

logger = logging.getLogger(__name__)


class SSESender:
    """SSE 消息发送器"""

    def __init__(self):
        self._subscribers: dict[str, asyncio.Queue] = {}

    def subscribe(self, session_id: str) -> asyncio.Queue:
        """订阅某个会话的 SSE 流"""
        queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self._subscribers[session_id] = queue
        return queue

    def unsubscribe(self, session_id: str):
        """取消订阅"""
        self._subscribers.pop(session_id, None)

    async def publish(self, session_id: str, data: dict[str, Any], event: str | None = None):
        """发布消息到会话的所有订阅者"""
        queue = self._subscribers.get(session_id)
        if not queue:
            return

        try:
            await queue.put({
                "event": event or "message",
                "data": json.dumps(data, ensure_ascii=False),
                "id": int(time.time() * 1000),
            })
        except asyncio.QueueFull:
            logger.warning(f"SSE queue full for session {session_id}")

    def format_sse(self, item: dict[str, Any]) -> str:
        """格式化为 SSE 文本"""
        lines = []
        if item.get("id"):
            lines.append(f"id: {item['id']}")
        if item.get("event"):
            lines.append(f"event: {item['event']}")
        lines.append(f"data: {item['data']}")
        return "\n".join(lines) + "\n\n"

    async def stream(self, session_id: str) -> AsyncGenerator[str, None]:
        """SSE 流式输出"""
        queue = self._subscribers.get(session_id)
        if not queue:
            yield f"data: {{\"error\": \"No subscription for {session_id}\"}}\n\n"
            return

        try:
            while True:
                item = await queue.get()
                yield self.format_sse(item)
        except asyncio.CancelledError:
            pass
        finally:
            self.unsubscribe(session_id)


# 全局 SSE sender
default_sse_sender = SSESender()
```

---

## 6. Agent 流式输出集成 `websocket/streaming.py`

```python
"""
Agent 流式输出到 WebSocket / SSE。

将 Agent 的思考过程、工具调用、最终回复实时推送到客户端。
"""

import asyncio
import logging
from typing import Any, AsyncGenerator

from websocket.protocol import MessageType, WSMessage

logger = logging.getLogger(__name__)


class StreamDispatcher:
    """流式消息分发器"""

    def __init__(self, handler, sse_sender=None):
        self._handler = handler
        self._sse_sender = sse_sender

    async def dispatch_agent_response(
        self, session_id: str, content_chunks: AsyncGenerator[str, None]
    ):
        """分发 Agent 的流式回复"""
        full_content = []

        async for chunk in content_chunks:
            full_content.append(chunk)

            # WebSocket 推送
            ws_msg = WSMessage.server_chunk(
                session_id=session_id, chunk=chunk, is_final=False
            )
            await self._handler.send_to_session(session_id, ws_msg)

            # SSE fallback 推送
            if self._sse_sender:
                await self._sse_sender.publish(
                    session_id, {"content": chunk}, event="chunk"
                )

        # 发送最终消息
        final_content = "".join(full_content)
        final_ws_msg = WSMessage.server_message(session_id=session_id, content=final_content)
        await self._handler.send_to_session(session_id, final_ws_msg)

    async def dispatch_tool_call(self, session_id: str, tool_name: str, params: dict):
        """分发工具调用事件"""
        msg = WSMessage.tool_call_msg(
            session_id=session_id, tool_name=tool_name, params=params
        )
        await self._handler.send_to_session(session_id, msg)

    async def dispatch_tool_result(self, session_id: str, tool_name: str, result: str):
        """分发工具调用结果"""
        msg = WSMessage.tool_result_msg(
            session_id=session_id, tool_name=tool_name, result=result
        )
        await self._handler.send_to_session(session_id, msg)

    async def dispatch_error(self, session_id: str, code: str, message: str):
        """分发错误事件"""
        msg = WSMessage.error_msg(code=code, message=message)
        await self._handler.send_to_session(session_id, msg)


# 全局 dispatcher
default_dispatcher = StreamDispatcher(
    handler=None,  # Will be set during initialization
    sse_sender=None,
)
```

---

## 7. YAML 配置 `config/websocket.yaml`

```yaml
websocket:
  enabled: true
  host: "0.0.0.0"
  port: 8765                    # WebSocket 专用端口（或复用 HTTP）
  path: "/ws/chat"              # WebSocket endpoint 路径

  heartbeat:
    interval_seconds: 30        # 心跳间隔
    timeout_seconds: 90         # 超时断开时间

  connection:
    max_per_user: 5             # 单用户最大连接数
    max_total: 10000            # 全局最大连接数

  sse_fallback:
    enabled: true               # WebSocket 不可用时启用 SSE
    path: "/sse/chat"           # SSE endpoint 路径
    queue_size: 1000            # 每个会话的消息队列大小
```

---

## 8. 架构总览

```
                    ┌─────────────────────┐
                    │   WebSocket Client   │
                    │   (Browser / App)    │
                    └──────────┬──────────┘
                               │ JSON messages
              ┌────────────────┼────────────────┐
              ▼                ▼                 ▼
   ┌────────────────┐ ┌─────────────┐  ┌────────────────┐
   │ ConnectionMgr  │ │ WS Handler  │  │ SSE Fallback   │
   │               │ │             │  │                │
   │ • conn pool   │ │ • msg parse │  │ • event stream │
   │ • heartbeat   │ │ • routing   │  │ • queue-based  │
   │ • timeout     │ │ • dispatch  │  └────────────────┘
   └───────┬───────┘ └──────┬──────┘
           │                │
           ▼                ▼
   ┌──────────────────────────────────┐
   │      StreamDispatcher            │
   │                                 │
   │  Agent chunks → WS / SSE push   │
   │  Tool calls → real-time events   │
   └───────────────┬─────────────────┘
                   │
                   ▼
          ┌────────────────┐
          │    Agent Core   │
          │  (streaming)   │
          └────────────────┘
```

---

## 9. 设计总结

| 特性 | 实现方式 |
|------|---------|
| **连接管理** | ConnectionManager 维护连接池，心跳检测 + 超时清理 |
| **消息协议** | 统一 JSON 格式，8 种消息类型覆盖完整交互流程 |
| **流式输出** | StreamDispatcher 将 Agent chunk / tool_call 实时推送 |
| **SSE Fallback** | Queue-based SSE，WebSocket 不可用时自动降级 |
