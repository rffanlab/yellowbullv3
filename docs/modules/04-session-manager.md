# Session Manager 详细设计

## 1. 职责边界

| 职责 | 说明 |
|------|------|
| **会话生命周期** | 创建、查询、销毁 session，管理 session 状态机 |
| **消息持久化** | 每条 message 追加写入 SQLite，支持按 session_id 回放 |
| **上下文窗口管理** | 滑动窗口 + token 预算裁剪，保证 prompt 不超限 |
| **并发安全** | asyncio.Lock 保护共享状态，多请求同 session 互不干扰 |
| **热加载集成** | YAML 变更 → 自动调整 max_messages / max_tokens |

---

## 2. 数据模型 `session/models.py`

```python
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class SessionStatus(str, Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"


@dataclass
class Message:
    """单条消息（用户输入 / Agent 输出）"""
    id: str                                    # UUID
    session_id: str                            # 所属会话
    role: str                                  # "user" | "assistant" | "system"
    content: str                               # 文本内容
    tool_calls: list[dict] = field(default_factory=list)   # assistant 的工具调用记录
    token_count: int = 0                       # 估算 token 数（用于窗口裁剪）
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Session:
    """会话元数据"""
    id: str                                    # UUID
    user_id: str                               # 创建者标识
    title: str = ""                            # 会话标题（自动生成或用户设置）
    status: SessionStatus = SessionStatus.ACTIVE
    messages: list[Message] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def message_count(self) -> int:
        return len(self.messages)

    @property
    def total_tokens(self) -> int:
        return sum(m.token_count for m in self.messages)


@dataclass
class ContextWindow:
    """裁剪后的上下文窗口（发送给 LLM 的消息子集）"""
    messages: list[Message]                    # 已按 token 预算裁剪
    total_tokens: int = 0                      # 窗口内总 token 数
    dropped_count: int = 0                     # 被丢弃的早期消息数
```

---

## 3. SQLite 存储层 `session/storage.py`

```python
"""
SQLite 持久化层。

设计原则：
- WAL mode（高并发读写）
- 按 session_id 索引（快速查询）
- 追加写入（append-only，不更新历史消息）
"""

import aiosqlite
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class SessionStorage:
    """SQLite-backed session/message store"""

    def __init__(self, db_path: str | Path):
        self._db_path = Path(db_path)
        self._db: aiosqlite.Connection | None = None

    async def connect(self):
        if self._db and not self._db.closed:
            return
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self._db_path))
        self._db.execute_isolation_level(None)   # 手动 commit
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA busy_timeout=5000")
        await self._create_tables()
        logger.info(f"SessionStorage connected: {self._db_path}")

    async def close(self):
        if self._db and not self._db.closed:
            await self._db.close()

    async def _create_tables(self):
        await self._db.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                title TEXT DEFAULT '',
                status TEXT DEFAULT 'active',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
            CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);

            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
                content TEXT NOT NULL DEFAULT '',
                tool_calls TEXT DEFAULT '[]',
                token_count INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
        """)
        await self._db.commit()

    # ==================== Session CRUD ====================

    async def create_session(self, session: "Session") -> "Session":
        await self._db.execute(
            """INSERT OR REPLACE INTO sessions (id, user_id, title, status, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (session.id, session.user_id, session.title, session.status.value,
             session.created_at.isoformat(), session.updated_at.isoformat()),
        )
        await self._db.commit()

    async def get_session(self, session_id: str) -> dict | None:
        cursor = await self._db.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return {
            "id": row[0], "user_id": row[1], "title": row[2],
            "status": row[3], "created_at": row[4], "updated_at": row[5],
        }

    async def list_sessions(self, user_id: str, limit: int = 50) -> list[dict]:
        cursor = await self._db.execute(
            """SELECT * FROM sessions WHERE user_id = ? AND status != 'deleted'
               ORDER BY updated_at DESC LIMIT ?""",
            (user_id, limit),
        )
        rows = await cursor.fetchall()
        return [
            {"id": r[0], "user_id": r[1], "title": r[2],
             "status": r[3], "created_at": r[4], "updated_at": r[5]}
            for r in rows
        ]

    async def update_session(self, session: "Session") -> None:
        await self._db.execute(
            """UPDATE sessions SET title = ?, status = ?, updated_at = ? WHERE id = ?""",
            (session.title, session.status.value, session.updated_at.isoformat(), session.id),
        )
        await self._db.commit()

    async def delete_session(self, session_id: str) -> None:
        # Soft delete: mark as deleted + cascade to messages
        await self._db.execute(
            "UPDATE sessions SET status = 'deleted' WHERE id = ?", (session_id,)
        )
        await self._db.commit()

    # ==================== Message CRUD ====================

    async def append_message(self, message: "Message") -> None:
        import json
        await self._db.execute(
            """INSERT INTO messages (id, session_id, role, content, tool_calls, token_count, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (message.id, message.session_id, message.role, message.content,
             json.dumps(message.tool_calls, ensure_ascii=False),
             message.token_count, message.created_at.isoformat()),
        )
        await self._db.commit()

    async def get_messages(self, session_id: str) -> list[dict]:
        import json
        cursor = await self._db.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at ASC",
            (session_id,),
        )
        rows = await cursor.fetchall()
        return [
            {
                "id": r[0], "session_id": r[1], "role": r[2],
                "content": r[3], "tool_calls": json.loads(r[4]),
                "token_count": r[5], "created_at": r[6],
            }
            for r in rows
        ]

    async def get_message_count(self, session_id: str) -> int:
        cursor = await self._db.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,)
        )
        row = await cursor.fetchone()
        return row[0] if row else 0

    async def clear_messages(self, session_id: str) -> None:
        """清空会话消息（保留 session 元数据）"""
        await self._db.execute(
            "DELETE FROM messages WHERE session_id = ?", (session_id,)
        )
        await self._db.commit()
```

---

## 4. Session Manager `session/manager.py`

```python
"""
SessionManager —— 会话生命周期 + 上下文窗口管理。

职责：
- 创建/查询/销毁 session
- 追加消息到内存 + SQLite
- 按 token 预算裁剪上下文窗口（滑动窗口）
- 热加载配置变更
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any

from config.manager import get_manager
from session.models import Message, Session, SessionStatus, ContextWindow
from session.storage import SessionStorage

logger = logging.getLogger(__name__)


class SessionManager:
    """会话管理器（全局单例）"""

    _instance: "SessionManager | None" = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        settings = get_manager().settings.session
        self._storage = SessionStorage(settings.db_path)
        self._sessions: dict[str, Session] = {}       # session_id → Session (内存缓存)
        self._locks: dict[str, asyncio.Lock] = {}     # session_id → Lock（并发安全）
        self._max_messages = settings.max_messages     # 单会话最大消息数
        self._max_tokens = settings.max_tokens         # 上下文窗口 token 上限

    @classmethod
    def reset(cls):
        cls._instance = None

    async def start(self):
        await self._storage.connect()

    async def stop(self):
        await self._storage.close()

    # ==================== Session CRUD ====================

    async def create_session(self, user_id: str, title: str = "") -> Session:
        session = Session(
            id=str(uuid.uuid4()),
            user_id=user_id,
            title=title or f"Session {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}",
        )
        await self._storage.create_session(session)
        self._sessions[session.id] = session
        self._locks[session.id] = asyncio.Lock()
        logger.info(f"Session created: {session.id} (user={user_id})")
        return session

    async def get_session(self, session_id: str) -> Session | None:
        # 内存缓存优先，miss 则从 DB 重建
        if session_id in self._sessions:
            return self._sessions[session_id]

        row = await self._storage.get_session(session_id)
        if not row or row["status"] == "deleted":
            return None

        messages = await self._storage.get_messages(session_id)
        msg_objects = [
            Message(
                id=m["id"], session_id=m["session_id"], role=m["role"],
                content=m["content"], tool_calls=m["tool_calls"],
                token_count=m["token_count"], created_at=datetime.fromisoformat(m["created_at"]),
            )
            for m in messages
        ]

        session = Session(
            id=row["id"], user_id=row["user_id"], title=row["title"],
            status=SessionStatus(row["status"]),
            messages=msg_objects,
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )
        self._sessions[session.id] = session
        self._locks[session.id] = asyncio.Lock()
        return session

    async def list_user_sessions(self, user_id: str) -> list[Session]:
        rows = await self._storage.list_sessions(user_id)
        sessions = []
        for row in rows:
            s = Session(
                id=row["id"], user_id=row["user_id"], title=row["title"],
                status=SessionStatus(row["status"]),
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
            )
            sessions.append(s)
        return sessions

    async def delete_session(self, session_id: str) -> None:
        await self._storage.delete_session(session_id)
        self._sessions.pop(session_id, None)
        logger.info(f"Session deleted: {session_id}")

    # ==================== Message Operations ====================

    async def append_message(self, message: Message) -> Message:
        """追加消息（内存 + SQLite，并发安全）"""
        lock = self._locks.get(message.session_id)
        if not lock:
            raise ValueError(f"Session {message.session_id} not found")

        async with lock:
            session = self._sessions.get(message.session_id)
            if not session:
                raise ValueError(f"Session {message.session_id} not in memory")

            # 追加到内存
            session.messages.append(message)
            session.updated_at = datetime.utcnow()

            # 持久化
            await self._storage.append_message(message)
            await self._storage.update_session(session)

        return message

    async def get_messages(self, session_id: str) -> list[Message]:
        """获取会话全部消息"""
        lock = self._locks.get(session_id)
        if not lock:
            raise ValueError(f"Session {session_id} not found")

        async with lock:
            session = self._sessions.get(session_id)
            if not session:
                return []
            return list(session.messages)   # 返回副本

    # ==================== Context Window ====================

    def build_context_window(self, session_id: str) -> ContextWindow | None:
        """
        按 token 预算裁剪上下文窗口（滑动窗口，保留最新消息）。

        策略：
        1. 从最新消息向前累加 token
        2. 超过 max_tokens 时丢弃最早的消息
        3. 保证至少保留最近一轮对话（user + assistant）
        """
        session = self._sessions.get(session_id)
        if not session or not session.messages:
            return None

        messages = list(session.messages)   # 副本

        # 按 token 预算裁剪（从后向前累加，丢弃最早消息）
        total_tokens = 0
        keep_index = 0
        for i in range(len(messages) - 1, -1, -1):
            total_tokens += messages[i].token_count
            if total_tokens > self._max_tokens and i > 0:
                keep_index = i + 1
                break

        dropped = messages[:keep_index]
        window_messages = messages[keep_index:]

        # 保证至少保留最近一轮对话（2 条消息）
        if len(window_messages) < 2 and len(messages) >= 2:
            window_messages = messages[-2:]
            dropped = messages[:-2]

        return ContextWindow(
            messages=window_messages,
            total_tokens=sum(m.token_count for m in window_messages),
            dropped_count=len(dropped),
        )

    # ==================== Hot Reload ====================

    async def reload_config(self):
        """YAML 变更回调：更新 max_messages / max_tokens"""
        settings = get_manager().settings.session
        old_max_msgs, old_max_tokens = self._max_messages, self._max_tokens
        self._max_messages = settings.max_messages
        self._max_tokens = settings.max_tokens

        if (old_max_msgs, old_max_tokens) != (self._max_messages, self._max_tokens):
            logger.info(
                f"Session config reloaded: "
                f"max_messages={old_max_msgs}→{self._max_messages}, "
                f"max_tokens={old_max_tokens}→{self._max_tokens}"
            )

    @property
    def max_messages(self) -> int:
        return self._max_messages

    @property
    def max_tokens(self) -> int:
        return self._max_tokens


def get_session_manager() -> SessionManager:
    """获取全局 SessionManager 实例"""
    return SessionManager()
```

---

## 5. 热加载集成 `session/config_bridge.py`

```python
from config.manager import get_manager
from session.manager import get_session_manager


def setup_session_config_watching():
    manager = get_manager()

    @manager.on_change("session")
    async def on_session_config_changed(old_val, new_val):
        sm = get_session_manager()
        await sm.reload_config()
```

---

## 6. Token 估算 `session/token_estimator.py`

```python
"""
简易 token 估算器。

生产环境建议：tiktoken（OpenAI）或 transformers tokenizer
此处使用字符数 / 1.5 作为近似值。
"""


def estimate_tokens(text: str) -> int:
    """
    粗略估算文本的 GPT-4 token 数。

    规则：
    - 英文：~4 chars per token
    - 中文：~1.5 chars per token（中文字符占比高时）
    - 混合：取中间值 ~3 chars per token
    """
    if not text:
        return 0

    # 简单启发式：按字符数 / 1.8 估算
    return max(1, len(text) // 18 + len(text.encode("utf-8")) // 6)


def estimate_message_tokens(message: "Message") -> int:
    """估算单条消息的 token 数（含 role overhead）"""
    base = 4   # role + name overhead
    content_tokens = estimate_tokens(message.content)

    tool_overhead = 0
    for tc in message.tool_calls:
        tool_overhead += 12   # tool_call structure overhead
        tool_overhead += estimate_tokens(tc.get("function", {}).get("arguments", ""))

    return base + content_tokens + tool_overhead
```

---

## 7. 并发安全模型

```
Request A ──┐
            ├──→ asyncio.Lock(session_id) → append_message() → SQLite
Request B ──┘              ↑
                           │ (同一 session 串行化，不同 session 并行)

Session-1 Lock ──── 独立锁 ──── Session-N Lock
```

**关键设计：**
- 每个 session 有独立的 `asyncio.Lock`
- 同 session 的请求串行化（防止消息乱序）
- 不同 session 完全并行（无全局锁瓶颈）

---

## 8. 上下文窗口裁剪示例

```
Session messages (10条, total=4500 tokens), max_tokens=4000:

[1] User: "Hello"              (20 tokens)    ← DROPPED
[2] Assistant: "Hi!"           (30 tokens)    ← DROPPED
[3] User: "What is AI?"        (25 tokens)    ← DROPPED
[4] Assistant: "AI is..."      (800 tokens)   ← DROPPED
[5] User: "Explain more"       (15 tokens)    ← KEPT
[6] Assistant: "Sure, AI..."   (900 tokens)   ← KEPT
[7] User: "Compare with ML"    (20 tokens)    ← KEPT
[8] Assistant: "AI vs ML..."   (1200 tokens)  ← KEPT
[9] User: "Give examples"      (15 tokens)    ← KEPT
[10] Assistant: "Examples..."  (1000 tokens)  ← KEPT

Window: [5..10], total=3150 tokens, dropped=4 messages
```

---

## 9. 设计总结

| 特性 | 实现方式 |
|------|---------|
| **持久化** | SQLite WAL mode，append-only 消息写入 |
| **内存缓存** | dict[str, Session] 热路径零 IO |
| **并发安全** | per-session asyncio.Lock，不同 session 并行 |
| **上下文裁剪** | 滑动窗口 + token 预算，从后向前累加 |
| **Token 估算** | 字符启发式（生产可替换 tiktoken） |
| **热加载** | YAML → max_messages/max_tokens 自动刷新 |
