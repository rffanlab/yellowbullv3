# 记忆系统详细设计（Memory System）

## 1. 职责边界

| 层级 | 说明 | TTL/容量 |
|------|------|---------|
| **短期记忆** | 会话内上下文，随对话推进自动裁剪 | 受 token 窗口限制 |
| **工作记忆** | Agent 推理过程中的临时状态（工具结果、中间结论） | 单次任务生命周期 |
| **长期记忆** | 跨会话持久化：用户偏好、关键事实、经验教训 | 无限，支持遗忘机制 |

---

## 2. 短期记忆 `memory/short_term.py`

```python
"""
短期记忆管理。

职责：
- 维护对话消息窗口（role + content）
- Token 预算控制：超出限制时自动裁剪最早的消息
- 摘要压缩：将旧对话压缩为摘要注入系统提示
"""

import logging
from dataclasses import dataclass, field
from typing import Protocol

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """单条消息"""
    role: str           # "user" | "assistant" | "system" | "tool"
    content: str
    token_count: int = 0
    timestamp: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class MemoryState:
    """记忆状态"""
    messages: list[Message] = field(default_factory=list)
    summary: str = ""               # 历史对话摘要（压缩后的旧消息）
    total_tokens: int = 0
    message_count: int = 0


class ShortTermMemory:
    """
    短期记忆管理器。

    Usage:
        memory = ShortTermMemory(max_tokens=4000, reserve_tokens=1000)
        memory.add_message(Message(role="user", content="Hello"))
        messages = memory.get_messages()          # 用于 LLM API
        summary = memory.get_summary_prompt()     # 压缩后的历史摘要
    """

    def __init__(self, max_tokens: int = 4000, reserve_tokens: int = 1000):
        self._max_tokens = max_tokens
        self._reserve = reserve_tokens            # 为回复预留的 token
        self._messages: list[Message] = []
        self._summary_parts: list[str] = []       # 已压缩的历史摘要片段

    @property
    def state(self) -> MemoryState:
        total = sum(m.token_count for m in self._messages)
        return MemoryState(
            messages=list(self._messages),
            summary="\n".join(self._summary_parts),
            total_tokens=total,
            message_count=len(self._messages),
        )

    def add_message(self, message: Message):
        """添加消息，自动裁剪超出 token 窗口的旧消息"""
        self._messages.append(message)
        self._evict_if_needed()

    def _estimate_tokens(self, text: str) -> int:
        """粗略估算 token 数（1 token ≈ 4 chars for Chinese/English mix）"""
        return max(1, len(text) // 4)

    def _evict_if_needed(self):
        """Token 超限时，将最早的消息压缩为摘要"""
        total = sum(m.token_count for m in self._messages)
        budget = self._max_tokens - self._reserve

        while total > budget and len(self._messages) > 2:
            # 取最早的 4 条消息进行压缩（保留最近对话）
            batch_size = min(4, len(self._messages) - 2)
            old_messages = self._messages[:batch_size]
            self._messages = self._messages[batch_size:]

            # 将旧消息摘要化
            summary_text = self._summarize_batch(old_messages)
            if summary_text:
                self._summary_parts.append(summary_text)

            total = sum(m.token_count for m in self._messages)

    def _summarize_batch(self, messages: list[Message]) -> str:
        """
        将一批消息压缩为摘要。

        注意：这里使用简单的规则摘要，生产环境应调用 LLM 生成摘要。
        """
        parts = []
        for m in messages:
            role_label = "User" if m.role == "user" else "Assistant" if m.role == "assistant" else m.role
            # 截断过长内容
            content = m.content[:200] + ("..." if len(m.content) > 200 else "")
            parts.append(f"{role_label}: {content}")

        return f"[Earlier conversation]\n{' | '.join(parts)}"

    def get_messages(self) -> list[dict]:
        """获取用于 LLM API 的消息列表"""
        result = []
        # 如果有历史摘要，注入为 system message
        if self._summary_parts:
            summary = "\n".join(self._summary_parts)
            result.append({
                "role": "system",
                "content": f"Previous conversation summary:\n{summary}",
            })

        for m in self._messages:
            msg = {"role": m.role, "content": m.content}
            if m.metadata.get("tool_calls"):
                msg["tool_calls"] = m.metadata["tool_calls"]
            if m.metadata.get("tool_call_id"):
                msg["tool_call_id"] = m.metadata["tool_call_id"]
            result.append(msg)

        return result

    def get_summary_prompt(self) -> str:
        """获取历史摘要（用于注入系统提示）"""
        return "\n".join(self._summary_parts) if self._summary_parts else ""

    def clear(self):
        """清空记忆（新会话开始时调用）"""
        self._messages.clear()
        self._summary_parts.clear()

    def trim_to_last_n(self, n: int):
        """只保留最近 N 条消息"""
        if len(self._messages) > n:
            removed = self._messages[:-n]
            summary = self._summarize_batch(removed)
            if summary:
                self._summary_parts.append(summary)
            self._messages = self._messages[-n:]

    def peek_last_user_message(self) -> Message | None:
        """查看最近一条用户消息"""
        for m in reversed(self._messages):
            if m.role == "user":
                return m
        return None
```

---

## 3. LLM 摘要压缩 `memory/llm_summarizer.py`

```python
"""
使用 LLM 对旧对话进行高质量摘要。

替代规则摘要，生成更准确、信息密度更高的历史摘要。
"""

import logging
from llm.provider import BaseLLMProvider

logger = logging.getLogger(__name__)

SUMMARIZE_SYSTEM_PROMPT = """你是一个专业的对话摘要助手。
请将以下对话内容压缩为简洁的要点式摘要。
要求：
1. 保留关键事实、决策和结论
2. 省略寒暄、重复内容
3. 使用第三人称叙述
4. 控制在 100 字以内"""


class LLMSummarizer:
    """LLM 驱动的对话摘要器"""

    def __init__(self, llm_provider: BaseLLMProvider):
        self._llm = llm_provider

    async def summarize(self, messages: list[dict]) -> str:
        """将消息列表压缩为摘要"""
        conversation = "\n".join(
            f"{m['role']}: {m['content']}" for m in messages
        )

        try:
            response = await self._llm.chat(
                system_prompt=SUMMARIZE_SYSTEM_PROMPT,
                user_message=f"请摘要以下对话：\n\n{conversation}",
                max_tokens=200,
                temperature=0.3,
            )
            return response.content.strip()
        except Exception as e:
            logger.warning(f"LLM summarization failed, falling back to rule-based: {e}")
            # 降级为规则摘要
            parts = []
            for m in messages[:4]:
                content = m["content"][:100] + ("..." if len(m["content"]) > 100 else "")
                parts.append(f"{m['role']}: {content}")
            return " | ".join(parts)

    async def summarize_incremental(
        self, existing_summary: str, new_messages: list[dict]
    ) -> str:
        """
        增量摘要：将新消息合并到已有摘要中。

        避免每次都重新处理全部历史，节省 token 成本。
        """
        conversation = "\n".join(
            f"{m['role']}: {m['content']}" for m in new_messages
        )

        prompt = f"""现有对话摘要：
{existing_summary}

新增对话内容：
{conversation}

请更新摘要，保留所有关键信息，控制在 150 字以内。"""

        try:
            response = await self._llm.chat(
                system_prompt=SUMMARIZE_SYSTEM_PROMPT,
                user_message=prompt,
                max_tokens=300,
                temperature=0.3,
            )
            return response.content.strip()
        except Exception as e:
            logger.warning(f"Incremental summarization failed: {e}")
            return f"{existing_summary}\n[New] {' | '.join(m['content'][:50] for m in new_messages)}"
```

---

## 4. 长期记忆 `memory/long_term.py`

```python
"""
长期记忆系统。

职责：
- 事实存储（Fact Store）：用户偏好、关键信息
- 经验存储（Episode Store）：重要对话片段，支持语义检索
- 技能存储（Skill Store）：Agent 学到的操作模式
- 遗忘机制：低重要性记忆自动衰减
"""

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """长期记忆条目"""
    id: str
    memory_type: str       # "fact" | "episode" | "skill"
    content: str           # 记忆内容（结构化文本）
    importance: float      # [0.0, 1.0] 重要性评分
    tags: list[str]        # 标签，用于检索过滤
    created_at: float      # 创建时间戳
    last_accessed: float   # 最后访问时间戳
    access_count: int = 0  # 访问次数
    metadata: dict = field(default_factory=dict)


class BaseMemoryStore(Protocol):
    """记忆存储后端协议"""

    async def save(self, entry: MemoryEntry) -> bool: ...
    async def get(self, memory_id: str) -> MemoryEntry | None: ...
    async def search(
        self, query: str, memory_type: str | None = None, top_k: int = 5
    ) -> list[MemoryEntry]: ...
    async def delete(self, memory_id: str) -> bool: ...
    async def list_all(
        self, memory_type: str | None = None, tag: str | None = None
    ) -> list[MemoryEntry]: ...
    async def update_access(self, memory_id: str) -> bool: ...


class JSONFileStore(BaseMemoryStore):
    """JSON 文件存储（轻量级，适合开发/个人使用）"""

    def __init__(self, storage_path: str = "./data/memory"):
        self._path = Path(storage_path)
        self._path.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    async def save(self, entry: MemoryEntry) -> bool:
        async with self._lock:
            file_path = self._path / f"{entry.id}.json"
            file_path.write_text(json.dumps(asdict(entry), ensure_ascii=False, indent=2))
        return True

    async def get(self, memory_id: str) -> MemoryEntry | None:
        file_path = self._path / f"{memory_id}.json"
        if not file_path.exists():
            return None
        data = json.loads(file_path.read_text())
        return MemoryEntry(**data)

    async def search(
        self, query: str, memory_type: str | None = None, top_k: int = 5
    ) -> list[MemoryEntry]:
        """简单关键词搜索（生产环境应使用向量检索）"""
        entries = await self.list_all(memory_type=memory_type)
        query_lower = query.lower()

        scored = []
        for entry in entries:
            score = 0.0
            if any(word in entry.content.lower() for word in query_lower.split()):
                score += 0.5
            if any(tag in query_lower for tag in entry.tags):
                score += 0.3
            # 访问频率加分
            score += min(0.2, entry.access_count * 0.01)
            # 时间衰减
            age_days = (time.time() - entry.created_at) / 86400
            score *= max(0.5, 1.0 - age_days * 0.01)

            if score > 0:
                scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:top_k]]

    async def delete(self, memory_id: str) -> bool:
        async with self._lock:
            file_path = self._path / f"{memory_id}.json"
            if file_path.exists():
                file_path.unlink()
                return True
        return False

    async def list_all(
        self, memory_type: str | None = None, tag: str | None = None
    ) -> list[MemoryEntry]:
        entries = []
        for file_path in self._path.glob("*.json"):
            try:
                data = json.loads(file_path.read_text())
                entry = MemoryEntry(**data)
                if memory_type and entry.memory_type != memory_type:
                    continue
                if tag and tag not in entry.tags:
                    continue
                entries.append(entry)
            except Exception:
                continue
        return entries

    async def update_access(self, memory_id: str) -> bool:
        entry = await self.get(memory_id)
        if entry:
            entry.last_accessed = time.time()
            entry.access_count += 1
            await self.save(entry)
            return True
        return False


class SQLiteMemoryStore(BaseMemoryStore):
    """SQLite 存储（生产级，支持全文检索）"""

    def __init__(self, db_path: str = "./data/memory.db"):
        import aiosqlite
        self._db_path = db_path
        self._lock = asyncio.Lock()
        self._initialize_db()

    def _initialize_db(self):
        """初始化数据库表（同步操作，在构造时执行）"""
        import sqlite3
        conn = sqlite3.connect(self._db_path)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                memory_type TEXT NOT NULL,
                content TEXT NOT NULL,
                importance REAL NOT NULL DEFAULT 0.5,
                tags TEXT NOT NULL DEFAULT '[]',
                created_at REAL NOT NULL,
                last_accessed REAL NOT NULL,
                access_count INTEGER NOT NULL DEFAULT 0,
                metadata TEXT NOT NULL DEFAULT '{}'
            )
        """)
        c.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
            USING fts5(content, tags, content='memories', content_rowid='rowid')
        """)
        conn.commit()
        conn.close()

    async def save(self, entry: MemoryEntry) -> bool:
        import aiosqlite
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """INSERT OR REPLACE INTO memories
                   (id, memory_type, content, importance, tags, created_at, last_accessed, access_count, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    entry.id, entry.memory_type, entry.content, entry.importance,
                    json.dumps(entry.tags), entry.created_at, entry.last_accessed,
                    entry.access_count, json.dumps(entry.metadata),
                ),
            )
            await db.commit()
        return True

    async def get(self, memory_id: str) -> MemoryEntry | None:
        import aiosqlite
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                "SELECT * FROM memories WHERE id = ?", (memory_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if not row:
                    return None
                return MemoryEntry(
                    id=row[0], memory_type=row[1], content=row[2], importance=row[3],
                    tags=json.loads(row[4]), created_at=row[5], last_accessed=row[6],
                    access_count=row[7], metadata=json.loads(row[8]),
                )

    async def search(
        self, query: str, memory_type: str | None = None, top_k: int = 5
    ) -> list[MemoryEntry]:
        import aiosqlite
        async with aiosqlite.connect(self._db_path) as db:
            where_clause = ""
            params = [query]
            if memory_type:
                where_clause = "AND memories.memory_type = ?"
                params.append(memory_type)

            sql = f"""
                SELECT m.*, rank FROM memories_fts fts
                JOIN memories m ON fts.rowid = m.rowid
                WHERE fts.content MATCH ? {where_clause}
                ORDER BY rank LIMIT ?
            """
            params.append(top_k)

            async with db.execute(sql, params) as cursor:
                rows = await cursor.fetchall()

        entries = []
        for row in rows:
            entries.append(MemoryEntry(
                id=row[0], memory_type=row[1], content=row[2], importance=row[3],
                tags=json.loads(row[4]), created_at=row[5], last_accessed=row[6],
                access_count=row[7], metadata=json.loads(row[8]),
            ))
        return entries

    async def delete(self, memory_id: str) -> bool:
        import aiosqlite
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            await db.commit()
        return True

    async def list_all(
        self, memory_type: str | None = None, tag: str | None = None
    ) -> list[MemoryEntry]:
        import aiosqlite
        async with aiosqlite.connect(self._db_path) as db:
            where = ""
            params = []
            if memory_type:
                where = "WHERE memory_type = ?"
                params.append(memory_type)

            async with db.execute(f"SELECT * FROM memories {where}", params) as cursor:
                rows = await cursor.fetchall()

        entries = []
        for row in rows:
            entry = MemoryEntry(
                id=row[0], memory_type=row[1], content=row[2], importance=row[3],
                tags=json.loads(row[4]), created_at=row[5], last_accessed=row[6],
                access_count=row[7], metadata=json.loads(row[8]),
            )
            if tag and tag not in entry.tags:
                continue
            entries.append(entry)
        return entries

    async def update_access(self, memory_id: str) -> bool:
        import aiosqlite
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "UPDATE memories SET last_accessed = ?, access_count = access_count + 1 WHERE id = ?",
                (time.time(), memory_id),
            )
            await db.commit()
        return True


class LongTermMemoryManager:
    """
    长期记忆管理器。

    Usage:
        manager = LongTermMemoryManager(store=SQLiteMemoryStore())
        # Store a fact
        await manager.store_fact("User prefers dark theme", user_id="user_001")
        # Recall relevant memories for a query
        memories = await manager.recall("What does the user prefer?", user_id="user_001")
    """

    def __init__(self, store: BaseMemoryStore):
        self._store = store

    async def store_fact(
        self, content: str, user_id: str, importance: float = 0.7, tags: list[str] | None = None
    ) -> MemoryEntry:
        """存储事实型记忆"""
        import uuid
        entry = MemoryEntry(
            id=f"fact_{user_id}_{uuid.uuid4().hex[:8]}",
            memory_type="fact",
            content=content,
            importance=importance,
            tags=tags or [f"user:{user_id}"],
            created_at=time.time(),
            last_accessed=time.time(),
        )
        await self._store.save(entry)
        logger.info(f"Stored fact memory: {entry.id}")
        return entry

    async def store_episode(
        self, content: str, user_id: str, importance: float = 0.5, tags: list[str] | None = None
    ) -> MemoryEntry:
        """存储经历型记忆（重要对话片段）"""
        import uuid
        entry = MemoryEntry(
            id=f"episode_{user_id}_{uuid.uuid4().hex[:8]}",
            memory_type="episode",
            content=content,
            importance=importance,
            tags=(tags or []) + [f"user:{user_id}"],
            created_at=time.time(),
            last_accessed=time.time(),
        )
        await self._store.save(entry)
        logger.info(f"Stored episode memory: {entry.id}")
        return entry

    async def store_skill(
        self, content: str, tags: list[str] | None = None
    ) -> MemoryEntry:
        """存储技能型记忆（Agent 学到的操作模式）"""
        import uuid
        entry = MemoryEntry(
            id=f"skill_{uuid.uuid4().hex[:8]}",
            memory_type="skill",
            content=content,
            importance=0.9,
            tags=tags or [],
            created_at=time.time(),
            last_accessed=time.time(),
        )
        await self._store.save(entry)
        logger.info(f"Stored skill memory: {entry.id}")
        return entry

    async def recall(
        self, query: str, user_id: str | None = None, top_k: int = 5
    ) -> list[MemoryEntry]:
        """检索相关记忆"""
        # 先搜索事实型记忆
        facts = await self._store.search(query, memory_type="fact", top_k=top_k)
        episodes = await self._store.search(query, memory_type="episode", top_k=top_k // 2)

        # 如果指定了用户，过滤出该用户的记忆
        if user_id:
            user_tag = f"user:{user_id}"
            facts = [e for e in facts if user_tag in e.tags]
            episodes = [e for e in episodes if user_tag in e.tags]

        # 合并并按重要性排序
        all_memories = facts + episodes
        all_memories.sort(key=lambda e: (e.importance, e.access_count), reverse=True)

        # 更新访问记录
        for entry in all_memories[:top_k]:
            await self._store.update_access(entry.id)

        return all_memories[:top_k]

    async def forget(
        self, memory_id: str | None = None, user_id: str | None = None, threshold: float = 0.1
    ) -> int:
        """
        遗忘机制。

        - 指定 ID：删除特定记忆
        - 指定用户 + 阈值：删除该用户下重要性低于阈值的记忆
        """
        if memory_id:
            await self._store.delete(memory_id)
            return 1

        if user_id:
            entries = await self._store.list_all(tag=f"user:{user_id}")
            count = 0
            for entry in entries:
                if entry.importance < threshold:
                    # 检查是否超过 30 天未访问
                    days_since_access = (time.time() - entry.last_accessed) / 86400
                    if days_since_access > 30:
                        await self._store.delete(entry.id)
                        count += 1
            return count

        return 0

    async def decay_importance(self, max_age_days: int = 90):
        """定期衰减记忆重要性"""
        entries = await self._store.list_all()
        now = time.time()

        for entry in entries:
            age_days = (now - entry.created_at) / 86400
            if age_days > max_age_days:
                # 指数衰减：每超过 max_age_days，重要性减半
                decay_factor = 2 ** (-age_days / max_age_days)
                entry.importance *= decay_factor

        for entry in entries:
            await self._store.save(entry)

        logger.info(f"Decayed importance for {len(entries)} memories")
```

---

## 5. Agent 集成 `memory/integration.py`

```python
"""
记忆系统与 Agent Core 的集成。
"""

import logging
from memory.short_term import ShortTermMemory, Message
from memory.long_term import LongTermMemoryManager, MemoryEntry

logger = logging.getLogger(__name__)


class MemoryContext:
    """
    Agent 使用的完整记忆上下文。

    将短期记忆和长期记忆合并为 LLM 可用的格式。
    """

    def __init__(
        self,
        short_term: ShortTermMemory,
        long_term: LongTermMemoryManager | None = None,
    ):
        self._short_term = short_term
        self._long_term = long_term

    async def build_messages(self, user_query: str) -> list[dict]:
        """构建包含长期记忆的完整消息列表"""
        messages = self._short_term.get_messages()

        # 检索相关长期记忆
        if self._long_term and user_query:
            relevant_memories = await self._long_term.recall(user_query, top_k=3)
            if relevant_memories:
                memory_context = self._format_long_term_memories(relevant_memories)
                # 注入为 system message（放在最前面）
                messages.insert(0, {
                    "role": "system",
                    "content": f"Relevant long-term memories:\n{memory_context}",
                })

        return messages

    def _format_long_term_memories(self, entries: list[MemoryEntry]) -> str:
        parts = []
        for entry in entries:
            type_label = {"fact": "事实", "episode": "经历", "skill": "技能"}.get(
                entry.memory_type, entry.memory_type
            )
            parts.append(f"- [{type_label}] {entry.content} (重要性: {entry.importance:.2f})")
        return "\n".join(parts) if parts else ""

    def add_message(self, role: str, content: str):
        """添加消息到短期记忆"""
        msg = Message(
            role=role,
            content=content,
            token_count=max(1, len(content) // 4),
        )
        self._short_term.add_message(msg)

    async def extract_and_store_memories(self, conversation: list[dict], user_id: str):
        """
        从对话中提取关键信息并存储为长期记忆。

        生产环境应调用 LLM 进行信息抽取，这里提供接口框架。
        """
        if not self._long_term:
            return

        # TODO: 调用 LLM 提取事实、经验
        # facts = await llm_extract_facts(conversation)
        # for fact in facts:
        #     await self._long_term.store_fact(fact, user_id=user_id)
```

---

## 6. YAML 配置 `config/memory.yaml`

```yaml
memory:
  short_term:
    max_tokens: 4000              # 短期记忆 token 上限
    reserve_tokens: 1000          # 为回复预留的 token
    summarize_enabled: true       # 是否启用 LLM 摘要压缩
    summarize_batch_size: 4       # 每次摘要的消息数量

  long_term:
    enabled: true
    store_type: "sqlite"          # json_file | sqlite
    db_path: "./data/memory.db"

    auto_extract: false           # 是否自动从对话中提取记忆（需 LLM）
    extract_interval_messages: 10 # 每 N 条消息触发一次提取

    forgetting:
      enabled: true
      min_importance: 0.1         # 低于此值的记忆可被遗忘
      max_age_days: 90            # 超过此天数的记忆开始衰减
      decay_interval_hours: 24    # 衰减检查间隔
```

---

## 7. 架构总览

```
                    ┌─────────────────────┐
                    │     Agent Core      │
                    │                     │
                    │   MemoryContext     │
                    │   ├─ ShortTermMem   │──→ LLM API messages[]
                    │   └─ LongTermMem    │──→ System prompt injection
                    └──────────┬──────────┘
                               │
              ┌────────────────┴────────────────┐
              ▼                                  ▼
   ┌─────────────────────┐          ┌─────────────────────┐
   │   Short-Term Memory  │          │   Long-Term Memory   │
   │                     │          │                       │
   │ • Message window    │          │ • Fact Store          │
   │ • Token budget      │          │ • Episode Store       │
   │ • Auto-eviction     │          │ • Skill Store         │
   │ • Summary compression│         │ • Forgetting mechanism│
   └─────────────────────┘          └──────────┬────────────┘
                                                │
                                    ┌───────────┴───────────┐
                                    ▼                       ▼
                           ┌──────────────┐        ┌──────────────┐
                           │ JSON File    │        │ SQLite       │
                           │ Store        │        │ (FTS5)       │
                           └──────────────┘        └──────────────┘
```

---

## 8. 设计总结

| 特性 | 实现方式 |
|------|---------|
| **短期记忆** | Token 窗口管理 + 自动裁剪 + LLM 摘要压缩 |
| **长期记忆** | 三类存储（事实/经历/技能），支持语义检索和遗忘机制 |
| **存储可切换** | JSON File（开发）/ SQLite FTS5（生产），统一 `BaseMemoryStore` 接口 |
| **Agent 集成** | `MemoryContext` 合并短期+长期记忆，注入 LLM prompt |
| **遗忘机制** | 重要性衰减 + 时间衰减，自动清理低价值记忆 |
