# 记忆系统（Memory System）详细设计

## 1. 职责边界

| 类型 | 说明 | 存储方式 |
|------|------|---------|
| **短期记忆** | 当前会话内的对话历史，上下文窗口裁剪 | Session Manager (SQLite) |
| **长期记忆** | 跨会话持久化的关键信息（用户偏好、事实、决策） | 向量数据库 + SQLite |
| **工作记忆** | Agent 执行任务时的临时状态和中间结果 | 内存（ephemeral） |

---

## 2. 短期记忆 `memory/short_term.py`

```python
"""
短期记忆 —— 基于 Session Manager 的上下文窗口管理。

核心能力：
- Token 预算控制：确保发送给 LLM 的消息不超过模型上下文窗口
- 滑动窗口裁剪：保留最近 N 条消息，超出部分丢弃或摘要化
- 系统提示始终保留：system message 不参与裁剪
"""

from typing import List, Optional


class ShortTermMemory:
    """短期记忆管理器"""

    def __init__(self, max_tokens: int = 8000, reserve_for_system: int = 1024):
        self.max_tokens = max_tokens
        self.reserve_for_system = reserve_for_system

    @property
    def available_tokens(self) -> int:
        """可用于对话历史的 token 预算"""
        return self.max_tokens - self.reserve_for_system

    def trim_messages(
        self, messages: List[dict], system_message: Optional[str] = None,
    ) -> List[dict]:
        """
        滑动窗口裁剪：从最新消息向前累积，直到达到 token 预算。

        Args:
            messages: [{"role": "user"/"assistant"/"tool", "content": "..."}, ...]
            system_message: 系统提示（始终保留）

        Returns:
            裁剪后的消息列表 [system, ..., recent messages]
        """
        import tiktoken

        encoder = tiktoken.get_encoding("cl100k_base")

        available = self.available_tokens

        # 从最新消息向前累积
        trimmed = []
        for msg in reversed(messages):
            msg_tokens = len(encoder.encode(msg["content"]))
            if msg_tokens + sum(len(encoder.encode(m["content"])) for m in trimmed) > available:
                break
            trimmed.append(msg)

        trimmed.reverse()  # 恢复时间顺序

        result = []
        if system_message:
            result.append({"role": "system", "content": system_message})
        result.extend(trimmed)

        return result


class ConversationSummarizer:
    """
    对话摘要器 —— 当历史消息超出窗口时，将旧消息压缩为摘要。

    策略：调用 LLM 对早期对话生成一段摘要，替换原始消息。
    """

    def __init__(self, llm_provider):
        self._llm = llm_provider

    async def summarize(self, messages: List[dict], max_length: int = 500) -> str:
        """将早期对话压缩为摘要"""
        conversation_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in messages
        )

        summary_prompt = f"""请将以下对话内容总结为一段简洁的摘要（不超过{max_length}字），保留关键信息：

{conversation_text}

摘要："""

        response = await self._llm.chat([{"role": "user", "content": summary_prompt}], stream=False)
        return response.content

    async def trim_with_summary(
        self, messages: List[dict], memory: ShortTermMemory,
    ) -> List[dict]:
        """裁剪 + 摘要：超出窗口的消息压缩为一条摘要"""
        import tiktoken

        encoder = tiktoken.get_encoding("cl100k_base")
        available = memory.available_tokens

        # 计算当前 token 总量（不含 system）
        total_tokens = sum(len(encoder.encode(m["content"])) for m in messages)

        if total_tokens <= available:
            return messages

        # 从最新消息向前累积，直到达到预算
        recent = []
        old_messages = []
        remaining = available

        for msg in reversed(messages):
            msg_tokens = len(encoder.encode(msg["content"]))
            if msg_tokens + sum(len(encoder.encode(m["content"])) for m in recent) > available:
                old_messages.insert(0, msg)
            else:
                recent.insert(0, msg)

        # 对旧消息生成摘要
        if old_messages:
            summary = await self.summarize(old_messages)
            return [
                {"role": "system", "content": f"[对话摘要] {summary}"},
            ] + recent

        return messages
```

---

## 3. 长期记忆 `memory/long_term.py`

```python
"""
长期记忆 —— 跨会话持久化的关键信息。

核心能力：
- 事实提取：从对话中自动识别重要事实（用户偏好、个人信息、决策）
- 语义检索：基于向量相似度查找相关记忆
- 手动管理：支持显式添加/删除/查询记忆条目
- TTL 过期：记忆条目的生存时间控制

存储结构：
{
    "id": "memory_uuid",
    "content": "用户偏好使用 Python 而非 JavaScript",
    "embedding": [...],
    "user_id": "user_001",
    "source_session": "session_uuid",
    "created_at": "2026-01-01T00:00:00Z",
    "last_accessed_at": "2026-01-05T12:00:00Z",
    "access_count": 3,
    "ttl_days": 90,           # 过期时间（可选）
    "tags": ["preference", "programming"],
}
"""

import uuid
from datetime import datetime, timezone, timedelta
from typing import List, Optional


class MemoryEntry:
    """记忆条目"""

    def __init__(
        self, content: str, user_id: str, source_session: str = "",
        tags: List[str] = None, ttl_days: int = 90,
    ):
        self.id = str(uuid.uuid4())
        self.content = content
        self.user_id = user_id
        self.source_session = source_session
        self.tags = tags or []
        self.ttl_days = ttl_days
        self.created_at = datetime.now(timezone.utc)
        self.last_accessed_at = self.created_at
        self.access_count = 0

    @property
    def is_expired(self) -> bool:
        if self.ttl_days <= 0:
            return False
        expiry = self.created_at + timedelta(days=self.ttl_days)
        return datetime.now(timezone.utc) > expiry

    def touch(self):
        """更新访问时间和计数"""
        self.last_accessed_at = datetime.now(timezone.utc)
        self.access_count += 1


class LongTermMemory:
    """长期记忆管理器"""

    def __init__(self, vector_store: "VectorStore", embedding_provider: "EmbeddingProvider"):
        from rag.vector_store import VectorStore
        from rag.embedding import EmbeddingProvider
        self._store = vector_store
        self._embedder = embedding_provider
        self._collection = "long_term_memory"

    async def add(self, entry: MemoryEntry) -> str:
        """添加记忆条目"""
        from rag.chunker import DocumentChunk

        chunk = DocumentChunk(
            text=entry.content,
            source=f"memory:{entry.id}",
            chunk_index=0,
            metadata={
                "user_id": entry.user_id,
                "source_session": entry.source_session,
                "tags": ",".join(entry.tags),
                "created_at": entry.created_at.isoformat(),
                "memory_id": entry.id,
            },
        )

        await self._store.add_documents([chunk], collection=self._collection)
        return entry.id

    async def search(
        self, query: str, user_id: str = None, top_k: int = 5,
    ) -> List[dict]:
        """语义检索相关记忆"""
        results = await self._store.search(query, collection=self._collection, top_k=top_k)

        # 过滤：用户隔离 + 过期检查
        filtered = []
        for r in results:
            if user_id and r["metadata"].get("user_id") != user_id:
                continue
            created_at = datetime.fromisoformat(r["metadata"]["created_at"])
            if created_at < datetime.now(timezone.utc) - timedelta(days=90):
                continue
            filtered.append(r)

        return filtered[:top_k]

    async def delete(self, memory_id: str):
        """删除记忆条目（通过 metadata filter）"""
        # ChromaDB 支持 where 条件删除
        self._store._client.get_collection(name=self._collection).delete(
            where={"metadata.memory_id": memory_id}
        )

    async def list_by_user(self, user_id: str, limit: int = 50) -> List[dict]:
        """列出用户的所有记忆"""
        results = await self._store.search("", user_id=user_id, top_k=limit)
        return [r for r in results if r["metadata"].get("user_id") == user_id]

    async def cleanup_expired(self):
        """清理过期记忆条目"""
        # 定期任务：扫描并删除过期条目
        pass


class MemoryExtractor:
    """
    自动事实提取器 —— 从对话中识别值得长期保存的信息。

    触发条件（可配置）：
    - 用户明确表达偏好："我喜欢..."、"我希望..."
    - 用户提供个人信息："我叫..."、"我在...工作"
    - 重要决策或结论
    """

    EXTRACT_PROMPT = """请从以下对话中提取值得长期保存的关键信息。只提取事实性内容，不要提取临时性的问答。

提取规则：
1. 用户偏好（喜欢的工具、语言、风格等）
2. 个人信息（职业、兴趣、习惯等）
3. 重要决策或结论
4. 项目相关信息

如果没有任何值得保存的信息，返回空列表。

对话内容：
{conversation}

请以 JSON 数组格式返回提取的结果：
[{{"content": "提取的内容", "tags": ["标签1", "标签2"]}}]"""

    def __init__(self, llm_provider):
        self._llm = llm_provider

    async def extract(self, conversation: List[dict], user_id: str) -> List[MemoryEntry]:
        """从对话中提取记忆条目"""
        conversation_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in conversation[-10:]  # 最近 10 条消息
        )

        prompt = self.EXTRACT_PROMPT.format(conversation=conversation_text)
        response = await self._llm.chat([{"role": "user", "content": prompt}], stream=False)

        import json
        try:
            extracted = json.loads(response.content)
            if not isinstance(extracted, list):
                return []

            entries = []
            for item in extracted:
                entry = MemoryEntry(
                    content=item["content"],
                    user_id=user_id,
                    tags=item.get("tags", ["auto_extracted"]),
                )
                entries.append(entry)
            return entries
        except (json.JSONDecodeError, KeyError):
            return []
```

---

## 4. Agent Core 集成 `memory/integration.py`

```python
"""
记忆系统与 Agent Core 的集成。

在 Agent 主循环中：
1. 会话开始时 → 检索相关长期记忆，注入 system prompt
2. 对话过程中 → 短期记忆管理（上下文窗口裁剪）
3. 会话结束时 → 自动提取新事实，写入长期记忆
"""


class MemoryManager:
    """统一记忆管理器"""

    def __init__(
        self,
        short_term: ShortTermMemory,
        long_term: LongTermMemory,
        summarizer: ConversationSummarizer = None,
        extractor: MemoryExtractor = None,
    ):
        self.short_term = short_term
        self.long_term = long_term
        self.summarizer = summarizer
        self.extractor = extractor

    async def prepare_context(
        self, user_message: str, user_id: str, system_prompt: str,
    ) -> str:
        """
        准备增强后的 system prompt：
        1. 检索相关长期记忆
        2. 注入到 system prompt
        """
        memories = await self.long_term.search(user_message, user_id=user_id, top_k=3)

        if not memories:
            return system_prompt

        memory_text = "\n".join(
            f"- {m['text']} (相关度: {m['score']})" for m in memories
        )

        enriched = f"""{system_prompt}

## 用户长期记忆（参考以下信息调整回复风格和内容）：
{memory_text}
"""
        return enriched

    async def trim_history(
        self, messages: List[dict], system_message: str,
    ) -> List[dict]:
        """裁剪对话历史（短期记忆管理）"""
        if self.summarizer and len(messages) > 50:
            return await self.summarizer.trim_with_summary(messages, self.short_term)
        return self.short_term.trim_messages(messages, system_message)

    async def post_session_extract(
        self, conversation: List[dict], user_id: str, session_id: str,
    ) -> int:
        """会话结束后提取并保存新记忆"""
        if not self.extractor:
            return 0

        new_memories = await self.extractor.extract(conversation, user_id)
        for entry in new_memories:
            entry.source_session = session_id
            await self.long_term.add(entry)

        return len(new_memories)
```

---

## 5. YAML 配置 `memory` section

```yaml
# config/settings.yaml (新增)
memory:
  short_term:
    max_tokens: 8000          # 模型上下文窗口大小（GPT-4o: 128k, Claude: 200k）
    reserve_for_system: 1024  # 为 system prompt 预留的 token
    summarize_threshold: 50   # 消息数量超过此值时触发摘要

  long_term:
    enabled: false            # 默认关闭，按需开启
    auto_extract: true        # 会话结束后自动提取事实
    ttl_days: 90              # 记忆条目过期时间（0 = 永不过期）
    max_per_user: 500         # 每个用户的最大记忆数量

  embedding:                  # 复用 RAG 的 embedding provider
    provider: "openai"
    model: "text-embedding-3-small"
```

---

## 6. 数据流图

```
会话开始
   │
   ▼
检索长期记忆 (LongTermMemory.search)
   │
   ▼
注入 System Prompt ──→ Agent Core 主循环
                              │
                    ┌─────────┴─────────┐
                    │ 短期记忆管理        │
                    │ (ShortTermMemory)  │
                    │ - 滑动窗口裁剪      │
                    │ - Token 预算控制    │
                    └─────────┬─────────┘
                              │
                              ▼
                      LLM.chat(trimmed messages)
                              │
                              ▼
会话结束
   │
   ▼
自动事实提取 (MemoryExtractor.extract)
   │
   ▼
写入长期记忆 (LongTermMemory.add)
```

---

## 7. 设计总结

| 特性 | 实现方式 |
|------|---------|
| **短期记忆** | Token 预算滑动窗口 + 可选对话摘要压缩 |
| **长期记忆** | 向量存储语义检索 + TTL 过期控制 |
| **自动提取** | LLM 驱动的对话事实识别，会话后异步写入 |
| **用户隔离** | 按 user_id 过滤，确保多租户安全 |
| **手动管理** | API 支持显式添加/删除/查询记忆条目 |

---

## 8. 隐私控制与跨会话记忆

### 8.1. 设计目标

- **用户隐私隔离**：确保不同用户的记忆数据完全隔离
- **敏感信息保护**：自动识别和脱敏 PII（个人身份信息）
- **跨会话持久化**：关键记忆在会话结束后保留，供未来会话使用

### 8.2. 隐私控制 `memory/privacy.py`

```python
from dataclasses import dataclass, field
from enum import Enum


class MemoryPrivacyLevel(str, Enum):
    PUBLIC = "public"           # 公开：可被所有模块访问
    SESSION_ONLY = "session_only"  # 会话级：仅当前会话可见
    PRIVATE = "private"         # 私有：仅用户本人和授权模块可见
    ENCRYPTED = "encrypted"     # 加密存储


@dataclass(frozen=True)
class PrivacyPolicy:
    """记忆隐私策略"""
    default_level: MemoryPrivacyLevel = MemoryPrivacyLevel.PRIVATE
    auto_pii_detection: bool = True       # 自动检测 PII
    auto_encrypt_sensitive: bool = True   # 加密敏感信息
    retention_days: int = 30              # 默认保留天数


class PrivacyController:
    """隐私控制器"""

    def __init__(self, policy: PrivacyPolicy | None = None):
        self._policy = policy or PrivacyPolicy()

    def classify(self, content: str) -> MemoryPrivacyLevel:
        """根据内容自动分类隐私级别"""
        if not self._policy.auto_pii_detection:
            return self._policy.default_level

        # 检测 PII（手机号、身份证号等）
        import re
        pii_patterns = [
            r"1[3-9]\d{9}",           # 手机号
            r"\b\d{17}[\dxX]\b",      # 身份证号
            r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",  # 邮箱
        ]

        for pattern in pii_patterns:
            if re.search(pattern, content):
                return MemoryPrivacyLevel.ENCRYPTED

        return self._policy.default_level

    def should_encrypt(self, privacy_level: MemoryPrivacyLevel) -> bool:
        """判断是否需要加密"""
        return (
            self._policy.auto_encrypt_sensitive and
            privacy_level == MemoryPrivacyLevel.ENCRYPTED
        )
```

### 8.3. 跨会话记忆持久化 `memory/cross_session.py`

```python
from dataclasses import dataclass, field


@dataclass(frozen=True)
class CrossSessionMemory:
    """跨会话记忆条目"""
    user_id: str                    # 用户 ID
    content: str                    # 记忆内容
    category: str                   # 分类：preference | fact | decision | goal
    confidence: float = 0.5         # 置信度 [0, 1]
    created_at: int = 0             # 创建时间戳
    last_accessed_at: int = 0       # 最后访问时间戳
    access_count: int = 0           # 访问次数


class CrossSessionMemoryStore:
    """跨会话记忆存储"""

    def __init__(self, storage_backend):
        self._storage = storage_backend

    async def add(self, memory: CrossSessionMemory) -> None:
        """添加跨会话记忆"""
        await self._storage.put(
            key=f"cross_session:{memory.user_id}:{memory.category}",
            value=memory.__dict__,
            ttl=self._policy.retention_days * 86400,
        )

    async def retrieve(
        self, user_id: str, query: str, limit: int = 5
    ) -> list[CrossSessionMemory]:
        """检索相关记忆"""
        # 语义搜索 + 用户过滤
        results = await self._storage.query(
            collection="cross_session_memories",
            filter={"user_id": user_id},
            query=query,
            limit=limit,
        )

        memories = [CrossSessionMemory(**r) for r in results]

        # 更新访问统计
        for m in memories:
            await self._update_access_stats(m.user_id, m.content)

        return memories

    async def _update_access_stats(self, user_id: str, content: str) -> None:
        """更新记忆访问统计"""
        pass  # TODO: 实现原子递增
```

### 8.4. Agent Core 集成

在 `agent/core.py` 中，会话结束时自动提取关键记忆：

```python
async def on_session_end(self, session_id: str):
    """会话结束时的记忆处理"""
    conversation = await self.memory.get_conversation(session_id)

    # LLM 驱动的关键信息提取
    key_memories = await self._extract_key_memories(conversation)

    for memory in key_memories:
        privacy_level = self.privacy_controller.classify(memory.content)
        cross_session = CrossSessionMemory(
            user_id=self.session_manager.get_user_id(session_id),
            content=memory.content,
            category=memory.category,
            confidence=memory.confidence,
        )
        await self.cross_session_store.add(cross_session)

    # 清理短期记忆（会话级）
    await self.memory.clear_short_term(session_id)
```

---

## 9. 更新后的设计总结

| 特性 | 实现方式 |
|------|---------|
| **短期记忆** | Token 预算滑动窗口 + 可选对话摘要压缩 |
| **长期记忆** | 向量存储语义检索 + TTL 过期控制 |
| **自动提取** | LLM 驱动的对话事实识别，会话后异步写入 |
| **用户隔离** | 按 user_id 过滤，确保多租户安全 |
| **手动管理** | API 支持显式添加/删除/查询记忆条目 |
| **隐私控制** | PrivacyController 自动分类 + PII 检测 + 加密存储 |
| **跨会话持久化** | CrossSessionMemoryStore 保留关键信息供未来使用 |
