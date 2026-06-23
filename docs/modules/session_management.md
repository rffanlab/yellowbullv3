# Session Management 模块详细设计

## 1. 概述

Session Management 负责会话的创建、查询、删除和过期清理。每个用户对话对应一个 Session，包含消息历史和状态信息。

**对应源码:** `core/session_manager.py`, `models/session.py`

### 职责
- 会话生命周期管理（创建、获取、删除）
- 消息历史存储与滑动窗口截断
- 多步任务状态追踪
- 过期会话自动清理

## 2. 数据模型

### SessionState (会话状态)

```python
@dataclass
class SessionState:
    step: int = 0                              # 当前步骤编号
    chain_depth: int = 0                       # ReAct 链式调用深度
    tool_retry_counts: dict[str, int] = {}     # 工具重试计数 {tool_name: count}
```

### Session (会话)

```python
@dataclass
class Session:
    session_id: str                            # UUID，自动生成
    user_id: str                               # 用户标识
    messages: list[Message]                    # 消息历史
    state: SessionState                        # 会话状态
    created_at: datetime                       # 创建时间
    updated_at: datetime                       # 最后更新时间
```

#### add_message

```python
def add_message(self, msg: Message) -> None
```

追加消息到历史记录，同时更新 `updated_at` 时间戳。

#### get_context_messages

```python
def get_context_messages(self, window_size: int) -> list[Message]
```

滑动窗口截断：保留所有 SYSTEM 消息 + 最后 N 条非 SYSTEM 消息。

**算法:**
1. 过滤出所有 `role != SYSTEM` 的消息 → `non_system`
2. 取 `non_system[-window_size:]`（最后 window_size 条）
3. 返回 `[SYSTEM messages] + [recent non-system messages]`

## 3. SessionManager 设计

### SessionManager

```python
class SessionManager:
    _sessions: dict[str, Session] = {}         # 内存存储，session_id → Session

    def create(self, user_id: str) -> Session
    def get(self, session_id: str) -> Session | None
    def delete(self, session_id: str) -> bool
    def cleanup_expired(self, max_age_seconds: int = 7200) -> int
```

#### 方法说明

| 方法 | 参数 | 返回值 | 说明 |
|---|---|---|---|
| `create` | user_id: str | Session | 创建新会话，生成 UUID |
| `get` | session_id: str | Session \| None | 按 ID 查询会话 |
| `delete` | session_id: str | bool | 删除会话，返回是否成功 |
| `cleanup_expired` | max_age_seconds: int = 7200 | int | 清理过期会话，返回清理数量 |

### cleanup_expired

基于 `updated_at` 时间戳判断会话是否过期。

**行为:**
1. 计算截止时间: `now - timedelta(seconds=max_age_seconds)`
2. 遍历所有会话，找出 `updated_at < cutoff_time` 的会话
3. 从 `_sessions` 字典中移除
4. 返回被清理的会话数量

**默认过期时间:** 7200 秒（2小时）

## 4. 与主文档的对应关系

| agent-design.md 章节 | 本模块覆盖内容 |
|---|---|
| Session管理 - 创建/查询/删除 | ✅ create, get, delete |
| Session管理 - 消息历史 | ✅ messages list + add_message |
| Session管理 - 滑动窗口上下文 | ✅ get_context_messages |
| Session管理 - 过期清理 | ✅ cleanup_expired |

## 5. 依赖关系

```
core/session_manager
    ├── models.session.Session, SessionState
    └── models.message.Message, MessageRole
```

## 6. 注意事项

- 当前为纯内存存储，重启后所有会话丢失
- 生产环境建议接入 Redis/数据库持久化
- `updated_at` 在每次 add_message 时自动更新
- 滑动窗口大小默认 48 条消息（约 10-15 轮对话）
