# Session 管理 — 验收标准

## SES-01: Session 创建

**优先级**: P0

- **Given** `SessionManager` 实例已初始化
- **When** 调用 `create_session(user_id, title)`
- **Then** 返回新 `Session` 对象，包含唯一 `session_id`、空 `messages` 列表、`created_at` 时间戳
- **验证方式**: 单元测试

## SES-02: Session 消息追加

**优先级**: P0

- **Given** 已创建的 session
- **When** 调用 `add_message(session_id, Message(role="user", content="hello"))`
- **Then** 消息追加到 `messages` 列表；超过 `max_turns` 时自动裁剪最早的消息对
- **验证方式**: 单元测试

## SES-03: Session 持久化（SQLite）

**优先级**: P1

- **Given** `SessionManager(storage="sqlite")`，数据库文件存在
- **When** session 创建/消息追加后调用 `save()`
- **Then** 数据写入 SQLite；进程重启后从数据库恢复，messages 不丢失
- **验证方式**: 集成测试 — 启动 manager → 写入 → 重建 manager → 断言数据一致

## SES-04: Session 自动清理

**优先级**: P1

- **Given** 存在多个 session，部分超过 `max_idle_hours`（默认 72h）
- **When** 调用 `cleanup()` 或定时任务触发
- **Then** 过期 session 被删除；返回清理数量；活跃 session 不受影响
- **验证方式**: 单元测试 — mock 时间戳

## SES-05: Session 查找

**优先级**: P0

- **Given** 数据库中存在多个 sessions
- **When** 调用 `get_session(session_id)` / `list_sessions(user_id)`
- **Then** 正确返回对应 session；不存在的 ID 返回 `None`
- **验证方式**: 单元测试

## SES-06: Session 消息裁剪策略

**优先级**: P1

- **Given** session 设置了 `max_turns=10`，已有 9 轮对话（18 条消息）
- **When** 追加第 10 轮对话
- **Then** 保留最近 10 轮（20 条），最早一轮被移除；system message 始终保留
- **验证方式**: 单元测试

## SES-07: Session 并发安全

**优先级**: P1

- **Given** 同一 session 被多个请求并发写入
- **When** 同时调用 `add_message()`
- **Then** 消息顺序正确，无丢失或重复；使用锁或原子操作保证一致性
- **验证方式**: 集成测试 — asyncio.gather 并发写入后断言消息数

## SES-08: Session 内存存储（默认）

**优先级**: P0

- **Given** `SessionManager(storage="memory")`（默认）
- **When** 创建和查询 sessions
- **Then** 全部在内存中操作，无磁盘 I/O；进程重启后数据丢失（预期行为）
- **验证方式**: 单元测试

## SES-09: Session 元数据

**优先级**: P2

- **Given** session 包含 `user_id`、`title`、`updated_at` 等元数据
- **When** 追加消息时
- **Then** `updated_at` 自动更新；`message_count` 递增
- **验证方式**: 单元测试

## SES-10: Session 删除

**优先级**: P0

- **Given** 已存在的 session
- **When** 调用 `delete_session(session_id)`
- **Then** session 从存储中移除，返回 `True`；不存在的 ID 返回 `False`
- **验证方式**: 单元测试
