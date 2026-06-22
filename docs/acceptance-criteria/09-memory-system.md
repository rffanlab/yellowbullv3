# 记忆系统 — 验收标准

## MEM-01: ShortTermMemory — 消息追加与检索

**优先级**: P0

- **Given** `ShortTermMemory` 实例，容量限制 20 条消息
- **When** 调用 `add(message)` 25 次
- **Then** 仅保留最近 20 条；`get_recent(n=5)` 返回最新 5 条（按时间倒序）
- **验证方式**: 单元测试

## MEM-02: ShortTermMemory — Token 限制裁剪

**优先级**: P1

- **Given** `ShortTermMemory(max_tokens=4096)`
- **When** 追加消息导致总 token 数超限
- **Then** 自动移除最早的消息直到总 token <= max_tokens；system message 始终保留
- **验证方式**: 单元测试 — mock tokenizer

## MEM-03: LongTermMemory — 事实存储

**优先级**: P1

- **Given** `LongTermMemory` 实例（SQLite + Embedding）
- **When** 调用 `store_fact(user_id, fact="用户喜欢 Python")`
- **Then** 事实持久化到数据库；包含 embedding vector、user_id、created_at
- **验证方式**: 集成测试

## MEM-04: LongTermMemory — 语义检索

**优先级**: P1

- **Given** 已存储 50 条 facts
- **When** 调用 `search(user_id, query="编程语言偏好", top_k=3)`
- **Then** 返回 3 条最相关的 facts，按 relevance score 排序；仅检索该 user_id 的事实
- **验证方式**: 集成测试

## MEM-05: LongTermMemory — 事实去重/合并

**优先级**: P2

- **Given** 已存储 "用户喜欢 Python"
- **When** 调用 `store_fact(user_id, fact="用户对 Python 感兴趣")`（语义相似）
- **Then** 检测到重复，更新/合并而非新增；记录 info 日志
- **验证方式**: 集成测试 — 断言 facts 数量不变

## MEM-06: MemoryService — 自动提取事实

**优先级**: P1

- **Given** `MemoryService` 已绑定 AgentCore
- **When** 对话中包含可提取的事实（如 "我叫张三"）
- **Then** LLM 被调用提取 facts；新事实存入 LongTermMemory；不阻塞主对话流
- **验证方式**: 集成测试 — mock LLM fact extraction

## MEM-07: MemoryService — 上下文注入

**优先级**: P1

- **Given** LongTermMemory 中有用户相关 facts
- **When** AgentCore 发起新对话
- **Then** 相关 facts 作为 system prompt 补充注入（"已知信息：..."）；不超过 token 预算
- **验证方式**: 集成测试 — 断言 messages[0] 包含记忆内容

## MEM-08: MemoryService — 记忆摘要

**优先级**: P2

- **Given** ShortTermMemory 积累了大量消息
- **When** 调用 `summarize()`（手动或定时触发）
- **Then** LLM 生成对话摘要；摘要存入 LongTermMemory；ShortTermMemory 可清空
- **验证方式**: 集成测试 — mock LLM summary

## MEM-09: Memory 隔离

**优先级**: P1

- **Given** 多用户场景（user_a, user_b）
- **When** user_a 存储 facts，user_b 检索
- **Then** user_b 无法获取 user_a 的 facts；完全按 user_id 隔离
- **验证方式**: 集成测试

## MEM-10: Memory 持久化恢复

**优先级**: P1

- **Given** LongTermMemory 使用 SQLite 存储
- **When** 进程重启后重建 MemoryService
- **Then** facts 从数据库恢复；embedding 索引可用
- **验证方式**: 集成测试 — 写入 → 重建 → 断言数据一致
