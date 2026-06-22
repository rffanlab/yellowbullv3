# 消息队列与异步任务 — 验收标准

## MQ-01: TaskQueue — 任务提交与执行

**优先级**: P0

- **Given** `TaskQueue` 已启动（worker 运行中）
- **When** 调用 `submit(task_name, payload)` 
- **Then** 任务入队；worker 消费并执行；返回 task_id；执行完成后 status = "completed"
- **验证方式**: 集成测试

## MQ-02: TaskQueue — 任务状态查询

**优先级**: P0

- **Given** 已提交的任务
- **When** 调用 `get_status(task_id)`
- **Then** 返回 `{task_id, status, result/error, created_at, completed_at}`；status ∈ {pending, running, completed, failed}
- **验证方式**: 集成测试

## MQ-03: TaskQueue — 任务失败与重试

**优先级**: P1

- **Given** 任务执行抛出异常，配置 `max_retries=3`
- **When** worker 处理该任务
- **Then** 自动重试最多 3 次；每次重试间隔递增（exponential backoff）；全部失败后 status = "failed"，包含最后一次错误信息
- **验证方式**: 集成测试 — mock 失败任务

## MQ-04: TaskQueue — Dead Letter Queue

**优先级**: P1

- **Given** 任务超过 max_retries 仍失败
- **Then** 任务移入 dead letter queue；`get_dead_letters()` 可查询；支持手动重放 `replay(task_id)`
- **验证方式**: 集成测试

## MQ-05: TaskQueue — 并发控制

**优先级**: P1

- **Given** 配置 `max_concurrent_workers=4`
- **When** 同时提交 10 个长耗时任务
- **Then** 最多 4 个任务并行执行；其余排队等待；无任务丢失
- **验证方式**: 集成测试 — 断言并发数

## MQ-06: TaskQueue — 任务优先级

**优先级**: P2

- **Given** 队列中存在 high/normal/low 优先级的任务
- **When** worker 消费任务
- **Then** 高优先级任务先于低优先级执行；同优先级按 FIFO
- **验证方式**: 集成测试

## MQ-07: Redis Backend — 消息持久化

**优先级**: P1

- **Given** TaskQueue 使用 Redis backend
- **When** 进程重启后重建 TaskQueue
- **Then** pending/running 任务从 Redis 恢复；completed 任务保留可查询
- **验证方式**: 集成测试 — 写入 → 重启 → 断言数据一致

## MQ-08: In-Memory Backend（开发模式）

**优先级**: P1

- **Given** TaskQueue 使用 memory backend（无 Redis）
- **When** 提交和执行任务
- **Then** 功能正常；进程重启后队列清空（预期行为）
- **验证方式**: 单元测试

## MQ-09: Scheduled Tasks — 定时执行

**优先级**: P2

- **Given** `TaskScheduler` 已启动
- **When** 注册 cron 任务 `"0 */6 * * *"`（每 6 小时）
- **Then** 按时触发执行；支持暂停/恢复/删除
- **验证方式**: 集成测试 — mock time + 短间隔

## MQ-10: Task Result TTL

**优先级**: P2

- **Given** 配置 `result_ttl=3600`（1小时）
- **When** completed 任务超过 TTL
- **Then** result 被清除；status 保留但 result = None；节省存储空间
- **验证方式**: 集成测试

## MQ-11: Task API — GET /api/tasks/{id}

**优先级**: P1

- **Given** 已提交的任务
- **When** `GET /api/tasks/{task_id}`
- **Then** 返回任务状态 JSON；不存在的 ID 返回 404
- **验证方式**: 集成测试

## MQ-12: Task API — POST /api/tasks/dead-letter/replay

**优先级**: P2

- **Given** dead letter queue 中有失败任务
- **When** `POST /api/tasks/dead-letter/replay` 发送 `{"task_id": "xxx"}`
- **Then** 任务重新入队执行；返回新 task_id
- **验证方式**: 集成测试
