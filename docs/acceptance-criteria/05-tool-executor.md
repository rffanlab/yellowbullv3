# Tool Executor — 验收标准

## TEX-01: 单次工具执行成功

**优先级**: P0

- **Given** `ToolExecutor` 已初始化，注册表中存在 `calculator` 工具
- **When** 调用 `execute_batch([ToolCallRequest(tool_name="calculator", arguments={"expression": "2+3"})])`
- **Then** 返回 `BatchExecutionResult`，其中包含一个 `ExecutionResult(status=SUCCESS, content="5")`
- **验证方式**: 单元测试

## TEX-02: 工具超时处理

**优先级**: P0

- **Given** 注册表中存在一个执行时间超过 10s 的慢工具
- **When** 调用 `execute_batch()`，设置 `default_timeout=5.0`
- **Then** 返回 `ExecutionResult(status=TIMEOUT, error="timed out after 5s")`；不阻塞其他并行工具的执行
- **验证方式**: 单元测试 — mock 慢工具

## TEX-03: 重试机制

**优先级**: P0

- **Given** 注册表中存在一个前 2 次调用失败、第 3 次成功的工具
- **When** 调用 `execute_batch()`，设置 `max_retries=2`
- **Then** 返回 `ExecutionResult(status=SUCCESS, retry_count=2)`；重试之间有指数退避延迟
- **验证方式**: 单元测试 — mock 失败→成功序列

## TEX-04: 重试耗尽

**优先级**: P0

- **Given** 注册表中存在一个始终失败的工具
- **When** 调用 `execute_batch()`，设置 `max_retries=2`
- **Then** 返回 `ExecutionResult(status=RETRY_EXHAUSTED, error="Failed after 2 retries: ...")`
- **验证方式**: 单元测试

## TEX-05: 并行执行

**优先级**: P0

- **Given** 3 个独立工具，每个耗时约 1s
- **When** 调用 `execute_batch()`（构造时设置 `parallel=True`）
- **Then** 总耗时 ≈ 1s（而非 3s）；所有结果正确返回
- **验证方式**: 单元测试 — mock 固定延迟工具

## TEX-06: 串行执行

**优先级**: P1

- **Given** 3 个独立工具，每个耗时约 1s
- **When** 调用 `execute_batch()`（构造时设置 `parallel=False`）
- **Then** 总耗时 ≈ 3s；结果按请求顺序返回
- **验证方式**: 单元测试

## TEX-07: 未知工具处理

**优先级**: P0

- **Given** 注册表中不存在 `nonexistent_tool`
- **When** 调用 `execute_batch([ToolCallRequest(tool_name="nonexistent_tool", ...)])`
- **Then** 返回 `ExecutionResult(status=FAILED, error="Unknown tool: nonexistent_tool")`
- **验证方式**: 单元测试

## TEX-08: 错误隔离

**优先级**: P0

- **Given** 2 个工具：一个正常，一个抛出异常
- **When** 调用 `execute_batch()`（并行模式）
- **Then** 正常工具返回 SUCCESS；失败工具返回 FAILED/RETRY_EXHAUSTED；两者互不影响
- **验证方式**: 单元测试

## TEX-09: 结果缓存命中

**优先级**: P1

- **Given** `enable_cache=True`，同一工具的相同参数已执行过一次
- **When** 再次调用 `execute_batch()` 使用相同的 tool_name + arguments
- **Then** 直接返回缓存结果；工具的实际 execute 方法不被调用
- **验证方式**: 单元测试 — mock 计数器验证调用次数

## TEX-10: 缓存清空

**优先级**: P2

- **Given** 缓存中有若干条目
- **When** 调用 `executor.clear_cache()`
- **Then** 后续相同参数的请求会重新执行工具（而非命中缓存）
- **验证方式**: 单元测试

## TEX-11: 执行审计日志

**优先级**: P1

- **Given** 正常执行的 batch
- **When** 调用 `execute_batch()`
- **Then** 日志中包含每个工具的耗时、状态信息；`BatchExecutionResult.total_duration_ms` 正确记录总耗时
- **验证方式**: 单元测试 — caplog 捕获日志输出

## TEX-12: Agent Core 集成

**优先级**: P0

- **Given** `Agent` 初始化时注入 `ToolExecutor`
- **When** LLM 返回多个 tool calls，Agent 调用 `_execute_tools()`
- **Then** ToolExecutor 执行所有工具；结果正确回注到 session history
- **验证方式**: 集成测试 — mock LLM + real executor
