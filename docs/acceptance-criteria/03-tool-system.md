# 工具系统 — 验收标准

## TOL-01: ToolDefinition 协议

**优先级**: P0

- **Given** `ToolDefinition` 定义在 `llm/protocol.py`（唯一来源）
- **When** tools 模块 import `ToolDefinition`
- **Then** 从 `llm.protocol` 导入，不重复定义；包含 `name`、`description`、`parameters` 字段
- **验证方式**: 单元测试 — 断言类型来源

## TOL-02: ToolResult 协议

**优先级**: P0

- **Given** `ToolResult` 定义了成功/失败两种状态
- **When** 工具执行返回 `ToolResult`
- **Then** 包含 `content`（结果文本）、`is_error`（是否错误）字段；错误时 `is_error=True`
- **验证方式**: 单元测试

## TOL-03: ToolRegistry 注册与查找

**优先级**: P0

- **Given** `ToolRegistry` 实例，通过 `register()` 添加了 3 个工具
- **When** 调用 `get("tool_name")` / `list_tools()`
- **Then** 正确返回对应工具定义；不存在的名称返回 `None`
- **验证方式**: 单元测试

## TOL-04: ToolRegistry get_definitions()

**优先级**: P0

- **Given** `ToolRegistry` 注册了多个工具
- **When** 调用 `get_definitions()`
- **Then** 返回 `list[ToolDefinition]`，格式与 LLM protocol 兼容（含 JSON Schema parameters）
- **验证方式**: 单元测试 — 断言返回结构与 OpenAI tool format 一致

## TOL-05: ToolRegistry execute()

**优先级**: P0

- **Given** `ToolRegistry` 注册了可执行工具
- **When** 调用 `execute(tool_name, arguments)`
- **Then** 返回 `ToolResult`；tool_name 不存在时返回错误 `ToolResult`
- **验证方式**: 单元测试

## TOL-06: Web Search 工具

**优先级**: P1

- **Given** `web_search` 工具已注册，搜索引擎 provider 配置有效
- **When** 调用 `registry.execute("web_search", {"query": "Python async"})`
- **Then** 返回搜索结果列表（标题、摘要、URL）；API 失败时返回错误 ToolResult 不抛异常
- **验证方式**: 集成测试（mock HTTP）

## TOL-07: File Read/Write 工具

**优先级**: P1

- **Given** `read_file` / `write_file` 工具已注册
- **When** 读取存在的文件 / 写入新文件
- **Then** 返回文件内容 / 成功确认；路径不存在或权限不足时返回错误 ToolResult
- **验证方式**: 单元测试（临时文件）

## TOL-08: Code Execution 工具

**优先级**: P1

- **Given** `execute_code` 工具已注册，关联沙箱执行器
- **When** 调用 `registry.execute("execute_code", {"code": "print(1+1)", "language": "python"})`
- **Then** 返回 `ToolResult(content="2\n")`；超时或危险代码返回错误
- **验证方式**: 集成测试

## TOL-09: Tool 参数校验

**优先级**: P1

- **Given** 工具定义了 JSON Schema parameters（含 required 字段）
- **When** `execute()` 传入缺少 required 字段的 arguments
- **Then** 返回错误 `ToolResult`，说明缺失的参数名
- **验证方式**: 单元测试

## TOL-10: Tool 超时控制

**优先级**: P1

- **Given** 工具执行时间超过配置的 timeout（默认 30s）
- **When** `execute()` 被调用
- **Then** 返回错误 `ToolResult`，包含超时信息；不阻塞后续请求
- **验证方式**: 单元测试 — mock 慢工具 + asyncio.wait_for

## TOL-11: Tool 日志记录

**优先级**: P2

- **Given** 任意工具被调用
- **When** 执行完成（成功或失败）
- **Then** 记录 info 级别日志，包含 tool_name、arguments（脱敏）、duration_ms、success/failure
- **验证方式**: 单元测试 — 断言日志输出

## TOL-12: Tool 热注册/注销

**优先级**: P2

- **Given** `ToolRegistry` 已初始化并运行中
- **When** 调用 `register()` / `unregister()` 动态增删工具
- **Then** `get_definitions()` 立即反映变更，无需重启
- **验证方式**: 单元测试
