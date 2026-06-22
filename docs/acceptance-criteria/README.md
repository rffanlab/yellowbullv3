# 验收标准索引

每个子模块对应一份验收标准文档，基于详细设计文档编写。

| # | 模块 | 验收文档 |
|---|------|----------|
| 01 | 配置系统 | [01-config-system.md](./01-config-system.md) |
| 02 | LLM 抽象层 | [02-llm-abstraction.md](./02-llm-abstraction.md) |
| 03 | 工具系统 | [03-tool-system.md](./03-tool-system.md) |
| 04 | Session 管理 | [04-session-manager.md](./04-session-manager.md) |
| 05 | Agent Core | [05-agent-core.md](./05-agent-core.md) |
| 06 | API Server | [06-api-server.md](./06-api-server.md) |
| 07 | 可观测性 | [07-observability.md](./07-observability.md) |
| 08 | RAG | [08-rag.md](./08-rag.md) |
| 09 | 记忆系统 | [09-memory-system.md](./09-memory-system.md) |
| 10 | 安全治理 | [10-security-governance.md](./10-security-governance.md) |
| 11 | 缓存层 | [11-cache-layer.md](./11-cache-layer.md) |
| 12 | 消息队列与异步任务 | [12-message-queue.md](./12-message-queue.md) |
| 13 | 部署与 DevOps | [13-deployment-devops.md](./13-deployment-devops.md) |
| 14 | 代码执行沙箱 | [14-code-execution.md](./14-code-execution.md) |
| 15 | 多智能体协作 | [15-multi-agent-collaboration.md](./15-multi-agent-collaboration.md) |
| 16 | NL2SQL | [16-nl2sql-database-operations.md](./16-nl2sql-database-operations.md) |
| 17 | Prompt 模板管理 | [17-prompt-template-management.md](./17-prompt-template-management.md) |

## 验收标准格式说明

每条验收标准包含：

- **ID**: 唯一标识（模块缩写 + 序号）
- **标题**: 功能描述
- **优先级**: P0（必须）、P1（重要）、P2（可选）
- **验收条件**: Given/When/Then 格式的判定条件
- **验证方式**: 单元测试、集成测试或手动验证
