# 多智能体协作 — 验收标准

## MAA-01: AgentRouter — 基于意图的路由

**优先级**: P0

- **Given** `AgentRouter` 注册了 "research"、"coding"、"analysis" 三个子 agent
- **When** 调用 `route(user_message="帮我搜索 Python async 文档")`
- **Then** LLM 判断意图，返回目标 agent name；路由结果可配置（非硬编码）
- **验证方式**: 单元测试 — mock LLM intent detection

## MAA-02: AgentRouter — 路由失败回退

**优先级**: P1

- **Given** `AgentRouter` 已初始化
- **When** LLM 无法判断意图或返回未知 agent name
- **Then** 使用配置的 default_agent；记录 warning 日志
- **验证方式**: 单元测试

## MAA-03: AgentOrchestrator — Plan 生成

**优先级**: P1

- **Given** `AgentOrchestrator` + 注册的子 agents
- **When** 调用 `orchestrate(complex_task)`
- **Then** LLM 生成执行计划（steps list），每步指定 agent_name + task_description；返回 plan_id
- **验证方式**: 集成测试 — mock LLM

## MAA-04: AgentOrchestrator — Plan 执行

**优先级**: P1

- **Given** 已生成的 execution plan
- **When** 调用 `execute_plan(plan_id)`
- **Then** 按步骤顺序执行；每步结果传递给下一步作为上下文；最终返回汇总结果
- **验证方式**: 集成测试

## MAA-05: AgentOrchestrator — 步骤失败处理

**优先级**: P1

- **Given** plan 中某一步骤执行失败（agent 返回错误）
- **When** 执行 plan
- **Then** 记录失败步骤；尝试继续执行后续不依赖该步骤的步骤；最终结果包含部分成功信息
- **验证方式**: 集成测试 — mock agent failure

## MAA-06: AgentOrchestrator — Plan 状态查询

**优先级**: P2

- **Given** 正在执行的 plan
- **When** 调用 `get_plan_status(plan_id)`
- **Then** 返回 `{plan_id, status, current_step, completed_steps, results}`；status ∈ {planning, executing, completed, failed}
- **验证方式**: 集成测试

## MAA-07: AgentRegistry — Agent 注册与发现

**优先级**: P1

- **Given** `AgentRegistry` 实例
- **When** 调用 `register(name, agent_instance, description)` / `list_agents()`
- **Then** agent 可被路由/编排器发现；返回包含 name + description 的列表（供 LLM 决策）
- **验证方式**: 单元测试

## MAA-08: Agent 间消息传递

**优先级**: P1

- **Given** 两个子 agent A、B
- **When** A 的输出作为 B 的输入
- **Then** 消息格式统一（Message protocol）；context 正确传递；不丢失信息
- **验证方式**: 集成测试

## MAA-09: Agent 隔离

**优先级**: P1

- **Given** 多个子 agent 独立配置（不同 LLM model、tools）
- **When** 并行执行不同 agents
- **Then** sessions 互不干扰；工具调用各自独立；无共享状态泄漏
- **验证方式**: 集成测试

## MAA-10: Agent 超时与熔断

**优先级**: P2

- **Given** 子 agent 配置了 timeout + circuit breaker（连续失败 N 次后熔断）
- **When** agent 持续失败达到阈值
- **Then** 路由/编排器不再调用该 agent；返回降级结果；记录 error 日志
- **验证方式**: 集成测试 — mock agent failures
