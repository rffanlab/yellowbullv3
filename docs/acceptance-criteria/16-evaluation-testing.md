# 评估与测试 — 验收标准

## EVAL-01: Unit Test Coverage

**优先级**: P0

- **Given** `pytest tests/unit` 执行完成
- **When** 生成覆盖率报告（`coverage report`）
- **Then** 总覆盖率 >= 80%；核心模块（AgentCore、LLM abstraction、ToolRegistry）>= 90%
- **验证方式**: CI pipeline — coverage check

## EVAL-02: Integration Test Suite

**优先级**: P1

- **Given** `pytest tests/integration` 执行完成（Docker compose 环境）
- **When** 所有集成测试通过
- **Then** 覆盖 API endpoints、LLM providers（mock）、Tool execution、Session persistence；无 flaky tests
- **验证方式**: CI pipeline

## EVAL-03: LLM Response Eval — 准确性

**优先级**: P1

- **Given** `eval/` 目录包含评估脚本和测试数据集
- **When** 运行 `python -m eval.run --dataset qa_accuracy`
- **Then** 输出准确率指标（correct / total）；支持指定 provider/model 对比
- **验证方式**: 手动运行 — 查看报告

## EVAL-04: LLM Response Eval — 工具调用正确性

**优先级**: P1

- **Given** 测试集包含需要工具调用的问题
- **When** 运行评估
- **Then** 统计 tool_call_rate（应调用工具的占比）、tool_success_rate（工具执行成功率）；输出详细报告
- **验证方式**: 手动运行

## EVAL-05: LLM Response Eval — 延迟评估

**优先级**: P2

- **Given** 测试集包含 N 个问题
- **When** 运行延迟评估
- **Then** 记录 TTFT（Time To First Token）、TPOT（Time Per Output Token）、total_latency；输出 p50/p95/p99 分位数
- **验证方式**: 手动运行

## EVAL-06: LLM Response Eval — 成本评估

**优先级**: P2

- **Given** LLM provider 返回 usage 信息 + 已知定价
- **When** 运行评估
- **Then** 计算总 token 消耗和预估费用；按 provider/model 分组统计
- **验证方式**: 手动运行

## EVAL-07: Benchmark Dataset — 内置测试集

**优先级**: P1

- **Given** `eval/datasets/` 包含标准测试集
- **When** 加载数据集
- **Then** 包含至少以下类别：通用问答、代码生成、工具调用、多轮对话；每个类别 >= 20 条样本
- **验证方式**: 手动检查

## EVAL-08: Benchmark Dataset — 自定义数据集

**优先级**: P2

- **Given** `eval/datasets/` 目录结构
- **When** 添加新的 JSON dataset（`{"questions": [{"input": "...", "expected": "..."}]}`）
- **Then** 评估脚本自动发现并支持新数据集；无需修改代码
- **验证方式**: 手动验证

## EVAL-09: Eval Report — HTML 报告

**优先级**: P2

- **Given** 评估完成
- **When** 生成报告
- **Then** 输出 HTML 文件，包含各指标表格、provider 对比图表、失败案例详情；可离线查看
- **验证方式**: 手动检查

## EVAL-10: Eval Report — CI 集成

**优先级**: P2

- **Given** CI workflow 包含 eval stage（可选触发）
- **When** 运行 `python -m eval.run`
- **Then** 结果上传为 artifact；指标变化生成 PR comment（如准确率下降 > 5% 时 warning）
- **验证方式**: 手动验证 — 查看 CI artifacts
