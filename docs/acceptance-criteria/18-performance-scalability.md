# 性能与可扩展性 — 验收标准

## PERF-01: Chat API — p95 延迟

**优先级**: P0

- **Given** 单轮对话（无工具调用），并发用户数 = 10
- **When** 压测 60s（wrk / locust）
- **Then** p95 latency < 2s；p99 < 3s；错误率 < 1%
- **验证方式**: 性能测试 — locust/wrk

## PERF-02: Chat API — 流式 TTFT

**优先级**: P0

- **Given** 流式对话请求
- **When** 压测中测量 Time To First Token
- **Then** p95 TTFT < 1s；p99 < 2s
- **验证方式**: 性能测试 — 自定义 metrics

## PERF-03: Chat API — 并发容量

**优先级**: P1

- **Given** 单实例部署（4C8G）
- **When** 逐步增加并发用户数
- **Then** 支持 >= 50 并发对话；超过容量时 graceful degradation（排队而非崩溃）
- **验证方式**: 性能测试 — ramp-up test

## PERF-04: Session Storage — 写入延迟

**优先级**: P1

- **Given** SQLite storage，session 包含 100 条消息
- **When** 追加新消息
- **Then** 单次写入 < 5ms（p95）；不阻塞 chat response
- **验证方式**: 性能测试 — benchmark

## PERF-05: Vector Search — 查询延迟

**优先级**: P1

- **Given** ChromaDB 索引 10,000 chunks
- **When** 执行向量搜索（top_k=5）
- **Then** p95 latency < 200ms；支持 metadata filter
- **验证方式**: 性能测试 — benchmark

## PERF-06: Embedding Generation — 批量处理

**优先级**: P1

- **Given** EmbeddingService，batch_size=32
- **When** 对 100 个 chunks 生成 embeddings
- **Then** 分批调用 API；总耗时 < batch_size=1 的 70%（利用批量优化）
- **验证方式**: 性能测试

## PERF-07: Cache Hit Rate — LLM Response

**优先级**: P2

- **Given** LLMResponseCache 启用，重复问题占比 30%
- **When** 处理 1000 个请求
- **Then** cache hit rate >= 25%（考虑 key 匹配率）；命中时延迟 < 10ms
- **验证方式**: 性能测试 — 模拟重复请求

## PERF-08: Memory Usage — 内存增长控制

**优先级**: P1

- **Given** 服务运行 24h，持续处理对话
- **When** 监控 RSS memory
- **Then** 内存增长 < 50MB/hour；无 memory leak（GC 正常回收）
- **验证方式**: 性能测试 — long-running test + memory profiler

## PERF-09: Startup Time — 冷启动

**优先级**: P1

- **Given** 容器从停止状态启动
- **When** 执行 `docker compose up`
- **Then** app 在 15s 内通过 health check；不依赖外部服务预热
- **验证方式**: 手动测试 — time docker compose up

## PERF-10: Horizontal Scaling — 多实例部署

**优先级**: P2

- **Given** 3 个 app 实例 + Redis session storage
- **When** 负载均衡分发请求
- **Then** 总容量线性增长（~3x）；session 跨实例共享无问题；无单点故障
- **验证方式**: 手动测试 — docker compose scale

## PERF-11: Database Connection Pool

**优先级**: P2

- **Given** SQLite/Redis connection pool 配置
- **When** 高并发请求
- **Then** 连接复用率 > 80%；无 connection exhaustion；pool size 可配置
- **验证方式**: 性能测试 — connection metrics

## PERF-12: Graceful Degradation — LLM API 降级

**优先级**: P2

- **Given** 主 LLM provider 不可用（超时/错误率 > 50%）
- **When** 请求到达
- **Then** 自动切换到 fallback provider（如配置了）；或返回友好错误；不阻塞其他功能
- **验证方式**: 集成测试 — mock provider failure
