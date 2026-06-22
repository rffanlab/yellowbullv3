# 缓存层 — 验收标准

## CACHE-01: CacheManager — get/set/delete

**优先级**: P0

- **Given** `CacheManager` 实例（Redis backend）
- **When** 调用 `set("key", "value", ttl=60)` → `get("key")` → `delete("key")`
- **Then** set 返回 True；get 返回 `"value"`；delete 后 get 返回 None
- **验证方式**: 集成测试（Redis）

## CACHE-02: CacheManager — TTL 过期

**优先级**: P0

- **Given** `set("key", "value", ttl=1)`
- **When** 等待 2s 后调用 `get("key")`
- **Then** 返回 None（已过期）；不抛异常
- **验证方式**: 集成测试

## CACHE-03: CacheManager — JSON 序列化

**优先级**: P0

- **Given** Redis backend
- **When** `set("key", {"nested": [1, 2, 3]})` → `get("key")`
- **Then** 自动 JSON 序列化/反序列化；返回 dict 类型，结构一致
- **验证方式**: 集成测试

## CACHE-04: LLM Response Cache — 缓存命中

**优先级**: P1

- **Given** `LLMResponseCache` 已包装 BaseLLM
- **When** 用相同 messages + model 调用 `chat()` 两次
- **Then** 第二次从 cache 返回，不触发真实 API 调用；响应内容一致
- **验证方式**: 单元测试 — mock LLM + 断言调用次数 = 1

## CACHE-05: LLM Response Cache — Key 生成

**优先级**: P1

- **Given** `LLMResponseCache` 实例
- **When** messages 顺序不同或 content 微小差异
- **Then** 生成不同的 cache key；不会错误命中
- **验证方式**: 单元测试 — 断言 `_cache_key()` 输出

## CACHE-06: LLM Response Cache — Stream 不支持缓存

**优先级**: P1

- **Given** `LLMResponseCache` 实例
- **When** 调用 `chat_stream()`
- **Then** 直接透传到底层 LLM，不走缓存；不抛异常
- **验证方式**: 单元测试

## CACHE-07: Embedding Cache — 文本去重

**优先级**: P1

- **Given** `EmbeddingCache` 已包装 EmbeddingService
- **When** 对相同文本调用 `embed()` 两次
- **Then** 第二次从 cache 返回；embedding API 只被调用一次
- **验证方式**: 单元测试 — mock embedding + 断言调用次数

## CACHE-08: LRU In-Memory Fallback

**优先级**: P1

- **Given** `CacheManager` 配置 Redis 但连接失败
- **When** 调用 `set()` / `get()`
- **Then** 自动降级为 LRU in-memory cache；记录 warning 日志；功能正常
- **验证方式**: 单元测试 — mock Redis connection error

## CACHE-09: Cache Stats

**优先级**: P2

- **Given** `CacheManager` 已处理多次 get/set
- **When** 调用 `get_stats()`
- **Then** 返回 `{hits, misses, sets, evictions}`；hits + misses = total gets
- **验证方式**: 单元测试

## CACHE-10: Cache Clear

**优先级**: P2

- **Given** cache 中存在多个 entries
- **When** 调用 `clear()`
- **Then** 所有 entries 被清除；后续 get 全部 miss
- **验证方式**: 集成测试
