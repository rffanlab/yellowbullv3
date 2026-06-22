# 缓存层（Cache Layer）详细设计

## 1. 职责边界

| 缓存类型 | 说明 | TTL |
|---------|------|-----|
| **Embedding 缓存** | 相同文本的向量结果复用，避免重复调用 Embedding API | 24h |
| **LLM 响应缓存** | 相同 prompt + model 参数的响应缓存（语义哈希） | 1h |
| **工具结果缓存** | 幂等工具的返回结果缓存（如时间查询、计算器） | 5min |
| **配置缓存** | YAML 解析后的配置对象缓存，热加载时失效 | 手动失效 |

---

## 2. 缓存抽象 `cache/base.py`

```python
"""
Cache abstraction。

支持：内存 LRU（开发/测试）、Redis（生产）。
"""

from abc import ABC, abstractmethod
from typing import Any, Optional


class CacheBackend(ABC):
    """缓存后端接口"""

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        ...

    @abstractmethod
    async def set(self, key: str, value: Any, ttl_seconds: int = 3600) -> None:
        ...

    @abstractmethod
    async def delete(self, key: str) -> bool:
        ...

    @abstractmethod
    async def invalidate_pattern(self, pattern: str) -> int:
        """批量失效（glob pattern）"""
        ...

    @abstractmethod
    async def stats(self) -> dict:
        """返回命中率等统计信息"""
        ...


class LRUCacheBackend(CacheBackend):
    """内存 LRU 缓存（开发/测试用）"""

    def __init__(self, maxsize: int = 1024):
        from collections import OrderedDict
        import time
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._maxsize = maxsize
        self._hits = 0
        self._misses = 0

    async def get(self, key: str) -> Optional[Any]:
        if key not in self._cache:
            self._misses += 1
            return None

        value, expiry = self._cache[key]
        if expiry and time.time() > expiry:
            del self._cache[key]
            self._misses += 1
            return None

        self._hits += 1
        self._cache.move_to_end(key)
        return value

    async def set(self, key: str, value: Any, ttl_seconds: int = 3600) -> None:
        import time
        expiry = time.time() + ttl_seconds if ttl_seconds > 0 else None

        if len(self._cache) >= self._maxsize and key not in self._cache:
            self._cache.popitem(last=False)

        self._cache[key] = (value, expiry)
        self._cache.move_to_end(key)

    async def delete(self, key: str) -> bool:
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    async def invalidate_pattern(self, pattern: str) -> int:
        import fnmatch
        keys_to_delete = [k for k in self._cache if fnmatch.fnmatch(k, pattern)]
        for k in keys_to_delete:
            del self._cache[k]
        return len(keys_to_delete)

    async def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "maxsize": self._maxsize,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 4) if total > 0 else 0,
        }


class RedisCacheBackend(CacheBackend):
    """Redis 缓存（生产用）"""

    def __init__(self, url: str = "redis://localhost:6379/0"):
        import redis.asyncio as aioredis
        self._client = aioredis.from_url(url, decode_responses=True)
        self._hits = 0
        self._misses = 0

    async def get(self, key: str) -> Optional[Any]:
        import json
        value = await self._client.get(key)
        if value is None:
            self._misses += 1
            return None
        self._hits += 1
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value

    async def set(self, key: str, value: Any, ttl_seconds: int = 3600) -> None:
        import json
        serialized = json.dumps(value, ensure_ascii=False, default=str)
        if ttl_seconds > 0:
            await self._client.setex(key, ttl_seconds, serialized)
        else:
            await self._client.set(key, serialized)

    async def delete(self, key: str) -> bool:
        result = await self._client.delete(key)
        return result > 0

    async def invalidate_pattern(self, pattern: str) -> int:
        keys = await self._client.keys(pattern)
        if keys:
            return await self._client.delete(*keys)
        return 0

    async def stats(self) -> dict:
        total = self._hits + self._misses
        info = await self._client.info("stats")
        return {
            "backend": "redis",
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 4) if total > 0 else 0,
            "redis_keys": info.get("db0", {}).get("keys", 0),
        }


def create_cache_backend(config: dict) -> CacheBackend:
    """Factory function"""
    backend_type = config.get("type", "lru")

    if backend_type == "lru":
        return LRUCacheBackend(maxsize=config.get("maxsize", 1024))
    elif backend_type == "redis":
        return RedisCacheBackend(url=config.get("url", "redis://localhost:6379/0"))
    else:
        raise ValueError(f"Unknown cache backend: {backend_type}")
```

---

## 3. Embedding 缓存 `cache/embedding_cache.py`

```python
"""
Embedding 结果缓存。

相同文本 → 相同向量，无需重复调用 API。
Key 格式：embedding:{hash(text)}:{model_name}
"""

import hashlib


class EmbeddingCache:
    """Embedding 缓存层"""

    def __init__(self, backend: "CacheBackend", ttl_seconds: int = 86400):
        self._backend = backend
        self._ttl = ttl_seconds

    @staticmethod
    def _make_key(text: str, model: str) -> str:
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        return f"embedding:{text_hash}:{model}"

    async def get(self, text: str, model: str):
        """尝试从缓存获取 embedding"""
        key = self._make_key(text, model)
        return await self._backend.get(key)

    async def set(self, text: str, model: str, embedding: list[float]):
        """缓存 embedding 结果"""
        key = self._make_key(text, model)
        await self._backend.set(key, embedding, ttl_seconds=self._ttl)


class CachedEmbeddingProvider:
    """带缓存的 Embedding Provider 包装器"""

    def __init__(self, provider: "EmbeddingProvider", cache: EmbeddingCache):
        self._provider = provider
        self._cache = cache
        self._model_name = getattr(provider, "_model", "default")

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        results = []
        for text in texts:
            cached = await self._cache.get(text, self._model_name)
            if cached is not None:
                results.append(cached)
            else:
                embeddings = await self._provider.embed_documents([text])
                result = embeddings[0]
                await self._cache.set(text, self._model_name, result)
                results.append(result)
        return results

    async def embed_query(self, text: str) -> list[float]:
        cached = await self._cache.get(text, self._model_name)
        if cached is not None:
            return cached

        embedding = await self._provider.embed_query(text)
        await self._cache.set(text, self._model_name, embedding)
        return embedding

    @property
    def dimension(self) -> int:
        return self._provider.dimension
```

---

## 4. LLM 响应缓存 `cache/llm_cache.py`

```python
"""
LLM 响应缓存。

基于 prompt 的语义哈希，相同输入复用之前结果。
仅对非流式调用生效（流式调用的实时性要求更高）。
"""

import hashlib
import json


class LLMResponseCache:
    """LLM 响应缓存"""

    def __init__(self, backend: "CacheBackend", ttl_seconds: int = 3600):
        self._backend = backend
        self._ttl = ttl_seconds

    @staticmethod
    def _make_key(messages: list[dict], model: str, temperature: float) -> str:
        """生成缓存 key"""
        # 对 messages 做确定性序列化
        msg_str = json.dumps(messages, ensure_ascii=False, sort_keys=True)
        content_hash = hashlib.sha256(msg_str.encode()).hexdigest()[:16]
        return f"llm:{content_hash}:{model}:t{temperature}"

    async def get(self, messages: list[dict], model: str, temperature: float):
        """尝试从缓存获取 LLM 响应"""
        # 温度 > 0 时不缓存（非确定性输出）
        if temperature > 0:
            return None

        key = self._make_key(messages, model, temperature)
        return await self._backend.get(key)

    async def set(self, messages: list[dict], model: str, temperature: float, response: dict):
        """缓存 LLM 响应"""
        if temperature > 0:
            return

        key = self._make_key(messages, model, temperature)
        await self._backend.set(key, response, ttl_seconds=self._ttl)


class CachedLLMProvider:
    """带缓存的 LLM Provider 包装器"""

    def __init__(self, provider: "LLMProvider", cache: LLMResponseCache):
        self._provider = provider
        self._cache = cache

    async def chat(self, messages: list[dict], stream: bool = True, **kwargs) -> any:
        model = getattr(self._provider, "_model", "default")
        temperature = kwargs.get("temperature", 0.7)

        if not stream and temperature == 0:
            cached = await self._cache.get(messages, model, temperature)
            if cached is not None:
                return type("ChatResponse", (), {"content": cached})()

        response = await self._provider.chat(messages, stream=stream, **kwargs)

        if not stream and temperature == 0:
            await self._cache.set(
                messages, model, temperature, response.content
            )

        return response
```

---

## 5. 工具结果缓存 `cache/tool_cache.py`

```python
"""
工具执行结果缓存。

仅对幂等工具生效（如时间查询、计算器、只读搜索）。
Key 格式：tool:{tool_name}:{hash(args)}
"""

import hashlib
import json


class ToolResultCache:
    """工具结果缓存"""

    def __init__(self, backend: "CacheBackend", ttl_seconds: int = 300):
        self._backend = backend
        self._ttl = ttl_seconds

    @staticmethod
    def _make_key(tool_name: str, args: dict) -> str:
        args_str = json.dumps(args, sort_keys=True, default=str)
        args_hash = hashlib.sha256(args_str.encode()).hexdigest()[:12]
        return f"tool:{tool_name}:{args_hash}"

    async def get(self, tool_name: str, args: dict):
        key = self._make_key(tool_name, args)
        return await self._backend.get(key)

    async def set(self, tool_name: str, args: dict, result: any):
        key = self._make_key(tool_name, args)
        await self._backend.set(key, result, ttl_seconds=self._ttl)


class CachedToolExecutor:
    """带缓存的工具执行器"""

    def __init__(self, registry: "ToolRegistry", cache: ToolResultCache):
        self._registry = registry
        self._cache = cache

    async def execute(self, tool_name: str, args: dict) -> any:
        tool = self._registry.get(tool_name)

        # 非幂等工具不缓存
        if not getattr(tool, "idempotent", False):
            return await tool.execute(**args)

        cached = await self._cache.get(tool_name, args)
        if cached is not None:
            return cached

        result = await tool.execute(**args)
        await self._cache.set(tool_name, args, result)
        return result
```

---

## 6. YAML 配置 `cache` section

```yaml
# config/settings.yaml (新增)
cache:
  backend:
    type: "lru"                 # lru | redis
    maxsize: 2048               # LRU 最大条目数（仅 lru）
    url: "redis://localhost:6379/0"  # Redis URL（仅 redis）

  embedding:
    enabled: true
    ttl_seconds: 86400          # 24h

  llm_response:
    enabled: false              # 默认关闭（流式调用不适用）
    ttl_seconds: 3600           # 1h

  tool_result:
    enabled: true
    ttl_seconds: 300            # 5min
```

---

## 7. 设计总结

| 特性 | 实现方式 |
|------|---------|
| **缓存后端** | LRU（内存）/ Redis，ABC + Factory 切换 |
| **Embedding 缓存** | SHA-256 hash key，TTL 24h，大幅降低 API cost |
| **LLM 响应缓存** | prompt 语义哈希，仅 temperature=0 时生效 |
| **工具结果缓存** | 幂等工具自动缓存，非幂等跳过 |
| **Prometheus 指标** | hit_rate、cache_size 暴露为 metrics |
