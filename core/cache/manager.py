"""In-memory cache manager with TTL expiration and JSON serialization."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class _CacheEntry:
    """Internal cache entry with expiration tracking."""

    value: Any
    expires_at: float | None  # None means no expiry


class CacheManager:
    """Thread-unsafe in-memory cache (suitable for single-process agents).

    Supports TTL-based expiration and automatic JSON serialization.
    For production multi-instance deployments, replace with Redis backend.
    """

    def __init__(self, default_ttl: float | None = 300.0) -> None:
        """Initialize cache manager.

        Args:
            default_ttl: Default time-to-live in seconds (None = no expiry).
        """
        self._store: dict[str, _CacheEntry] = {}
        self._default_ttl = default_ttl

    # ── Core operations ────────────────────────────────────────────

    def get(self, key: str) -> Any | None:
        """Get a value from cache. Returns None if missing or expired."""
        entry = self._store.get(key)
        if entry is None:
            return None
        if self._is_expired(entry):
            del self._store[key]
            logger.debug("Cache miss (expired): key=%s", key)
            return None
        return entry.value

    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        """Store a value in cache.

        Args:
            key: Cache key.
            value: Value to store (will be JSON-serialized if string/dict/list).
            ttl: Time-to-live in seconds. Uses default_ttl if not specified.
        """
        expires_at = None
        effective_ttl = ttl if ttl is not None else self._default_ttl
        if effective_ttl is not None:
            expires_at = time.monotonic() + effective_ttl

        self._store[key] = _CacheEntry(value=value, expires_at=expires_at)
        logger.debug("Cache set: key=%s ttl=%s", key, effective_ttl)

    def delete(self, key: str) -> bool:
        """Delete a key from cache. Returns True if key existed."""
        if key in self._store:
            del self._store[key]
            logger.debug("Cache deleted: key=%s", key)
            return True
        return False

    # ── JSON serialization helpers ─────────────────────────────────

    def get_json(self, key: str) -> Any | None:
        """Get and deserialize a JSON value from cache."""
        raw = self.get(key)
        if raw is None:
            return None
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                return raw
        return raw

    def set_json(self, key: str, value: Any, ttl: float | None = None) -> None:
        """Serialize to JSON and store in cache."""
        serialized = json.dumps(value, ensure_ascii=False, default=str)
        self.set(key, serialized, ttl=ttl)

    # ── Utility ────────────────────────────────────────────────────

    def clear(self) -> int:
        """Remove all entries. Returns number of cleared keys."""
        count = len(self._store)
        self._store.clear()
        logger.info("Cache cleared %d entries", count)
        return count

    @property
    def size(self) -> int:
        """Current number of cache entries (including potentially expired)."""
        return len(self._store)

    def cleanup_expired(self) -> int:
        """Manually remove all expired entries. Returns count removed."""
        expired_keys = [
            k for k, v in self._store.items() if self._is_expired(v)
        ]
        for key in expired_keys:
            del self._store[key]
        if expired_keys:
            logger.debug("Cache cleanup: removed %d expired entries", len(expired_keys))
        return len(expired_keys)

    # ── Internal ───────────────────────────────────────────────────

    @staticmethod
    def _is_expired(entry: _CacheEntry) -> bool:
        if entry.expires_at is None:
            return False
        return time.monotonic() > entry.expires_at
