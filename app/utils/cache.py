"""
In-Memory TTL & LRU Cache Implementation.
Implements LLD v2.0 Section 24.
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class CacheEntry:
    """Cache entry metadata and payload."""

    value: Any
    expires_at: float
    access_count: int = 1
    last_accessed: float = 0.0
    access_seq: int = 0


class TTLCache:
    """
    Thread-safe & Async-friendly In-Memory Cache with TTL and LRU Eviction.
    Used for Idempotency tracking (5 min TTL), search query caching (60s TTL), and quota caching (5s TTL).
    Implements LLD v2.0 Section 24.
    """

    def __init__(self, max_size: int = 10000, default_ttl_seconds: int = 300):
        self._store: dict[str, CacheEntry] = {}
        self._max_size: int = max_size
        self._default_ttl: int = default_ttl_seconds
        self._lock: asyncio.Lock = asyncio.Lock()
        self._hits: int = 0
        self._misses: int = 0
        self._evictions: int = 0
        self._seq: int = 0

    def get(self, key: str) -> Any | None:
        """Synchronously retrieve value if present and unexpired."""
        now = time.time()
        entry = self._store.get(key)
        if entry is None:
            self._misses += 1
            return None

        if entry.expires_at < now:
            self._store.pop(key, None)
            self._misses += 1
            self._evictions += 1
            return None

        self._seq += 1
        entry.access_count += 1
        entry.last_accessed = now
        entry.access_seq = self._seq
        self._hits += 1
        return entry.value

    def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        """Synchronously insert or update key with TTL."""
        now = time.time()
        ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl
        expires_at = now + ttl

        if len(self._store) >= self._max_size and key not in self._store:
            self._evict_lru()

        self._seq += 1
        self._store[key] = CacheEntry(
            value=value,
            expires_at=expires_at,
            last_accessed=now,
            access_seq=self._seq,
        )

    async def get_async(self, key: str) -> Any | None:
        """Async-safe retrieval."""
        async with self._lock:
            return self.get(key)

    async def set_async(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        """Async-safe mutation."""
        async with self._lock:
            self.set(key, value, ttl_seconds=ttl_seconds)

    def contains(self, key: str) -> bool:
        """Check if unexpired key exists in cache."""
        return self.get(key) is not None

    def delete(self, key: str) -> bool:
        """Remove key from cache."""
        return self._store.pop(key, None) is not None

    def clear(self) -> None:
        """Remove all items from cache."""
        self._store.clear()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._seq = 0

    def size(self) -> int:
        """Return number of cached items."""
        return len(self._store)

    def get_stats(self) -> dict[str, int]:
        """Return cache hit, miss, eviction, and size statistics."""
        return {
            "size": len(self._store),
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "max_size": self._max_size,
        }

    def evict_expired(self) -> int:
        """Remove all expired entries and return count of evicted items."""
        now = time.time()
        expired_keys = [k for k, v in self._store.items() if v.expires_at < now]
        for k in expired_keys:
            self._store.pop(k, None)
            self._evictions += 1
        return len(expired_keys)

    def _evict_lru(self) -> None:
        """Evict the least recently accessed item when cache is full."""
        self.evict_expired()
        if len(self._store) < self._max_size:
            return

        oldest_key = min(
            self._store.keys(),
            key=lambda k: (self._store[k].access_seq, self._store[k].last_accessed),
        )
        self._store.pop(oldest_key, None)
        self._evictions += 1


def create_web_search_cache() -> TTLCache:
    """Factory creating Web Search Results Cache (LLD Section 24.3: TTL 60s, Max Size 1,000)."""
    return TTLCache(max_size=1000, default_ttl_seconds=60)


def create_idempotency_cache() -> TTLCache:
    """Factory creating Kafka Idempotency Cache (LLD Section 24.3: TTL 300s, Max Size 10,000)."""
    return TTLCache(max_size=10000, default_ttl_seconds=300)


def create_quota_cache() -> TTLCache:
    """Factory creating Provider Quota Cache (LLD Section 24.3: TTL 5s, Max Size 10)."""
    return TTLCache(max_size=10, default_ttl_seconds=5)
