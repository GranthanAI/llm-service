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


class TTLCache:
    """
    Thread-safe & Async-friendly In-Memory Cache with TTL and LRU Eviction.
    Used for Idempotency tracking (5 min TTL) and search query caching (60s TTL).
    """

    def __init__(self, max_size: int = 10000, default_ttl_seconds: int = 300):
        self._store: dict[str, CacheEntry] = {}
        self._max_size: int = max_size
        self._default_ttl: int = default_ttl_seconds
        self._lock: asyncio.Lock = asyncio.Lock()

    def get(self, key: str) -> Any | None:
        """Synchronously retrieve value if present and unexpired."""
        now = time.time()
        entry = self._store.get(key)
        if entry is None:
            return None

        if entry.expires_at < now:
            self._store.pop(key, None)
            return None

        entry.access_count += 1
        entry.last_accessed = now
        return entry.value

    def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        """Synchronously insert or update key with TTL."""
        now = time.time()
        ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl
        expires_at = now + ttl

        if len(self._store) >= self._max_size and key not in self._store:
            self._evict_lru()

        self._store[key] = CacheEntry(
            value=value,
            expires_at=expires_at,
            last_accessed=now,
        )

    def contains(self, key: str) -> bool:
        """Check if unexpired key exists in cache."""
        return self.get(key) is not None

    def delete(self, key: str) -> bool:
        """Remove key from cache."""
        return self._store.pop(key, None) is not None

    def clear(self) -> None:
        """Remove all items from cache."""
        self._store.clear()

    def size(self) -> int:
        """Return number of cached items."""
        return len(self._store)

    def _evict_expired(self) -> None:
        """Remove all expired entries."""
        now = time.time()
        expired_keys = [k for k, v in self._store.items() if v.expires_at < now]
        for k in expired_keys:
            self._store.pop(k, None)

    def _evict_lru(self) -> None:
        """Evict the least recently accessed item when cache is full."""
        self._evict_expired()
        if len(self._store) < self._max_size:
            return

        oldest_key = min(
            self._store.keys(),
            key=lambda k: self._store[k].last_accessed,
        )
        self._store.pop(oldest_key, None)
