"""
A tiny async Redis client (RESP over a socket) with the few commands the shared Groq budget needs, spelled like redis-py so
`GroqBudget` runs unchanged on either. llm-service has no redis package in its environment and takes no new dependency for
this. One short connection per command: the budget makes a handful of calls per model request.
"""

import asyncio
from typing import Any
from urllib.parse import urlparse


class MiniRedisError(Exception):
    pass


def _encode(*parts: Any) -> bytes:
    out = [f"*{len(parts)}\r\n".encode()]
    for p in parts:
        b = p if isinstance(p, bytes) else str(p).encode()
        out.append(f"${len(b)}\r\n".encode() + b + b"\r\n")
    return b"".join(out)


async def _read(reader: asyncio.StreamReader) -> Any:
    line = await reader.readline()
    if not line:
        raise MiniRedisError("connection closed")
    kind, rest = line[:1], line[1:-2]
    if kind == b"+":
        return rest.decode()
    if kind == b"-":
        raise MiniRedisError(rest.decode())
    if kind == b":":
        return int(rest)
    if kind == b"$":
        n = int(rest)
        if n < 0:
            return None
        data = await reader.readexactly(n + 2)
        return data[:-2]
    if kind == b"*":
        n = int(rest)
        return None if n < 0 else [await _read(reader) for _ in range(n)]
    raise MiniRedisError(f"unexpected reply {line!r}")


class MiniRedis:
    def __init__(self, url: str, timeout_s: float = 0.5) -> None:
        u = urlparse(url)
        self.host, self.port = u.hostname or "127.0.0.1", u.port or 6379
        self.db = int((u.path or "/0").lstrip("/") or 0)
        self.password = u.password
        self.timeout_s = timeout_s

    async def _call(self, *parts: Any) -> Any:
        async def run() -> Any:
            reader, writer = await asyncio.open_connection(self.host, self.port)
            try:
                if self.password:
                    writer.write(_encode("AUTH", self.password))
                    await _read(reader)
                if self.db:
                    writer.write(_encode("SELECT", self.db))
                    await _read(reader)
                writer.write(_encode(*parts))
                await writer.drain()
                return await _read(reader)
            finally:
                writer.close()

        return await asyncio.wait_for(run(), self.timeout_s)

    async def get(self, key: str) -> bytes | None:
        return await self._call("GET", key)

    async def set(self, key: str, value: Any, ex: int | None = None) -> Any:
        return await (self._call("SET", key, value, "EX", ex) if ex else self._call("SET", key, value))

    async def ttl(self, key: str) -> int:
        return await self._call("TTL", key)

    async def exists(self, key: str) -> int:
        return await self._call("EXISTS", key)

    async def decrby(self, key: str, n: int) -> int:
        return await self._call("DECRBY", key, n)

    async def incrby(self, key: str, n: int) -> int:
        return await self._call("INCRBY", key, n)

    async def expire(self, key: str, seconds: int) -> int:
        return await self._call("EXPIRE", key, seconds)

    async def zadd(self, key: str, mapping: dict[str, float]) -> int:
        args: list[Any] = []
        for member, score in mapping.items():
            args += [score, member]
        return await self._call("ZADD", key, *args)

    async def zrem(self, key: str, member: str) -> int:
        return await self._call("ZREM", key, member)

    async def zremrangebyscore(self, key: str, low: float, high: float) -> int:
        return await self._call("ZREMRANGEBYSCORE", key, low, high)

    async def zrange(self, key: str, start: int, stop: int, withscores: bool = False) -> list:
        raw = await (self._call("ZRANGE", key, start, stop, "WITHSCORES") if withscores else self._call("ZRANGE", key, start, stop))
        if not withscores:
            return raw
        return [(raw[i], float(raw[i + 1])) for i in range(0, len(raw), 2)]
