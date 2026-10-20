"""
Manim job model and stores.

State machine (idempotent: a transition that is not allowed is refused, a repeated terminal write is a no-op):

    pending -> validating -> rendering -> ready
                  |             |
                  +--> repairing <--+      (repairing -> validating again, bounded by the attempt budget)
    any non-terminal -> failed | cancelled | expired          ready -> expired

Job ids are derived from (owner, canonical spec), so submitting the same animation twice returns the same job
(`create_if_absent`) and can never render twice or leak across tenants (the owner is part of the hash).
Two stores share one interface: in-memory (tests / single process) and Redis (API + separate worker processes).
"""

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Protocol


class JobState(StrEnum):
    PENDING = "pending"
    VALIDATING = "validating"
    RENDERING = "rendering"
    REPAIRING = "repairing"
    READY = "ready"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


TERMINAL = frozenset({JobState.READY, JobState.FAILED, JobState.CANCELLED, JobState.EXPIRED})
ACTIVE = frozenset({JobState.PENDING, JobState.VALIDATING, JobState.RENDERING, JobState.REPAIRING})
_ALLOWED = {
    JobState.PENDING: {JobState.VALIDATING, JobState.FAILED, JobState.CANCELLED, JobState.EXPIRED},
    JobState.VALIDATING: {JobState.RENDERING, JobState.REPAIRING, JobState.FAILED, JobState.CANCELLED, JobState.EXPIRED},
    JobState.RENDERING: {JobState.READY, JobState.REPAIRING, JobState.VALIDATING, JobState.FAILED, JobState.CANCELLED, JobState.EXPIRED},
    JobState.REPAIRING: {JobState.VALIDATING, JobState.FAILED, JobState.CANCELLED, JobState.EXPIRED},
    JobState.READY: {JobState.EXPIRED},
    JobState.FAILED: set(), JobState.CANCELLED: set(), JobState.EXPIRED: set(),
}


def can_transition(old: JobState, new: JobState) -> bool:
    return new in _ALLOWED[old]


def job_id_for(owner: str, spec_canonical: str) -> str:
    return "mj_" + hashlib.sha256(f"{owner}\x00{spec_canonical}".encode()).hexdigest()[:32]


@dataclass
class JobRecord:
    id: str
    owner: str
    spec: dict                         # the (validated) specification currently being rendered
    state: JobState = JobState.PENDING
    attempt: int = 0                   # attempts begun so far (1..max)
    max_attempts: int = 3
    failure: str = ""                  # FailureCategory value of the last failure (content-free)
    media_key: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    history: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        d = asdict(self)
        d["state"] = self.state.value
        return json.dumps(d)

    @classmethod
    def from_json(cls, raw: str | bytes) -> "JobRecord":
        d = json.loads(raw)
        d["state"] = JobState(d["state"])
        return cls(**d)

    def public(self) -> dict:
        """What the API returns: never the spec, the owner or an internal path."""
        return {"id": self.id, "state": self.state.value, "attempt": self.attempt, "max_attempts": self.max_attempts,
                "failure": self.failure or None, "ready": self.state == JobState.READY}


class JobStore(Protocol):
    async def create_if_absent(self, record: JobRecord) -> tuple[JobRecord, bool]: ...
    async def get(self, job_id: str) -> JobRecord | None: ...
    async def update(self, job_id: str, new_state: JobState, **fields) -> JobRecord | None: ...
    async def active_count(self, owner: str) -> int: ...
    async def hourly_count(self, owner: str, limit: int) -> bool: ...   # True if still within the limit (and counts it)
    async def enqueue(self, job_id: str) -> None: ...
    async def dequeue(self, timeout: float = 1.0) -> str | None: ...
    async def claim(self, job_id: str) -> bool: ...                    # one worker only


class MemoryJobStore:
    def __init__(self) -> None:
        self.jobs: dict[str, JobRecord] = {}
        self.queue: list[str] = []
        self.claimed: set[str] = set()
        self.hits: dict[str, list[float]] = {}

    async def create_if_absent(self, record):
        existing = self.jobs.get(record.id)
        # a finished-but-unusable job (failed/cancelled/expired) may be resubmitted; ready or active ones are reused
        if existing and existing.state in (JobState.READY, *ACTIVE):
            return existing, False
        self.jobs[record.id] = record
        self.claimed.discard(record.id)
        return record, True

    async def get(self, job_id):
        return self.jobs.get(job_id)

    async def update(self, job_id, new_state, **fields):
        rec = self.jobs.get(job_id)
        if rec is None:
            return None
        if rec.state == new_state and new_state in TERMINAL:
            return rec                                   # idempotent terminal write
        if not can_transition(rec.state, new_state):
            return None
        rec.state = new_state
        for k, v in fields.items():
            setattr(rec, k, v)
        rec.updated_at = time.time()
        return rec

    async def active_count(self, owner):
        return sum(1 for r in self.jobs.values() if r.owner == owner and r.state in ACTIVE)

    async def hourly_count(self, owner, limit):
        now = time.time()
        recent = [t for t in self.hits.get(owner, []) if now - t < 3600]
        if len(recent) >= limit:
            self.hits[owner] = recent
            return False
        recent.append(now)
        self.hits[owner] = recent
        return True

    async def enqueue(self, job_id):
        self.queue.append(job_id)

    async def dequeue(self, timeout=1.0):
        return self.queue.pop(0) if self.queue else None

    async def claim(self, job_id):
        if job_id in self.claimed:
            return False
        self.claimed.add(job_id)
        return True


class RedisJobStore:
    """Redis-backed store shared by the API and worker processes (requires the optional `redis` package)."""

    def __init__(self, url: str, ttl_seconds: int = 3600, prefix: str = "viz") -> None:
        import redis.asyncio as aioredis        # imported lazily: only needed when Manim is actually deployed
        self.r = aioredis.from_url(url, decode_responses=True)
        self.ttl = ttl_seconds
        self.p = prefix

    def _k(self, *parts: str) -> str:
        return ":".join((self.p, *parts))

    async def create_if_absent(self, record):
        key = self._k("job", record.id)
        if await self.r.set(key, record.to_json(), nx=True, ex=self.ttl):
            await self.r.sadd(self._k("owner", record.owner), record.id)
            return record, True
        existing = await self.get(record.id)
        if existing and existing.state in (JobState.READY, *ACTIVE):
            return existing, False
        await self.r.set(key, record.to_json(), ex=self.ttl)
        await self.r.delete(self._k("claim", record.id))
        return record, True

    async def get(self, job_id):
        raw = await self.r.get(self._k("job", job_id))
        return JobRecord.from_json(raw) if raw else None

    async def update(self, job_id, new_state, **fields):
        key = self._k("job", job_id)
        async with self.r.pipeline() as pipe:        # optimistic CAS
            for _ in range(5):
                try:
                    await pipe.watch(key)
                    raw = await pipe.get(key)
                    if not raw:
                        return None
                    rec = JobRecord.from_json(raw)
                    if rec.state == new_state and new_state in TERMINAL:
                        return rec
                    if not can_transition(rec.state, new_state):
                        return None
                    rec.state = new_state
                    for k, v in fields.items():
                        setattr(rec, k, v)
                    rec.updated_at = time.time()
                    pipe.multi()
                    pipe.set(key, rec.to_json(), ex=self.ttl)
                    await pipe.execute()
                    return rec
                except Exception as exc:             # WatchError -> retry
                    if exc.__class__.__name__ != "WatchError":
                        raise
        return None

    async def active_count(self, owner):
        n = 0
        for jid in await self.r.smembers(self._k("owner", owner)):
            rec = await self.get(jid)
            if rec and rec.state in ACTIVE:
                n += 1
        return n

    async def hourly_count(self, owner, limit):
        key = self._k("hour", owner)
        n = await self.r.incr(key)
        if n == 1:
            await self.r.expire(key, 3600)
        return n <= limit

    async def enqueue(self, job_id):
        await self.r.rpush(self._k("queue"), job_id)

    async def dequeue(self, timeout=1.0):
        item = await self.r.blpop(self._k("queue"), timeout=max(1, int(timeout)))
        return item[1] if item else None

    async def claim(self, job_id):
        return bool(await self.r.set(self._k("claim", job_id), "1", nx=True, ex=self.ttl))
