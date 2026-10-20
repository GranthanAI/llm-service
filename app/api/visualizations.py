"""
Visualization endpoints (routed through the API Gateway as /api/v1/visualizations/...).

The gateway authenticates the caller and sets X-User-Id from the JWT (any client-supplied value is stripped there). This
service is not publicly exposed, so the header is trusted but still must be a UUID. Every job is owner-scoped: another user's
job answers 404 (never 403) so ids cannot be probed. Responses never contain specs, filesystem paths, source text or errors.

`/api/internal/visualizations/repair` is for the isolated worker only (shared token) and is NOT routed by the gateway.
"""

import asyncio
import hmac
import json
import re
import uuid
from typing import Literal

import structlog
from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.api.dependencies import ContainerDep
from app.utils import metrics as viz_metrics
from app.visualization.jobs import ACTIVE, JobRecord, JobState, MemoryJobStore, job_id_for
from app.visualization.manim_templates import SpecError, template_catalog, validate_spec
from app.visualization.policy import FailureCategory, user_message
from app.visualization.renderer import MediaStore
from app.visualization.repair import repair_block, validate_candidate
from app.visualization.runner import run_job

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/v1/visualizations", tags=["visualizations"])
internal_router = APIRouter(prefix="/internal/visualizations", tags=["visualizations-internal"])
_JOB_ID = re.compile(r"^mj_[0-9a-f]{32}$")


def _user(x_user_id: str | None) -> str:
    try:
        return str(uuid.UUID(x_user_id or ""))
    except ValueError:
        raise HTTPException(status_code=401, detail="Authentication required") from None


def _services(container):
    """Lazily create the job store + media store (Redis when a separate worker is deployed, otherwise in-process memory)."""
    if getattr(container, "viz_store", None) is None:
        cfg = container.config
        store = None
        if cfg.manim_enabled and cfg.visualization_redis_url and not getattr(cfg, "manim_inline_worker", False):
            try:
                from app.visualization.jobs import RedisJobStore
                store = RedisJobStore(cfg.visualization_redis_url, cfg.manim_job_ttl_seconds)
                store.inline = False
            except ImportError:
                store = None
        if store is None:
            store = MemoryJobStore()
            store.inline = True
        container.viz_store = store
        container.viz_media = MediaStore(cfg.manim_media_dir)
        container.viz_tasks = set()
    return container.viz_store, container.viz_media


def _inline_repair(container):
    async def repair(kind: str, spec_json: str, error: str, attempt: int) -> str | None:
        cfg = container.config
        out = await repair_block(container.generation_router, kind, spec_json, error, attempt,
                                 max_manim=cfg.manim_max_spec_chars)
        return out.content if out.ok else None
    return repair


def _start_inline(container, job_id: str) -> None:
    cfg = container.config
    store, media = _services(container)

    async def go() -> None:
        try:
            await run_job(store, media, job_id, repair=_inline_repair(container), python=cfg.manim_python,
                          quality=cfg.manim_quality, timeout_seconds=cfg.manim_render_timeout_seconds,
                          max_output_bytes=cfg.manim_max_output_mb * 1024 * 1024, max_spec_chars=cfg.manim_max_spec_chars,
                          repair_enabled=cfg.visualization_repair_enabled)
        except Exception as exc:                       # a worker bug must end the job, not strand it in `rendering`
            logger.error("visualization job crashed", job_id=job_id, error=type(exc).__name__)
            await store.update(job_id, JobState.FAILED, failure=FailureCategory.RENDER_ERROR.value)

    sem = getattr(container, "viz_sem", None)
    if sem is None:
        sem = container.viz_sem = asyncio.Semaphore(cfg.manim_max_concurrent_jobs)

    async def guarded() -> None:
        async with sem:
            await go()

    task = asyncio.create_task(guarded())
    container.viz_tasks.add(task)
    task.add_done_callback(container.viz_tasks.discard)


# ------------------------------------------------------------------------------------------------ public API

@router.get("/config")
async def viz_config(container: ContainerDep) -> dict:
    cfg = container.config
    on = cfg.visualization_enabled
    return {
        "schema_version": "1.0",
        "enabled": on,
        "mermaid": on and cfg.mermaid_enabled,
        "manim": on and cfg.manim_enabled,
        "max_attempts": cfg.visualization_max_attempts,
        "repair": cfg.visualization_repair_enabled,
        "max_blocks": cfg.visualization_max_blocks_per_response,
        "templates": [t["name"] for t in template_catalog()] if cfg.manim_enabled else [],
    }


class JobRequest(BaseModel):
    spec: dict


@router.post("/manim/jobs", status_code=202)
async def create_job(body: JobRequest, container: ContainerDep, x_user_id: str | None = Header(default=None)) -> dict:
    cfg = container.config
    owner = _user(x_user_id)
    if not (cfg.visualization_enabled and cfg.manim_enabled):
        raise HTTPException(status_code=403, detail={"failure": "disabled", "message": user_message("disabled")})
    try:
        spec = validate_spec(body.spec, cfg.manim_max_spec_chars)
    except SpecError as exc:
        cat = FailureCategory.UNSUPPORTED_TEMPLATE if exc.unsupported else FailureCategory.INVALID_SPEC
        raise HTTPException(status_code=422, detail={"failure": cat.value, "message": user_message(cat)}) from None
    store, _ = _services(container)
    jid = job_id_for(owner, spec.canonical_json())
    existing = await store.get(jid)
    if existing is None or existing.state not in (JobState.READY, *ACTIVE):
        if await store.active_count(owner) >= cfg.manim_max_active_jobs_per_user:
            raise HTTPException(status_code=429, detail={"failure": "budget_exhausted", "message": "Too many animations in progress."})
        if not await store.hourly_count(owner, cfg.visualization_jobs_per_user_per_hour):
            raise HTTPException(status_code=429, detail={"failure": "budget_exhausted", "message": user_message("budget_exhausted")})
    record = JobRecord(id=jid, owner=owner, spec=json.loads(spec.canonical_json()), max_attempts=cfg.visualization_max_attempts)
    record, created = await store.create_if_absent(record)
    if created:
        if getattr(store, "inline", False):
            _start_inline(container, jid)
        else:
            await store.enqueue(jid)
    return record.public()


async def _owned(container, job_id: str, x_user_id: str | None) -> JobRecord:
    owner = _user(x_user_id)
    store, _ = _services(container)
    rec = await store.get(job_id) if _JOB_ID.match(job_id) else None
    if rec is None or rec.owner != owner:
        raise HTTPException(status_code=404, detail="Not found")
    return rec


@router.get("/manim/jobs/{job_id}")
async def get_job(job_id: str, container: ContainerDep, x_user_id: str | None = Header(default=None)) -> dict:
    rec = await _owned(container, job_id, x_user_id)
    out = rec.public()
    if rec.failure:
        out["message"] = user_message(rec.failure)
    return out


@router.get("/manim/jobs/{job_id}/media")
async def get_media(job_id: str, container: ContainerDep, x_user_id: str | None = Header(default=None)):
    rec = await _owned(container, job_id, x_user_id)
    _, media = _services(container)
    path = media.path_for(job_id)
    if rec.state != JobState.READY or path is None or not path.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(path, media_type="video/mp4", headers={"Cache-Control": "private, max-age=3600",
                                                              "X-Content-Type-Options": "nosniff"})


@router.delete("/manim/jobs/{job_id}", status_code=202)
async def cancel_job(job_id: str, container: ContainerDep, x_user_id: str | None = Header(default=None)) -> dict:
    rec = await _owned(container, job_id, x_user_id)
    store, _ = _services(container)
    if rec.state in ACTIVE:
        rec = await store.update(job_id, JobState.CANCELLED, failure=FailureCategory.CANCELLED.value) or rec
    return rec.public()


class RepairRequest(BaseModel):
    kind: Literal["mermaid", "manim"]
    content: str = Field(max_length=8000)          # the failed mermaid source or manim spec JSON
    error: str = Field(default="syntax", max_length=60)
    attempt: int = Field(ge=2, le=5)


@router.post("/repair")
async def repair(body: RepairRequest, container: ContainerDep, x_user_id: str | None = Header(default=None)) -> dict:
    """Repair ONE failed block (used by the browser for Mermaid). Attempt 1 was the original, so repairs start at 2."""
    cfg = container.config
    owner = _user(x_user_id)
    if not (cfg.visualization_enabled and cfg.visualization_repair_enabled):
        return {"ok": False, "failure": "disabled"}
    if body.attempt > cfg.visualization_max_attempts:
        return {"ok": False, "failure": "budget_exhausted"}
    if body.kind == "mermaid" and not cfg.mermaid_enabled or body.kind == "manim" and not cfg.manim_enabled:
        return {"ok": False, "failure": "disabled"}
    store, _ = _services(container)
    if not await store.hourly_count(f"repair:{owner}", cfg.visualization_jobs_per_user_per_hour * 3):
        return {"ok": False, "failure": "budget_exhausted"}
    out = await repair_block(container.generation_router, body.kind, body.content, body.error, body.attempt,
                             max_mermaid=cfg.mermaid_max_source_chars, max_manim=cfg.manim_max_spec_chars)
    viz_metrics.VISUALIZATION_REPAIR_TOTAL.labels(kind=body.kind, result="ok" if out.ok else "failed").inc()
    return {"ok": out.ok, "content": out.content if out.ok else None, "failure": None if out.ok else (out.category or "render_error")}


# ------------------------------------------------------------------------------------------------ worker-only API

class InternalRepair(BaseModel):
    kind: Literal["manim"] = "manim"
    content: str = Field(max_length=8000)
    error: str = Field(default="render_error", max_length=60)
    attempt: int = Field(ge=2, le=5)


@internal_router.post("/repair")
async def internal_repair(body: InternalRepair, container: ContainerDep, x_internal_token: str | None = Header(default=None)) -> dict:
    cfg = container.config
    secret = cfg.visualization_internal_token.get_secret_value()
    if not secret or not hmac.compare_digest(secret, x_internal_token or ""):
        raise HTTPException(status_code=401, detail="Unauthorized")
    out = await repair_block(container.generation_router, body.kind, body.content, body.error, body.attempt,
                             max_manim=cfg.manim_max_spec_chars)
    return {"ok": out.ok, "content": out.content if out.ok else None}


__all__ = ["router", "internal_router", "validate_candidate"]
