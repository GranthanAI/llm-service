"""
Runs one Manim job through the state machine with the shared attempt policy.

Attempt 1 renders the original spec. If it fails with a repairable category, attempts 2 and 3 each ask `repair` for a corrected
spec (only this block) and render again; transient infrastructure failures are retried without calling the model; permanent
ones stop immediately. After the last attempt the job is `failed` with a content-free category and the client shows the text
fallback for that block only. `repair` is injected: in-process it calls the provider chain, in the worker process it calls the
llm-service internal endpoint (so the worker never holds provider secrets).
"""

import asyncio
import json
from collections.abc import Awaitable, Callable

from app.utils import metrics as viz_metrics
from app.visualization.jobs import ACTIVE, JobRecord, JobState, JobStore
from app.visualization.manim_templates import SpecError, validate_spec
from app.visualization.policy import AttemptTracker, FailureCategory
from app.visualization.renderer import MediaStore, RenderResult, render_spec

# (kind, spec_json, error_category, attempt) -> corrected canonical spec JSON, or None
RepairFn = Callable[[str, str, str, int], Awaitable[str | None]]
RenderFn = Callable[..., Awaitable[RenderResult]]


async def _run_job(
    store: JobStore,
    media: MediaStore,
    job_id: str,
    *,
    repair: RepairFn | None,
    render: RenderFn = render_spec,
    python: str = "",
    quality: str = "l",
    timeout_seconds: int = 90,
    max_output_bytes: int = 20 * 1024 * 1024,
    max_spec_chars: int = 4000,
    repair_enabled: bool = True,
) -> JobRecord | None:
    if not await store.claim(job_id):
        return await store.get(job_id)                       # another worker owns it (idempotent no-op)
    record = await store.get(job_id)
    if record is None or record.state not in ACTIVE:
        return record

    tracker = AttemptTracker(max_attempts=record.max_attempts)
    spec_json = json_dump(record.spec)
    last = FailureCategory.RENDER_ERROR
    need_repair = False

    async def cancelled_now() -> bool:
        cur = await store.get(job_id)
        return cur is None or cur.state in (JobState.CANCELLED, JobState.EXPIRED)

    while True:
        try:
            attempt = tracker.begin_attempt()
        except Exception:
            break
        rec = await store.update(job_id, JobState.VALIDATING, attempt=attempt)
        if rec is None:                                       # cancelled / expired underneath us
            return await store.get(job_id)

        if need_repair:
            await store.update(job_id, JobState.REPAIRING)
            fixed = await repair(record_kind(), spec_json, last.value, attempt) if (repair and repair_enabled) else None
            viz_metrics.VISUALIZATION_REPAIR_TOTAL.labels(kind="manim", result="ok" if fixed else "failed").inc()
            if not fixed:
                last = FailureCategory.INVALID_SPEC if repair else FailureCategory.WORKER_UNAVAILABLE
                tracker.record_failure(last)
                if tracker.next_step(last) == "give_up":
                    break
                continue
            spec_json = fixed
            await store.update(job_id, JobState.VALIDATING)
            need_repair = False

        try:
            validated = validate_spec(json.loads(spec_json), max_spec_chars)
        except SpecError as exc:
            last = FailureCategory.UNSUPPORTED_TEMPLATE if exc.unsupported else FailureCategory.INVALID_SPEC
        except (ValueError, RecursionError):
            last = FailureCategory.INVALID_SPEC
        else:
            await store.update(job_id, JobState.RENDERING)
            flag = {"c": False}

            async def watch() -> None:
                while not flag["c"]:
                    await asyncio.sleep(1.0)
                    if await cancelled_now():
                        flag["c"] = True

            watcher = asyncio.create_task(watch())
            try:
                result = await render(validated, python=python, quality=quality, timeout_seconds=timeout_seconds,
                                      max_output_bytes=max_output_bytes, cancelled=lambda: flag["c"])
            finally:
                watcher.cancel()
            if result.ok and result.path is not None:
                temp_files = [result.path, result.thumbnail]       # always removed below, whatever happens to `result`
                try:
                    key = media.save(job_id, result.path)
                except Exception:
                    result = RenderResult(False, FailureCategory.STORAGE_FAILURE)
                else:
                    if result.thumbnail is not None:
                        try:
                            media.save_thumbnail(job_id, result.thumbnail)
                        except Exception:
                            pass                              # a preview is optional; the video is what matters
                    done = await store.update(job_id, JobState.READY, media_key=key, failure="", history=tracker.history)
                    return done or await store.get(job_id)
                finally:
                    for tmp in temp_files:
                        if tmp is not None:
                            tmp.unlink(missing_ok=True)
            last = result.category or FailureCategory.RENDER_ERROR
            if last == FailureCategory.CANCELLED:
                await store.update(job_id, JobState.CANCELLED)
                return await store.get(job_id)

        tracker.record_failure(last)
        step = tracker.next_step(last)
        if step == "give_up":
            break
        need_repair = step == "repair"

    return await store.update(job_id, JobState.FAILED, failure=last.value, history=tracker.history) or await store.get(job_id)


def json_dump(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def record_kind() -> str:
    return "manim"


async def run_job(*args, **kwargs) -> JobRecord | None:
    """Run a job and record its final state in metrics (bounded labels: state + failure category)."""
    rec = await _run_job(*args, **kwargs)
    if rec is not None and rec.state not in ACTIVE:
        viz_metrics.MANIM_JOBS_TOTAL.labels(state=rec.state.value, failure=rec.failure or "none").inc()
    return rec
