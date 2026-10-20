"""
Isolated Manim worker process:  python -m app.visualization.worker

Consumes job ids from Redis and renders them with `runner.run_job`. It holds NO provider keys: repairs are requested from the
llm-service internal endpoint with a shared token. Run it with the Manim-capable interpreter in a container with no network
other than Redis + llm-service, a read-only root filesystem, a writable media volume and CPU/memory limits.
"""

import asyncio
import signal

import httpx
import structlog

from app.config.settings import get_settings
from app.visualization.jobs import RedisJobStore
from app.visualization.renderer import MediaStore
from app.visualization.runner import run_job

logger = structlog.get_logger("viz.worker")


def _http_repair(base_url: str, token: str):
    async def repair(kind: str, spec_json: str, error: str, attempt: int) -> str | None:
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.post(f"{base_url}/api/internal/visualizations/repair", headers={"X-Internal-Token": token},
                                      json={"kind": kind, "content": spec_json, "error": error, "attempt": attempt})
            data = r.json() if r.status_code == 200 else {}
            return data.get("content") if data.get("ok") else None
        except Exception:
            return None
    return repair


async def main() -> None:
    cfg = get_settings()
    store = RedisJobStore(cfg.visualization_redis_url, cfg.manim_job_ttl_seconds)
    media = MediaStore(cfg.manim_media_dir)
    repair = _http_repair(f"http://localhost:{getattr(cfg, 'port', 8003)}", cfg.visualization_internal_token.get_secret_value())
    stop = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            asyncio.get_running_loop().add_signal_handler(sig, stop.set)
        except NotImplementedError:       # Windows
            pass
    sem = asyncio.Semaphore(cfg.manim_max_concurrent_jobs)
    tasks: set[asyncio.Task] = set()
    logger.info("manim worker started", concurrency=cfg.manim_max_concurrent_jobs)

    async def handle(job_id: str) -> None:
        async with sem:
            try:
                await run_job(store, media, job_id, repair=repair, python=cfg.manim_python, quality=cfg.manim_quality,
                              timeout_seconds=cfg.manim_render_timeout_seconds,
                              max_output_bytes=cfg.manim_max_output_mb * 1024 * 1024, max_spec_chars=cfg.manim_max_spec_chars,
                              repair_enabled=cfg.visualization_repair_enabled)
            except Exception as exc:
                logger.error("job crashed", job_id=job_id, error=type(exc).__name__)

    last_purge = 0.0
    while not stop.is_set():
        await sem.acquire()
        sem.release()
        job_id = await store.dequeue(timeout=2)
        if job_id:
            task = asyncio.create_task(handle(job_id))
            tasks.add(task)
            task.add_done_callback(tasks.discard)
        loop_time = asyncio.get_running_loop().time()
        if loop_time - last_purge > 600:
            last_purge = loop_time
            media.purge_older_than(cfg.manim_media_retention_hours)
    await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == "__main__":
    asyncio.run(main())
