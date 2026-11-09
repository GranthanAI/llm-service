"""
Catalog check: billing's model catalog (GET {BILLING_SERVICE_URL}/billing/v1/models) against this service's registry.

At startup and every 10 minutes: log and gauge (llm_catalog_model_missing{model}) every catalog model this service cannot
serve (disabled, no credentials, no adapter) and every registry model billing does not list (or lists in another
group). Never blocks startup; billing down is logged and retried at the next cycle.
"""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
from prometheus_client import Gauge

from app.config.logging import get_logger

logger = get_logger("catalog_check")

CATALOG_MODEL_MISSING = Gauge(
    "llm_catalog_model_missing",
    "1 when a billing catalog model cannot be served here (or a registry model is not in the catalog), else 0",
    ["model", "reason"],
)
REASONS = ("disabled", "no_credentials", "no_adapter", "not_in_registry", "not_in_catalog", "group_mismatch")

CHECK_INTERVAL_S = 600.0


def compare(catalog: list[dict[str, Any]], registry: Any, has_credentials: Callable[[str], bool],
            has_adapter: Callable[[Any], bool]) -> dict[str, list[str]]:
    """{reason: [model ids]} for everything that does not line up (empty lists omitted)."""
    problems: dict[str, list[str]] = {}

    def add(reason: str, model: str) -> None:
        problems.setdefault(reason, []).append(model)

    by_id = {m["id"]: m for m in catalog}
    for model_id, entry in by_id.items():
        spec = registry.models.get(model_id)
        if spec is None:
            add("not_in_registry", model_id)
        elif spec.group != entry.get("group"):
            add("group_mismatch", model_id)
        elif not spec.enabled:
            add("disabled", model_id)
        elif not has_credentials(spec.provider):
            add("no_credentials", model_id)
        elif not has_adapter(spec):
            add("no_adapter", model_id)
    for model_id, spec in registry.models.items():
        if model_id not in by_id and spec.enabled:
            add("not_in_catalog", model_id)
    return problems


def publish(problems: dict[str, list[str]], previous: set[tuple[str, str]]) -> set[tuple[str, str]]:
    """Set the gauge for the current problems and clear the ones that went away."""
    current = {(m, reason) for reason, models in problems.items() for m in models}
    for model, reason in previous - current:
        CATALOG_MODEL_MISSING.labels(model=model, reason=reason).set(0)
    for model, reason in current:
        CATALOG_MODEL_MISSING.labels(model=model, reason=reason).set(1)
    return current


async def fetch_catalog(base_url: str, timeout_s: float = 5.0) -> list[dict[str, Any]]:
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        resp = await client.get(f"{base_url.rstrip('/')}/billing/v1/models")
        resp.raise_for_status()
        return list(resp.json()["models"])


class CatalogChecker:
    def __init__(self, base_url: str, registry: Any, has_credentials: Callable[[str], bool],
                 has_adapter: Callable[[Any], bool],
                 fetch: Callable[[str], Awaitable[list[dict[str, Any]]]] = fetch_catalog):
        self.base_url, self.registry = base_url, registry
        self.has_credentials, self.has_adapter, self._fetch = has_credentials, has_adapter, fetch
        self._previous: set[tuple[str, str]] = set()
        self._task: asyncio.Task | None = None

    async def check_once(self) -> dict[str, list[str]] | None:
        """One cycle. None when billing could not be reached (logged; the next cycle retries)."""
        try:
            catalog = await self._fetch(self.base_url)
        except Exception as exc:
            logger.warning("catalog_check_billing_unreachable", error=type(exc).__name__)
            return None
        problems = compare(catalog, self.registry, self.has_credentials, self.has_adapter)
        self._previous = publish(problems, self._previous)
        if problems:
            logger.warning("catalog_check_mismatch", **{k: v for k, v in problems.items()})
        else:
            logger.info("catalog_check_ok", models=len(catalog))
        return problems

    async def _loop(self) -> None:
        while True:
            try:
                await self.check_once()
            except Exception as exc:                       # never let the checker die
                logger.error("catalog_check_failed", error=type(exc).__name__)
            await asyncio.sleep(CHECK_INTERVAL_S)

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
