"""
Which model ids the PROVIDER still serves (so a retired id shows as unavailable instead of failing every request).

Two sources, both cached for an hour:
  * Google's free `models.list` with the configured key: an id that is not listed is retired.
  * what actually happened: a generation that answered 404 / "no longer available" marks that id retired, because Google
    can keep a retired id in the list while refusing it ("no longer available to new users").
Anything unknown (no key, list failed) fails OPEN: the model counts as available.
"""

import time
from collections.abc import Awaitable, Callable

import httpx

TTL_SECONDS = 3600.0
RETIRED_REASON = "retired by provider"


async def fetch_google_ids(api_key: str, timeout_s: float = 5.0) -> set[str]:
    ids: set[str] = set()
    token = ""
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        for _ in range(5):
            resp = await client.get(
                "https://generativelanguage.googleapis.com/v1beta/models",
                params={"pageSize": 200, **({"pageToken": token} if token else {})},
                headers={"x-goog-api-key": api_key},
            )
            resp.raise_for_status()
            body = resp.json()
            for m in body.get("models", []):
                if "generateContent" in (m.get("supportedGenerationMethods") or []):
                    ids.add(m["name"].split("/", 1)[-1])
            token = body.get("nextPageToken") or ""
            if not token:
                break
    return ids


class ProviderModels:
    def __init__(self, api_key: str = "", fetch: Callable[[str], Awaitable[set[str]]] = fetch_google_ids,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.api_key, self._fetch, self._now = api_key, fetch, clock
        self._listed: tuple[float, set[str] | None] | None = None
        self._retired: dict[str, float] = {}

    async def google_ids(self) -> set[str] | None:
        """Ids Google lists for generation with this key; None when they cannot be listed (fail open)."""
        if not self.api_key:
            return None
        now = self._now()
        if self._listed is None or now - self._listed[0] > TTL_SECONDS:
            try:
                ids: set[str] | None = await self._fetch(self.api_key)
            except Exception:  # noqa: BLE001
                ids = None
            self._listed = (now, ids)
        return self._listed[1]

    def mark_retired(self, model_id: str) -> None:
        self._retired[model_id] = self._now()

    def is_retired(self, model_id: str) -> bool:
        at = self._retired.get(model_id)
        return at is not None and self._now() - at < TTL_SECONDS


provider_models = ProviderModels()
