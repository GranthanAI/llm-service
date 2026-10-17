"""
Production Multi-Source Web Search Tool.
Orchestrates 7 free public search engines in parallel:
- DuckDuckGo
- Wikipedia
- SearXNG
- Wikidata
- ArXiv
- OpenAlex
- Stack Exchange
"""

import asyncio
from typing import Any
from urllib.parse import urlparse

import httpx
import structlog

from app.config.logging import get_logger
from app.models.tool import (
    ToolParams,
    ToolResult,
    ToolSchema,
    ValidationResult,
)
from app.tools.base import BaseTool
from app.tools.normalizer import ToolNormalizer
from app.tools.search_providers import (
    ArxivProvider,
    BaseSearchProvider,
    DuckDuckGoProvider,
    OpenAlexProvider,
    SearXNGProvider,
    StackExchangeProvider,
    WikidataProvider,
    WikipediaProvider,
)
from app.tools.validator import ToolValidator
from app.utils.cache import TTLCache


def _result_of(task: "asyncio.Future[Any]") -> Any:
    """A finished provider task's list, or the exception it raised (never raises itself)."""
    try:
        return task.result()
    except BaseException as exc:  # includes CancelledError of a provider that cancelled itself
        return exc


class WebSearchTool(BaseTool):
    """
    Multi-source open search aggregator.
    Executes searches across 7 free providers in parallel with zero API key dependencies.
    """

    name: str = "web_search"
    description: str = "Search the web, Wikipedia, Wikidata, arXiv scholarly papers, OpenAlex, and StackOverflow in parallel."
    version: str = "2.0.0"
    timeout_ms: int = 5000
    is_async: bool = True

    def __init__(
        self,
        providers: list[BaseSearchProvider] | None = None,
        timeout_ms: int = 5000,
        http_client: httpx.AsyncClient | None = None,
        cache_ttl_seconds: int = 60,
        api_key: Any = None,
        logger: structlog.stdlib.BoundLogger | None = None,
    ):
        self.timeout_ms = timeout_ms
        # Providers still running at this point are abandoned. 60% of the tool timeout leaves the executor's
        # hard limit (timeout_ms) unreachable while sources are answering.
        self.gather_deadline_s = max(1.0, timeout_ms / 1000.0 * 0.6)
        self.api_key = api_key
        self._http_client: httpx.AsyncClient | None = http_client
        self._cache = TTLCache(max_size=1000, default_ttl_seconds=cache_ttl_seconds)
        self.logger = logger or get_logger("web_search_tool")

        # Initialize all 7 providers
        self.providers: list[BaseSearchProvider] = providers or [
            DuckDuckGoProvider(),
            WikipediaProvider(),
            SearXNGProvider(),
            WikidataProvider(),
            ArxivProvider(),
            OpenAlexProvider(),
            StackExchangeProvider(),
        ]

    def get_schema(self) -> ToolSchema:
        """Return self-describing ToolSchema."""
        return ToolSchema(
            name=self.name,
            description=self.description,
            version=self.version,
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to execute across all 7 sources.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of search results per provider (default 5).",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
            examples=[
                {
                    "query": "FastAPI async architecture",
                    "max_results": 3,
                }
            ],
        )

    def validate_params(self, params: ToolParams) -> ValidationResult:
        """Validate that 'query' parameter is present and non-empty."""
        res = ToolValidator.validate_required_fields(params, ["query"])
        if not res.valid:
            return res
        query_val = params.params.get("query")
        if not isinstance(query_val, str) or not query_val.strip():
            return ValidationResult(valid=False, error="Search query cannot be empty")
        return ValidationResult(valid=True)

    async def execute(self, params: ToolParams) -> ToolResult:
        """Execute multi-provider search in parallel with caching and deduplication."""
        query: str = str(params.params.get("query", "")).strip()
        max_results: int = int(params.params.get("max_results", 5))

        cache_key = f"websearch:{query.lower()}:{max_results}"

        # 1. Check TTL Cache
        cached_data = self._cache.get(cache_key)
        if cached_data is not None:
            self.logger.debug("Web search cache hit", query=query)
            return ToolNormalizer.normalize_web_search(
                query=query,
                results=cached_data,
                metadata={"cache_hit": True, "provider_count": len(self.providers)},
            )

        # 2. Gather searches across all 7 providers in parallel
        raw_results = await self._gather_provider_searches(query, max_results)

        # 3. Deduplicate results across domains and URLs
        deduped_results = self._deduplicate_by_domain(raw_results)

        # 4. Populate TTL Cache
        self._cache.set(cache_key, deduped_results)

        return ToolNormalizer.normalize_web_search(
            query=query,
            results=deduped_results,
            metadata={
                "cache_hit": False,
                "provider_count": len(self.providers),
                "total_found": len(raw_results),
            },
        )

    async def _gather_provider_searches(self, query: str, max_results: int) -> list[dict[str, Any]]:
        """Query all search providers concurrently."""
        # One shared client: building a new one per search costs ~0.7 s of SSL setup on Windows, which counts
        # against the tool's hard timeout and defeats connection reuse.
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=self.timeout_ms / 1000.0, follow_redirects=True)
        client = self._http_client

        tasks = [
            asyncio.ensure_future(provider.search(query=query, max_results=max_results, client=client))
            for provider in self.providers
        ]
        # Whatever has answered by the deadline is used; slower providers are cancelled, not awaited.
        # The deadline sits BELOW the executor's hard tool timeout, so one slow source can never turn a
        # search that other sources already answered into a failed (and, for required tools, fatal) one.
        _done, pending = await asyncio.wait(tasks, timeout=self.gather_deadline_s)
        for t in pending:
            t.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        responses: list[Any] = [
            (TimeoutError("provider did not answer before the search deadline") if t in pending else _result_of(t))
            for t in tasks
        ]

        aggregated: list[dict[str, Any]] = []
        counts: dict[str, int] = {}
        failed: list[str] = []
        for provider, resp in zip(self.providers, responses, strict=False):
            if isinstance(resp, list):
                self.logger.debug(
                    "Search provider responded",
                    provider=provider.name,
                    count=len(resp),
                )
                aggregated.extend(resp)
                counts[provider.name] = len(resp)
            elif isinstance(resp, BaseException):
                failed.append(provider.name)
                self.logger.debug(
                    "Search provider failed or timed out",
                    provider=provider.name,
                    error=str(resp),
                )

        # One line per search: which sources contributed and which did not. A provider failing is
        # expected (rate limits, outages) and never fails the search while others respond.
        (self.logger.warning if failed or not aggregated else self.logger.info)(
            "web_search_providers",
            results=len(aggregated),
            per_provider=counts,
            failed=failed,
            empty=sorted(n for n, c in counts.items() if c == 0),
        )
        return aggregated

    def _deduplicate_by_domain(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Keep only the highest-ranked result per domain name (LLD2 Section 16.1)."""
        seen_domains: set[str] = set()
        deduped: list[dict[str, Any]] = []

        for r in results:
            url = r.get("url", "").strip()
            if not url:
                continue

            try:
                domain = urlparse(url).netloc.lower()
            except Exception:
                domain = url

            if domain and domain not in seen_domains:
                seen_domains.add(domain)
                deduped.append(r)
            elif not domain:
                deduped.append(r)

        return deduped
