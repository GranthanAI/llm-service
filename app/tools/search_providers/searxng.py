"""
SearXNG Metasearch Engine Provider.
Queries public or self-hosted SearXNG instances.
"""

from typing import Any

import httpx

from app.tools.search_providers.base import BaseSearchProvider


class SearXNGProvider(BaseSearchProvider):
    """SearXNG metasearch provider."""

    name: str = "searxng"

    def __init__(self, base_url: str = "https://searx.be"):
        self.base_url = base_url.rstrip("/")

    async def search(
        self,
        query: str,
        max_results: int,
        client: httpx.AsyncClient,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        try:
            resp = await client.get(
                f"{self.base_url}/search",
                params={"q": query, "format": "json"},
                headers={"User-Agent": "GraphGPT-LLMService/2.0"},
            )
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get("results", [])[:max_results]:
                    results.append(
                        {
                            "title": item.get("title", ""),
                            "url": item.get("url", ""),
                            "snippet": item.get("content", item.get("snippet", "")),
                            "source": self.name,
                        }
                    )
        except Exception:
            pass

        return results[:max_results]
