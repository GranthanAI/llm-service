"""
Wikipedia Search Provider.
Uses MediaWiki API for English Wikipedia.
"""

import re
from typing import Any
from urllib.parse import quote

import httpx

from app.tools.search_providers.base import BaseSearchProvider


class WikipediaProvider(BaseSearchProvider):
    """Wikipedia API search provider."""

    name: str = "wikipedia"

    async def search(
        self,
        query: str,
        max_results: int,
        client: httpx.AsyncClient,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        try:
            resp = await client.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "utf8": 1,
                    "format": "json",
                    "srlimit": max_results,
                },
                headers={"User-Agent": "GraphGPT-LLMService/2.0 (contact@graphgpt.ai)"},
            )
            if resp.status_code == 200:
                data = resp.json()
                items = data.get("query", {}).get("search", [])
                for item in items:
                    title = item.get("title", "")
                    raw_snippet = item.get("snippet", "")
                    clean_snippet = re.sub(r"<[^>]+>", "", raw_snippet)
                    page_url = f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"
                    results.append(
                        {
                            "title": f"{title} — Wikipedia",
                            "url": page_url,
                            "snippet": clean_snippet,
                            "source": self.name,
                        }
                    )
        except Exception:
            pass

        return results[:max_results]
