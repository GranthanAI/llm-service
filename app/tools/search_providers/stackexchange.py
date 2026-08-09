"""
Stack Exchange / Stack Overflow Search Provider.
Queries Stack Exchange public API for technical questions, answers, and discussions.
"""

from typing import Any

import httpx

from app.tools.search_providers.base import BaseSearchProvider


class StackExchangeProvider(BaseSearchProvider):
    """Stack Exchange and Stack Overflow search provider."""

    name: str = "stackexchange"

    async def search(
        self,
        query: str,
        max_results: int,
        client: httpx.AsyncClient,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        try:
            resp = await client.get(
                "https://api.stackexchange.com/2.3/search/advanced",
                params={
                    "order": "desc",
                    "sort": "relevance",
                    "q": query,
                    "site": "stackoverflow",
                    "pagesize": max_results,
                },
                headers={"User-Agent": "GraphGPT-LLMService/2.0"},
            )
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get("items", []):
                    title = item.get("title", "")
                    link = item.get("link", "")
                    score = item.get("score", 0)
                    is_answered = item.get("is_answered", False)
                    tags = ", ".join(item.get("tags", []))
                    snippet = f"Tags: [{tags}]. Score: {score}. Answered: {is_answered}."

                    if title and link:
                        results.append(
                            {
                                "title": f"[StackOverflow] {title}",
                                "url": link,
                                "snippet": snippet,
                                "source": self.name,
                            }
                        )
        except Exception:
            pass

        return results[:max_results]
