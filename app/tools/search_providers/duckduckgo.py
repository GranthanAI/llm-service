"""
DuckDuckGo Search Provider.
Uses DuckDuckGo Instant Answer and HTML search APIs.
"""

from typing import Any
from urllib.parse import quote_plus

import httpx

from app.tools.search_providers.base import BaseSearchProvider


class DuckDuckGoProvider(BaseSearchProvider):
    """Free DuckDuckGo Search Provider."""

    name: str = "duckduckgo"

    async def search(
        self,
        query: str,
        max_results: int,
        client: httpx.AsyncClient,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        try:
            # 1. Try Instant Answer API
            resp = await client.get(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
                headers={"User-Agent": "GraphGPT-LLMService/2.0"},
            )
            if resp.status_code == 200:
                data = resp.json()
                abstract = data.get("AbstractText")
                abstract_url = data.get("AbstractURL")
                heading = data.get("Heading")
                if abstract and abstract_url:
                    results.append(
                        {
                            "title": heading or query,
                            "url": abstract_url,
                            "snippet": abstract,
                            "source": self.name,
                        }
                    )

                # Related topics
                related = data.get("RelatedTopics", [])
                for topic in related:
                    if len(results) >= max_results:
                        break
                    text = topic.get("Text")
                    first_url = topic.get("FirstURL")
                    if text and first_url:
                        results.append(
                            {
                                "title": text.split(" - ")[0] if " - " in text else text[:50],
                                "url": first_url,
                                "snippet": text,
                                "source": self.name,
                            }
                        )

            if not results:
                # Direct DuckDuckGo link fallback
                results.append(
                    {
                        "title": f"DuckDuckGo search for '{query}'",
                        "url": f"https://duckduckgo.com/?q={quote_plus(query)}",
                        "snippet": f"Web results for '{query}' from DuckDuckGo.",
                        "source": self.name,
                    }
                )

        except Exception:
            pass

        return results[:max_results]
