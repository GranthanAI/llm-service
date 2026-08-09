"""
OpenAlex Open Scholarly Works Search Provider.
Queries OpenAlex open catalog of scholarly articles, datasets, and theses.
"""

from typing import Any

import httpx

from app.tools.search_providers.base import BaseSearchProvider


class OpenAlexProvider(BaseSearchProvider):
    """OpenAlex scholarly works provider."""

    name: str = "openalex"

    async def search(
        self,
        query: str,
        max_results: int,
        client: httpx.AsyncClient,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        try:
            resp = await client.get(
                "https://api.openalex.org/works",
                params={"search": query, "per-page": max_results},
                headers={"User-Agent": "GraphGPT-LLMService/2.0 (mailto:contact@graphgpt.ai)"},
            )
            if resp.status_code == 200:
                data = resp.json()
                for work in data.get("results", []):
                    title = work.get("display_name", "")
                    work_url = work.get("doi") or work.get("id") or ""
                    pub_year = work.get("publication_year", "")
                    cited_by = work.get("cited_by_count", 0)

                    snippet = f"Published: {pub_year}. Citations: {cited_by}."
                    host_venue = work.get("primary_location", {}).get("source", {})
                    if host_venue and host_venue.get("display_name"):
                        snippet += f" Venue: {host_venue.get('display_name')}."

                    if title and work_url:
                        results.append(
                            {
                                "title": f"[Scholarly] {title}",
                                "url": work_url,
                                "snippet": snippet,
                                "source": self.name,
                            }
                        )
        except Exception:
            pass

        return results[:max_results]
