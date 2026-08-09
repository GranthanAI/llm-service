"""
Wikidata Entity Search Provider.
Queries Wikidata API for structured knowledge and entities.
"""

from typing import Any

import httpx

from app.tools.search_providers.base import BaseSearchProvider


class WikidataProvider(BaseSearchProvider):
    """Wikidata entity search provider."""

    name: str = "wikidata"

    async def search(
        self,
        query: str,
        max_results: int,
        client: httpx.AsyncClient,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        try:
            resp = await client.get(
                "https://www.wikidata.org/w/api.php",
                params={
                    "action": "wbsearchentities",
                    "search": query,
                    "language": "en",
                    "format": "json",
                    "limit": max_results,
                },
                headers={"User-Agent": "GraphGPT-LLMService/2.0 (contact@graphgpt.ai)"},
            )
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get("search", []):
                    entity_id = item.get("id", "")
                    label = item.get("label", "")
                    desc = item.get("description", "No description available.")
                    concept_url = (
                        item.get("concepturi") or f"https://www.wikidata.org/wiki/{entity_id}"
                    )
                    results.append(
                        {
                            "title": f"{label} ({entity_id}) — Wikidata",
                            "url": concept_url,
                            "snippet": desc,
                            "source": self.name,
                        }
                    )
        except Exception:
            pass

        return results[:max_results]
