"""
Arxiv Scholarly Paper Search Provider.
Queries Arxiv public API for academic and research papers.
"""

import xml.etree.ElementTree as ET
from typing import Any

import httpx

from app.tools.search_providers.base import BaseSearchProvider


class ArxivProvider(BaseSearchProvider):
    """Arxiv academic paper search provider."""

    name: str = "arxiv"

    async def search(
        self,
        query: str,
        max_results: int,
        client: httpx.AsyncClient,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        try:
            resp = await client.get(
                "http://export.arxiv.org/api/query",
                params={
                    "search_query": f"all:{query}",
                    "start": 0,
                    "max_results": max_results,
                },
                headers={"User-Agent": "GraphGPT-LLMService/2.0"},
            )
            if resp.status_code == 200:
                root = ET.fromstring(resp.text)
                # Atom namespace
                ns = {"atom": "http://www.w3.org/2005/Atom"}
                for entry in root.findall("atom:entry", ns):
                    title_elem = entry.find("atom:title", ns)
                    summary_elem = entry.find("atom:summary", ns)
                    id_elem = entry.find("atom:id", ns)

                    title = (
                        title_elem.text.strip().replace("\n", " ")
                        if title_elem is not None and title_elem.text
                        else ""
                    )
                    summary = (
                        summary_elem.text.strip().replace("\n", " ")
                        if summary_elem is not None and summary_elem.text
                        else ""
                    )
                    paper_url = id_elem.text.strip() if id_elem is not None and id_elem.text else ""

                    if title and paper_url:
                        results.append(
                            {
                                "title": f"[arXiv] {title}",
                                "url": paper_url,
                                "snippet": summary[:300] + ("..." if len(summary) > 300 else ""),
                                "source": self.name,
                            }
                        )
        except Exception:
            pass

        return results[:max_results]
