"""
Unit tests for Multi-Source WebSearchTool and 7 free search providers.
Implements Phase 10 deliverables verification.
"""

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from app.models.tool import ToolParams
from app.tools.search_providers import (
    ArxivProvider,
    DuckDuckGoProvider,
    OpenAlexProvider,
    SearXNGProvider,
    StackExchangeProvider,
    WikidataProvider,
    WikipediaProvider,
)
from app.tools.web_search import WebSearchTool


@pytest.mark.asyncio
async def test_duckduckgo_provider():
    """Verify DuckDuckGoProvider extracts instant answer and related topics."""
    provider = DuckDuckGoProvider()
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "AbstractText": "Python is a programming language.",
        "AbstractURL": "https://en.wikipedia.org/wiki/Python_(programming_language)",
        "Heading": "Python",
        "RelatedTopics": [
            {"Text": "Python 3 - Latest major release", "FirstURL": "https://python.org/3"}
        ],
    }
    mock_client.get.return_value = mock_resp

    results = await provider.search("python", max_results=5, client=mock_client)
    assert len(results) >= 1
    assert results[0]["source"] == "duckduckgo"
    assert "Python" in results[0]["title"]


@pytest.mark.asyncio
async def test_wikipedia_provider():
    """Verify WikipediaProvider parses MediaWiki search JSON and strips HTML."""
    provider = WikipediaProvider()
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "query": {
            "search": [
                {
                    "title": "Quantum Mechanics",
                    "snippet": "<span>Quantum mechanics</span> is a fundamental theory in physics.",
                }
            ]
        }
    }
    mock_client.get.return_value = mock_resp

    results = await provider.search("quantum", max_results=5, client=mock_client)
    assert len(results) == 1
    assert results[0]["source"] == "wikipedia"
    assert "Quantum Mechanics" in results[0]["title"]
    assert "<span>" not in results[0]["snippet"]  # HTML stripped


@pytest.mark.asyncio
async def test_searxng_provider():
    """Verify SearXNGProvider parses metasearch JSON."""
    provider = SearXNGProvider(base_url="https://searx.be")
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "results": [{"title": "SearX hit", "url": "https://example.com", "content": "Meta snippet"}]
    }
    mock_client.get.return_value = mock_resp

    results = await provider.search("test", max_results=5, client=mock_client)
    assert len(results) == 1
    assert results[0]["source"] == "searxng"


@pytest.mark.asyncio
async def test_wikidata_provider():
    """Verify WikidataProvider extracts entity labels and concepts."""
    provider = WikidataProvider()
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "search": [
            {
                "id": "Q28865",
                "label": "Python",
                "description": "high-level programming language",
                "concepturi": "http://www.wikidata.org/entity/Q28865",
            }
        ]
    }
    mock_client.get.return_value = mock_resp

    results = await provider.search("Python", max_results=5, client=mock_client)
    assert len(results) == 1
    assert results[0]["source"] == "wikidata"
    assert "Q28865" in results[0]["title"]


@pytest.mark.asyncio
async def test_arxiv_provider():
    """Verify ArxivProvider parses Atom XML feed."""
    provider = ArxivProvider()
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <id>http://arxiv.org/abs/2301.00001</id>
        <title>Attention Is All You Need</title>
        <summary>The dominant sequence transduction models are based on complex recurrent neural networks.</summary>
      </entry>
    </feed>
    """
    mock_client.get.return_value = mock_resp

    results = await provider.search("transformers", max_results=5, client=mock_client)
    assert len(results) == 1
    assert results[0]["source"] == "arxiv"
    assert "Attention Is All You Need" in results[0]["title"]


@pytest.mark.asyncio
async def test_openalex_provider():
    """Verify OpenAlexProvider parses open scholarly works."""
    provider = OpenAlexProvider()
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "results": [
            {
                "id": "https://openalex.org/W12345",
                "display_name": "Deep Learning Survey",
                "publication_year": 2024,
                "cited_by_count": 150,
                "primary_location": {"source": {"display_name": "Nature"}},
            }
        ]
    }
    mock_client.get.return_value = mock_resp

    results = await provider.search("deep learning", max_results=5, client=mock_client)
    assert len(results) == 1
    assert results[0]["source"] == "openalex"
    assert "Deep Learning Survey" in results[0]["title"]


@pytest.mark.asyncio
async def test_stackexchange_provider():
    """Verify StackExchangeProvider parses StackOverflow questions."""
    provider = StackExchangeProvider()
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "items": [
            {
                "title": "How to do async in Python?",
                "link": "https://stackoverflow.com/questions/12345",
                "score": 42,
                "is_answered": True,
                "tags": ["python", "asyncio"],
            }
        ]
    }
    mock_client.get.return_value = mock_resp

    results = await provider.search("python async", max_results=5, client=mock_client)
    assert len(results) == 1
    assert results[0]["source"] == "stackexchange"
    assert "StackOverflow" in results[0]["title"]


@pytest.mark.asyncio
async def test_web_search_tool_multi_provider_aggregation_and_partial_failure():
    """Verify WebSearchTool aggregates across multiple providers and handles partial failures gracefully."""
    mock_p1 = AsyncMock()
    mock_p1.name = "provider1"
    mock_p1.search.return_value = [
        {"title": "P1 Doc", "url": "https://example.com/p1", "snippet": "Snippet 1", "source": "p1"}
    ]

    mock_p2 = AsyncMock()
    mock_p2.name = "provider2"
    mock_p2.search.return_value = [
        {
            "title": "P2 Doc",
            "url": "https://wikipedia.org/wiki/p2",
            "snippet": "Snippet 2",
            "source": "p2",
        }
    ]

    mock_failing = AsyncMock()
    mock_failing.name = "failing_provider"
    mock_failing.search.side_effect = TimeoutError("Simulated provider timeout")

    tool = WebSearchTool(providers=[mock_p1, mock_p2, mock_failing])

    params = ToolParams(tool_name="web_search", params={"query": "fastapi multi source"})
    res = await tool.execute(params)

    assert res.success is True
    assert res.data["type"] == "web_search"
    results = res.data["results"]
    assert len(results) == 2  # Collected from p1 and p2 despite failing_provider
    assert any(r["title"] == "P1 Doc" for r in results)
    assert any(r["title"] == "P2 Doc" for r in results)
