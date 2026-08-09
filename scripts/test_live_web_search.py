"""
Live verification script for all 7 free web search providers.
Queries live public endpoints to verify real answers from:
1. DuckDuckGo
2. Wikipedia
3. SearXNG
4. Wikidata
5. ArXiv
6. OpenAlex
7. Stack Exchange
"""

import asyncio

import httpx

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


async def test_individual_providers():
    providers = [
        ("DuckDuckGo", DuckDuckGoProvider()),
        ("Wikipedia", WikipediaProvider()),
        ("SearXNG", SearXNGProvider()),
        ("Wikidata", WikidataProvider()),
        ("ArXiv", ArxivProvider()),
        ("OpenAlex", OpenAlexProvider()),
        ("StackExchange", StackExchangeProvider()),
    ]

    query = "FastAPI async python"
    print("\n=======================================================")
    print(f"Testing All 7 Free Search Providers Live for: '{query}'")
    print("=======================================================\n")

    async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
        for name, provider in providers:
            try:
                results = await provider.search(query=query, max_results=2, client=client)
                print(f"[{name}] -> Returned {len(results)} results:")
                for idx, r in enumerate(results, 1):
                    print(f"   {idx}. {r.get('title')}")
                    print(f"      URL: {r.get('url')}")
                    snippet = r.get("snippet", "")
                    print(f"      Snippet: {snippet[:120]}...\n")
            except Exception as e:
                print(f"[{name}] -> Error: {e}\n")


async def test_full_web_search_tool():
    print("\n=======================================================")
    print("Testing Full Parallel WebSearchTool Aggregation")
    print("=======================================================\n")
    from app.models.tool import ToolParams

    tool = WebSearchTool(timeout_ms=10000)
    params = ToolParams(
        tool_name="web_search", params={"query": "Quantum computing algorithms", "max_results": 2}
    )
    result = await tool.execute(params)

    print(f"Success: {result.success}")
    print(f"Metadata: {result.metadata}")
    print(f"Total Normalized Results Aggregated: {len(result.data.get('results', []))}\n")
    for idx, item in enumerate(result.data.get("results", [])[:8], 1):
        print(f"[{idx}] {item.get('title')}")
        print(f"    URL: {item.get('url')}")
        print(f"    Snippet: {item.get('snippet', '')[:100]}...\n")


if __name__ == "__main__":
    asyncio.run(test_individual_providers())
    asyncio.run(test_full_web_search_tool())
