"""
Search Providers Public Exports.

BILLING: every provider here is free today, so web search is not metered. A paid provider must emit usage.metered
(billing integration plan 4.4.5) before it is added to this registry. (`tavily_api_key` in settings is unused.)
"""

from app.tools.search_providers.arxiv import ArxivProvider
from app.tools.search_providers.base import BaseSearchProvider
from app.tools.search_providers.duckduckgo import DuckDuckGoProvider
from app.tools.search_providers.openalex import OpenAlexProvider
from app.tools.search_providers.searxng import SearXNGProvider
from app.tools.search_providers.stackexchange import StackExchangeProvider
from app.tools.search_providers.wikidata import WikidataProvider
from app.tools.search_providers.wikipedia import WikipediaProvider

__all__ = [
    "BaseSearchProvider",
    "DuckDuckGoProvider",
    "WikipediaProvider",
    "SearXNGProvider",
    "WikidataProvider",
    "ArxivProvider",
    "OpenAlexProvider",
    "StackExchangeProvider",
]
