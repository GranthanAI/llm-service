"""
Base Search Provider Protocol.
"""

from abc import ABC, abstractmethod
from typing import Any

import httpx


class BaseSearchProvider(ABC):
    """Abstract base class for all free public search providers."""

    name: str

    @abstractmethod
    async def search(
        self,
        query: str,
        max_results: int,
        client: httpx.AsyncClient,
    ) -> list[dict[str, Any]]:
        """
        Execute search query against the provider.
        Returns list of dicts with keys: 'title', 'url', 'snippet', 'source'.
        """
        pass
