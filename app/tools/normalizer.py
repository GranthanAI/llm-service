"""
Tool Result Normalization.
Implements LLD v2.0 Section 15.6.
"""

from typing import Any

from app.models.tool import ToolResult


class ToolNormalizer:
    """Normalizes heterogeneous tool outputs into standardized schemas."""

    @staticmethod
    def normalize_web_search(
        query: str,
        results: list[dict[str, Any]],
        latency_ms: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> ToolResult:
        """
        Normalize WebSearchTool outputs into canonical schema:
        {
          "type": "web_search",
          "results": [{"title": ..., "url": ..., "snippet": ...}],
          "query": "..."
        }
        """
        normalized_results = []
        for r in results:
            normalized_results.append(
                {
                    "title": str(r.get("title", "")),
                    "url": str(r.get("url", "")),
                    "snippet": str(r.get("snippet", r.get("content", ""))),
                }
            )

        return ToolResult(
            tool_name="web_search",
            success=True,
            data={
                "type": "web_search",
                "results": normalized_results,
                "query": query,
            },
            latency_ms=latency_ms,
            metadata=metadata or {},
        )

    @staticmethod
    def normalize_error(
        tool_name: str,
        error_message: str,
        latency_ms: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> ToolResult:
        """Create standard error ToolResult."""
        return ToolResult(
            tool_name=tool_name,
            success=False,
            error=error_message,
            latency_ms=latency_ms,
            metadata=metadata or {},
        )
