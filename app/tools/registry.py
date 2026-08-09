"""
Tool Registry.
Implements LLD v2.0 Section 15.2.
"""

from typing import Any

import structlog

from app.config.logging import get_logger
from app.exceptions.tool import UnknownToolError
from app.models.tool import ToolSchema
from app.tools.base import BaseTool


class ToolRegistry:
    """
    Registry holding registered BaseTool instances and their activation states.
    Injected via DI container; not a global singleton.
    """

    def __init__(self, logger: structlog.stdlib.BoundLogger | None = None):
        self._tools: dict[str, BaseTool] = {}
        self._enabled: set[str] = set()
        self.logger = logger or get_logger("tool_registry")

    def register(self, tool: BaseTool, enabled: bool = True) -> None:
        """Register a new tool instance in the registry."""
        self._tools[tool.name] = tool
        if enabled:
            self._enabled.add(tool.name)
        self.logger.info(
            "Tool registered",
            tool_name=tool.name,
            version=tool.version,
            enabled=enabled,
        )

    def get(self, name: str) -> BaseTool:
        """Retrieve tool by name if registered and enabled."""
        if name not in self._tools:
            raise UnknownToolError(
                f"Tool '{name}' is not registered in ToolRegistry", tool_name=name
            )
        if name not in self._enabled:
            raise UnknownToolError(f"Tool '{name}' is disabled in ToolRegistry", tool_name=name)
        return self._tools[name]

    def is_enabled(self, name: str) -> bool:
        """Check if tool is registered and enabled."""
        return name in self._tools and name in self._enabled

    def enable(self, name: str) -> None:
        """Enable a registered tool."""
        if name in self._tools:
            self._enabled.add(name)
            self.logger.info("Tool enabled", tool_name=name)

    def disable(self, name: str) -> None:
        """Disable a registered tool."""
        self._enabled.discard(name)
        self.logger.info("Tool disabled", tool_name=name)

    def list_tools(self, capability_filter: str | None = None) -> list[ToolSchema]:
        """Return list of ToolSchemas for all enabled tools."""
        schemas: list[ToolSchema] = []
        for name in sorted(self._enabled):
            tool = self._tools[name]
            schema = tool.get_schema()
            if capability_filter is None or capability_filter.lower() in schema.description.lower():
                schemas.append(schema)
        return schemas

    def get_all_schemas(self) -> list[dict[str, Any]]:
        """Return raw dict schemas of all enabled tools for Request Analyzer prompt injection."""
        return [
            tool.get_schema().model_dump()
            for name, tool in self._tools.items()
            if name in self._enabled
        ]
