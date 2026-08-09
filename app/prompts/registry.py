"""
Prompt Template Registry.
Implements LLD v2.0 Section 17.1.
"""

from typing import Any

import structlog

from app.config.logging import get_logger
from app.exceptions.analysis import UnknownModeError
from app.prompts.loader import PromptLoader
from app.prompts.schemas import PromptTemplateConfig

logger = get_logger("prompt_registry")


class PromptRegistry:
    """
    Central repository for versioned prompt templates.
    Holds full mode-level templates, graph node fragments, system prompts, and reusable section partials.
    """

    def __init__(
        self,
        loader: PromptLoader | None = None,
        logger_instance: structlog.stdlib.BoundLogger | None = None,
    ):
        self.logger = logger_instance or logger
        self.mode_templates: dict[tuple[str, str], PromptTemplateConfig] = {}
        self.graph_node_templates: dict[tuple[str, str, str], PromptTemplateConfig] = {}
        self.system_templates: dict[tuple[str, str], PromptTemplateConfig] = {}
        self.partial_templates: dict[tuple[str, str], PromptTemplateConfig] = {}

        self.latest_mode_versions: dict[str, str] = {}
        self.latest_node_versions: dict[tuple[str, str], str] = {}
        self.latest_system_versions: dict[str, str] = {}
        self.latest_partial_versions: dict[str, str] = {}

        if loader is not None:
            self.load_from_loader(loader)

    def load_from_loader(self, loader: PromptLoader) -> None:
        """Loads all templates from the given loader and registers them."""
        configs = loader.load_all()
        for config in configs:
            if config.engine == "system" or (config.mode and config.mode.startswith("system")):
                self.register_system(config.name, config.version, config)
            elif config.engine == "template" or (
                config.mode and config.mode.startswith("template")
            ):
                self.register_template(config.name, config.version, config)
            elif config.mode:
                self.register_mode(config.mode, config.version, config)
            elif config.graph and config.node:
                self.register_graph_node(config.graph, config.node, config.version, config)

    def register_system(self, name: str, version: str, template: PromptTemplateConfig) -> None:
        """Registers a global system prompt template."""
        key = (name.lower(), version)
        self.system_templates[key] = template
        self.latest_system_versions[name.lower()] = version
        self.logger.debug("Registered system prompt template", name=name, version=version)

    def register_template(self, name: str, version: str, template: PromptTemplateConfig) -> None:
        """Registers a reusable section partial template."""
        key = (name.lower(), version)
        self.partial_templates[key] = template
        self.latest_partial_versions[name.lower()] = version
        self.logger.debug("Registered partial template", name=name, version=version)

    def get_system_template(self, name: str, version: str | None = None) -> PromptTemplateConfig:
        """Retrieves a system-level prompt template by name and optional version."""
        n = name.lower()
        v = version or self.latest_system_versions.get(n)
        if not v or (n, v) not in self.system_templates:
            if (n, "1.0") in self.system_templates:
                return self.system_templates[(n, "1.0")]
            raise KeyError(
                f"No system prompt template registered for '{name}' (version '{version}')"
            )
        return self.system_templates[(n, v)]

    def get_template(self, name: str, version: str | None = None) -> PromptTemplateConfig:
        """Retrieves a reusable section partial template by name and optional version."""
        n = name.lower()
        v = version or self.latest_partial_versions.get(n)
        if not v or (n, v) not in self.partial_templates:
            if (n, "1.0") in self.partial_templates:
                return self.partial_templates[(n, "1.0")]
            raise KeyError(f"No partial template registered for '{name}' (version '{version}')")
        return self.partial_templates[(n, v)]

    def register_mode(self, mode: str, version: str, template: PromptTemplateConfig) -> None:
        """Registers a mode-level prompt template."""
        key = (mode.lower(), version)
        self.mode_templates[key] = template
        self.latest_mode_versions[mode.lower()] = version
        self.logger.debug("Registered mode prompt template", mode=mode, version=version)

    def register_graph_node(
        self, graph: str, node: str, version: str, template: PromptTemplateConfig
    ) -> None:
        """Registers a graph node fragment prompt template."""
        key = (graph.lower(), node.lower(), version)
        self.graph_node_templates[key] = template
        self.latest_node_versions[(graph.lower(), node.lower())] = version
        self.logger.debug(
            "Registered graph node prompt template", graph=graph, node=node, version=version
        )

    def get_mode_template(self, mode: str, version: str | None = None) -> PromptTemplateConfig:
        """Retrieves a mode-level prompt template by mode and optional version."""
        m = mode.lower()
        v = version or self.latest_mode_versions.get(m)
        if not v or (m, v) not in self.mode_templates:
            if (m, "1.0") in self.mode_templates:
                return self.mode_templates[(m, "1.0")]
            raise UnknownModeError(
                f"No prompt template registered for mode '{mode}' (version '{version}')"
            )
        return self.mode_templates[(m, v)]

    def get_graph_node_template(
        self, graph: str, node: str, version: str | None = None
    ) -> PromptTemplateConfig:
        """Retrieves a graph node prompt fragment."""
        g = graph.lower()
        n = node.lower()
        v = version or self.latest_node_versions.get((g, n))
        if not v or (g, n, v) not in self.graph_node_templates:
            if (g, n, "1.0") in self.graph_node_templates:
                return self.graph_node_templates[(g, n, "1.0")]
            raise KeyError(
                f"No prompt template registered for graph '{graph}', node '{node}' (version '{version}')"
            )
        return self.graph_node_templates[(g, n, v)]

    def render(
        self,
        graph: str,
        node: str,
        state: dict[str, Any],
        version: str | None = None,
    ) -> str:
        """Renders a graph node prompt template with variables extracted from state."""
        template = self.get_graph_node_template(graph, node, version)
        system_text = template.system

        format_vars: dict[str, Any] = {}
        for var in template.variables:
            if var in state:
                format_vars[var] = state[var]
            elif var == "user_query":
                format_vars[var] = state.get("user_message", "")
            else:
                format_vars[var] = ""

        try:
            return system_text.format_map(format_vars)
        except Exception:
            rendered = system_text
            for k, val in format_vars.items():
                rendered = rendered.replace(f"{{{k}}}", str(val))
            return rendered
