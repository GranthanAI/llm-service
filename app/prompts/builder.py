"""
Prompt Builder and Section Composition Engine.
Implements LLD v2.0 Section 17.1, 17.4, 17.5, and 17.6.
"""

from typing import Any

import structlog
import tiktoken

from app.config.logging import get_logger
from app.context.schemas import GraphContext, MemoryContext, Message, RetrievalContext
from app.models.pipeline_context import PipelineContext
from app.models.tool import ToolResult
from app.prompts.registry import PromptRegistry
from app.prompts.schemas import ComposedPrompt, PromptSection, PromptTemplateConfig
from app.utils.metrics import PROMPT_BUILD_DURATION
from app.utils.tracing import get_tracer
from app.workflow_engine.workflow_result import WorkflowResult

logger = get_logger("prompt_builder")


def _count_tokens(text: str) -> int:
    """Counts tokens using tiktoken cl100k_base with character-heuristic fallback."""
    if not text:
        return 0
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        # Standard ~4 characters per token heuristic
        return max(1, len(text) // 4)


class PromptBuilder:
    """
    Shared Prompt Composition Engine.
    Called identically whether the upstream WorkflowResult came from a ModeHandler or a LangGraph graph.
    Builds prioritized PromptSections and formats final OpenAI-compatible message payloads.
    """

    def __init__(
        self,
        registry: PromptRegistry,
        logger_instance: structlog.stdlib.BoundLogger | None = None,
    ):
        self.registry = registry
        self.logger = logger_instance or logger
        self.tracer = get_tracer()

    def build(self, result: WorkflowResult, ctx: PipelineContext) -> ComposedPrompt:
        """Composes a structured ComposedPrompt from WorkflowResult and PipelineContext."""
        with self.tracer.start_as_current_span("prompt_builder.build") as span:
            span.set_attribute("mode", result.mode)
            span.set_attribute("engine_type", result.engine_type)

            with PROMPT_BUILD_DURATION.labels(mode=result.mode).time():
                template = self.registry.get_mode_template(result.mode)
                sections = self._build_sections(result, ctx, template)

                total_tokens = sum(s.token_count for s in sections)

                # Separate system vs context/user blocks for message formatting
                system_sections = [s.content for s in sections if s.name == "system"]
                system_prompt = "\n\n".join(system_sections)

                user_sections = [s.content for s in sections if s.name != "system"]
                user_prompt = "\n\n".join(user_sections)

                messages: list[dict[str, Any]] = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                if user_prompt:
                    messages.append({"role": "user", "content": user_prompt})

                self.logger.debug(
                    "Prompt composed successfully",
                    mode=result.mode,
                    engine_type=result.engine_type,
                    sections_count=len(sections),
                    total_tokens=total_tokens,
                )

                return ComposedPrompt(
                    sections=sections,
                    total_tokens=total_tokens,
                    messages=messages,
                    mode=result.mode,
                    engine_type=result.engine_type,
                    template_version=template.version,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )

    def _build_sections(
        self,
        result: WorkflowResult,
        ctx: PipelineContext,
        template: PromptTemplateConfig,
    ) -> list[PromptSection]:
        """Constructs prioritized PromptSections according to LLD Section 17.4."""
        sections: list[PromptSection] = []

        # 1. System Prompt Section (Priority 10 - Highest, never trimmed)
        system_content = self._format_system(result, ctx, template)
        sections.append(
            PromptSection(
                name="system",
                content=system_content,
                priority=10,
                token_count=_count_tokens(system_content),
                trimmable=False,
            )
        )

        bundle = ctx.context_bundle

        # 2. Long-term Memory Facts (Priority 7)
        if bundle and bundle.memory and bundle.memory.long_term_facts:
            mem_content = self._format_memory_long_term(bundle.memory)
            if mem_content:
                sections.append(
                    PromptSection(
                        name="long_term_memory",
                        content=mem_content,
                        priority=7,
                        token_count=_count_tokens(mem_content),
                        trimmable=True,
                    )
                )

        # 3. Knowledge Graph Context (Priority 5)
        if bundle and bundle.graph and (bundle.graph.entities or bundle.graph.relationships):
            graph_content = self._format_graph(bundle.graph)
            if graph_content:
                sections.append(
                    PromptSection(
                        name="graph_context",
                        content=graph_content,
                        priority=5,
                        token_count=_count_tokens(graph_content),
                        trimmable=True,
                    )
                )

        # 4. Retrieval Context (Priority 5)
        if bundle and bundle.retrieval and bundle.retrieval.chunks:
            retrieval_content = self._format_retrieval(bundle.retrieval)
            if retrieval_content:
                sections.append(
                    PromptSection(
                        name="retrieval",
                        content=retrieval_content,
                        priority=5,
                        token_count=_count_tokens(retrieval_content),
                        trimmable=True,
                    )
                )

        # 5. Tool Results (Priority 6)
        if result.tool_outputs:
            tools_content = self._format_tool_results(result.tool_outputs)
            if tools_content:
                sections.append(
                    PromptSection(
                        name="tool_results",
                        content=tools_content,
                        priority=6,
                        token_count=_count_tokens(tools_content),
                        trimmable=True,
                    )
                )

        # 6. Draft Content from LangGraph (Priority 9 - High Priority)
        if result.draft_content:
            draft_section_content = f"## Synthesized Evidence & Analysis\n{result.draft_content}"
            sections.append(
                PromptSection(
                    name="draft_content",
                    content=draft_section_content,
                    priority=9,
                    token_count=_count_tokens(draft_section_content),
                    trimmable=True,
                )
            )

        # 7. Conversation History (Priority 8)
        history = result.conversation_history or (
            bundle.memory.short_term_messages if bundle and bundle.memory else []
        )
        if history:
            history_content = self._format_history(history)
            if history_content:
                sections.append(
                    PromptSection(
                        name="conversation_history",
                        content=history_content,
                        priority=8,
                        token_count=_count_tokens(history_content),
                        trimmable=True,
                    )
                )

        # 8. User Query (Priority 10 - Highest, never trimmed)
        user_query_content = f"## Current User Message\n{result.user_message or ctx.user_message}"
        sections.append(
            PromptSection(
                name="user_query",
                content=user_query_content,
                priority=10,
                token_count=_count_tokens(user_query_content),
                trimmable=False,
            )
        )

        return sections

    def _format_system(
        self,
        result: WorkflowResult,
        ctx: PipelineContext,
        template: PromptTemplateConfig,
    ) -> str:
        """Renders system prompt template with user profile information."""
        user_name = ctx.user_id or "User"
        base_system = template.system.replace("{user_name}", user_name)
        return base_system.strip()

    def _format_memory_long_term(self, memory: MemoryContext) -> str:
        """Formats long-term memory facts as a structured section."""
        lines = ["## What I Know About You (Long-term Facts)"]
        for fact in memory.long_term_facts:
            lines.append(f"- {fact.statement} (Confidence: {fact.confidence:.2f})")
        return "\n".join(lines)

    def _format_graph(self, graph: GraphContext) -> str:
        """Formats knowledge graph entities and relationships."""
        lines = ["## Knowledge Graph Relationships"]
        if graph.entities:
            lines.append("Entities:")
            for e in graph.entities:
                lines.append(f"- {e.name} ({e.type})")
        if graph.relationships:
            lines.append("Relationships:")
            for r in graph.relationships:
                lines.append(f"- {r.source_id} --[{r.relation_type}]--> {r.target_id}")
        return "\n".join(lines)

    def _format_retrieval(self, retrieval: RetrievalContext) -> str:
        """Formats document retrieval chunks."""
        lines = ["## Retrieved Document Context"]
        for idx, chunk in enumerate(retrieval.chunks, 1):
            source_info = f" (Source: {chunk.file_id})" if chunk.file_id else ""
            lines.append(f"[{idx}]{source_info} {chunk.content.strip()}")
        return "\n".join(lines)

    def _format_tool_results(self, tool_outputs: list[ToolResult]) -> str:
        """Formats normalized tool execution outputs."""
        lines = ["## Live Tool Execution Results"]
        for tr in tool_outputs:
            if not tr.success:
                lines.append(f"- Tool `{tr.tool_name}`: Error - {tr.error}")
                continue

            if isinstance(tr.data, dict) and "results" in tr.data:
                res_list = tr.data.get("results", [])
                lines.append(f"- Tool `{tr.tool_name}` ({len(res_list)} sources retrieved):")
                for r in res_list:
                    title = r.get("title", "Result")
                    url = r.get("url", "")
                    snippet = r.get("snippet", "")
                    source = r.get("source", "web").upper()
                    lines.append(f"  * [{source}] [{title}]({url}): {snippet}")
            else:
                lines.append(f"- Tool `{tr.tool_name}`: {tr.data}")
        return "\n".join(lines)

    def _format_history(self, messages: list[Message]) -> str:
        """Formats short-term conversation history."""
        lines = ["## Recent Conversation History"]
        for msg in messages:
            role = msg.role.capitalize()
            lines.append(f"{role}: {msg.content.strip()}")
        return "\n".join(lines)
