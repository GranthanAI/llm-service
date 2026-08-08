"""
Request Analyzer Prompt Template and Builder.
Implements LLD v2.0 Section 10.2.
"""

import json
from typing import Any

from app.models.pipeline_context import PipelineContext


class AnalysisPromptBuilder:
    """
    Assembles the structured analysis system prompt and user payload for Groq (Call 1).
    """

    SYSTEM_TEMPLATE = """You are GraphGPT's Request Analyzer. Given a user message and baseline context, analyze the request and return a structured execution plan.

## User Context
{formatted_memory}

## Knowledge Graph
{formatted_graph}

## Retrieved Documents
{formatted_retrieval}

## Available Tools
{tools_json_schema}

## Client-Provided Mode Hint (advisory only — you may confirm or override)
{mode_hint}

## Output Format (strict JSON object, no markdown, no code blocks)
{{
  "intent": "GENERAL_CHAT | QUESTION_ANSWERING | CODE_GENERATION | CODE_DEBUGGING | CODE_EXPLANATION | RESEARCH | WEB_SEARCH | DOCUMENT_ANALYSIS | TUTORING | CREATIVE_WRITING | REASONING",
  "mode": "default | tutor | code | ask_files | web_search | smart | deep_research",
  "skill": "general_chat | tutor | coding | research | writing | reasoning",
  "reasoning": "DIRECT | CHAIN_OF_THOUGHT | REACT",
  "tools": [
    {{
      "tool_name": "string",
      "params": {{}},
      "parallel": true,
      "required": false
    }}
  ],
  "max_iterations": 1,
  "suggested_temperature": 0.7,
  "analysis_confidence": 1.0
}}
"""

    def build_prompt(
        self,
        ctx: PipelineContext,
        tools_schema: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, str]]:
        """Construct system and user messages for Groq JSON planning inference."""
        # 1. Format memory
        formatted_memory = "No previous conversation context available."
        if ctx.context_bundle and ctx.context_bundle.memory:
            mem = ctx.context_bundle.memory
            lines = []
            if mem.long_term_facts:
                lines.append("Long-Term User Facts:")
                for fact in mem.long_term_facts:
                    lines.append(f"- {fact.statement}")
            if mem.short_term_messages:
                lines.append("\nRecent Conversation History:")
                for msg in mem.short_term_messages[-5:]:  # Last 5 messages
                    lines.append(f"[{msg.role.upper()}]: {msg.content}")
            if lines:
                formatted_memory = "\n".join(lines)

        # 2. Format knowledge graph
        formatted_graph = "No knowledge graph context available."
        if ctx.context_bundle and ctx.context_bundle.graph:
            graph = ctx.context_bundle.graph
            lines = []
            if graph.subgraph_summary:
                lines.append(f"Summary: {graph.subgraph_summary}")
            if graph.entities:
                lines.append("Entities: " + ", ".join(e.name for e in graph.entities[:10]))
            if lines:
                formatted_graph = "\n".join(lines)

        # 3. Format retrieval documents
        formatted_retrieval = "No document chunks retrieved."
        if ctx.context_bundle and ctx.context_bundle.retrieval:
            retrieval = ctx.context_bundle.retrieval
            if retrieval.chunks:
                lines = []
                for i, chunk in enumerate(retrieval.chunks[:3], start=1):
                    snippet = chunk.content[:200].replace("\n", " ")
                    lines.append(f"[{i}] (Score: {chunk.score:.2f}): {snippet}...")
                formatted_retrieval = "\n".join(lines)

        # 4. Format tools
        tools_json = json.dumps(tools_schema or [], indent=2)

        # 5. Format mode hint
        mode_hint_str = ctx.mode_hint.value if ctx.mode_hint else "none"

        # Build populated system prompt
        system_content = self.SYSTEM_TEMPLATE.format(
            formatted_memory=formatted_memory,
            formatted_graph=formatted_graph,
            formatted_retrieval=formatted_retrieval,
            tools_json_schema=tools_json,
            mode_hint=mode_hint_str,
        )

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": ctx.user_message},
        ]
