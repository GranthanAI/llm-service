"""
Unit Tests for Prompt Engine (PromptLoader, PromptRegistry, PromptBuilder, and ComposedPrompt).
Implements test coverage for Phase 13 as per LLD v2.0 Section 17.
"""

from pathlib import Path

from app.context.schemas import (
    ContextBundle,
    DocumentChunk,
    EntityNode,
    Fact,
    GraphContext,
    MemoryContext,
    Message,
    RelationshipEdge,
    RetrievalContext,
)
from app.models.pipeline_context import PipelineContext
from app.models.tool import ToolResult
from app.prompts.builder import PromptBuilder
from app.prompts.loader import PromptLoader
from app.prompts.registry import PromptRegistry
from app.prompts.schemas import PromptTemplateConfig
from app.workflow_engine.workflow_result import WorkflowResult

# --- 1. PromptLoader Tests ---


def test_prompt_loader_loads_all_templates():
    """Verify PromptLoader scans directory and loads all YAML templates."""
    loader = PromptLoader()
    templates = loader.load_all()
    assert len(templates) >= 7

    names = {t.name for t in templates}
    assert "default" in names
    assert "tutor" in names
    assert "code" in names
    assert "ask_files" in names
    assert "web_search" in names
    assert "smart" in names
    assert "deep_research" in names
    assert "base_system" in names
    assert "safety_guardrail" in names
    assert "json_schema_format" in names


def test_prompt_loader_validates_schema(tmp_path: Path):
    """Verify PromptLoader enforces required schema fields."""
    invalid_yaml = tmp_path / "invalid.yaml"
    invalid_yaml.write_text("version: '1.0'\n", encoding="utf-8")

    loader = PromptLoader(prompts_dir=tmp_path)
    configs = loader.load_all()
    assert len(configs) == 0


# --- 2. PromptRegistry Tests ---


def test_prompt_registry_mode_registration_and_lookup():
    """Verify registering and retrieving mode-level prompt templates."""
    loader = PromptLoader()
    registry = PromptRegistry(loader=loader)

    # Lookup all 7 modes
    for mode in ["default", "tutor", "code", "ask_files", "web_search", "smart", "deep_research"]:
        tpl = registry.get_mode_template(mode)
        assert tpl.mode == mode
        assert len(tpl.system) > 0

    # Lookup system templates
    base_sys = registry.get_system_template("base_system")
    assert "GraphGPT" in base_sys.system

    safety_sys = registry.get_system_template("safety_guardrail")
    assert "Safety & Alignment" in safety_sys.system

    json_sys = registry.get_system_template("json_schema_format")
    assert "Output Format Specification" in json_sys.system

    # Lookup section partial templates
    mem_tpl = registry.get_template("memory_section")
    assert "What I Know About You" in mem_tpl.system

    graph_tpl = registry.get_template("graph_section")
    assert "Knowledge Graph Relationships" in graph_tpl.system

    retrieval_tpl = registry.get_template("retrieval_section")
    assert "Retrieved Document Context" in retrieval_tpl.system

    # Custom registration
    custom_cfg = PromptTemplateConfig(
        name="custom_mode",
        version="2.0",
        mode="custom_mode",
        system="Custom system prompt for {user_name}",
    )
    registry.register_mode("custom_mode", "2.0", custom_cfg)
    retrieved = registry.get_mode_template("custom_mode", "2.0")
    assert retrieved.version == "2.0"


def test_prompt_registry_graph_node_rendering():
    """Verify graph node template rendering with variable interpolation."""
    loader = PromptLoader()
    registry = PromptRegistry(loader=loader)

    state = {
        "user_message": "What is reinforcement learning?",
        "queries_issued": ["RL survey", "PPO algorithm"],
        "search_results": "Result 1: PPO",
    }
    rendered = registry.render(graph="deep_research", node="analyze", state=state)
    assert "What is reinforcement learning?" in rendered
    assert "RL survey" in rendered


# --- 3. PromptBuilder Section Composition Tests ---


def test_prompt_builder_builds_prioritized_sections_from_mode_handler():
    """Verify PromptBuilder constructs prioritized sections from a ModeHandler result."""
    loader = PromptLoader()
    registry = PromptRegistry(loader=loader)
    builder = PromptBuilder(registry=registry)

    # 1. Setup rich context bundle
    bundle = ContextBundle(
        memory=MemoryContext(
            long_term_facts=[
                Fact(fact_id="f1", statement="User is an AI researcher", confidence=0.95)
            ],
            short_term_messages=[
                Message(role="user", content="Hello GraphGPT"),
                Message(role="assistant", content="Hello! How can I assist you?"),
            ],
        ),
        graph=GraphContext(
            entities=[EntityNode(id="e1", name="PyTorch", type="Framework")],
            relationships=[
                RelationshipEdge(source_id="PyTorch", relation_type="implements", target_id="AdamW")
            ],
        ),
        retrieval=RetrievalContext(
            chunks=[
                DocumentChunk(
                    chunk_id="chk_1",
                    file_id="doc_vit.pdf",
                    content="Vision Transformers process image patches as tokens.",
                    score=0.9,
                )
            ]
        ),
    )

    ctx = PipelineContext(
        conversation_id="conv_prompt_test",
        user_id="Alice",
        message_id="msg_1",
        request_id="req_1",
        user_message="Explain AdamW optimization in PyTorch",
        context_bundle=bundle,
    )

    result = WorkflowResult(
        mode="tutor",
        engine_type="mode_handler",
        draft_content=None,
        tool_outputs=[
            ToolResult(
                tool_name="web_search",
                success=True,
                data={
                    "results": [
                        {
                            "source": "arxiv",
                            "title": "Decoupled Weight Decay",
                            "url": "https://arxiv.org/abs/1711.05101",
                            "snippet": "AdamW fixes L2 regularization in Adam.",
                        }
                    ]
                },
            )
        ],
        conversation_history=bundle.memory.short_term_messages,
        user_message="Explain AdamW optimization in PyTorch",
    )

    composed = builder.build(result, ctx)

    # Verify ComposedPrompt attributes
    assert composed.mode == "tutor"
    assert composed.engine_type == "mode_handler"
    assert composed.total_tokens > 0
    assert len(composed.sections) >= 6
    assert len(composed.messages) == 2  # system message + user/context message

    # Verify section priority ordering and contents
    section_map = {s.name: s for s in composed.sections}
    assert section_map["system"].priority == 10
    assert not section_map["system"].trimmable
    assert "Alice" in section_map["system"].content

    assert section_map["long_term_memory"].priority == 7
    assert "User is an AI researcher" in section_map["long_term_memory"].content

    assert section_map["graph_context"].priority == 5
    assert "PyTorch" in section_map["graph_context"].content

    assert section_map["retrieval"].priority == 5
    assert "Vision Transformers" in section_map["retrieval"].content

    assert section_map["tool_results"].priority == 6
    assert "Decoupled Weight Decay" in section_map["tool_results"].content

    assert section_map["conversation_history"].priority == 8
    assert "Hello GraphGPT" in section_map["conversation_history"].content

    assert section_map["user_query"].priority == 10
    assert not section_map["user_query"].trimmable
    assert "Explain AdamW optimization in PyTorch" in section_map["user_query"].content


def test_prompt_builder_handles_langgraph_draft_content():
    """Verify PromptBuilder assigns priority 9 to LangGraph draft_content."""
    loader = PromptLoader()
    registry = PromptRegistry(loader=loader)
    builder = PromptBuilder(registry=registry)

    ctx = PipelineContext(
        conversation_id="conv_lg_test",
        user_id="Bob",
        message_id="msg_2",
        request_id="req_2",
        user_message="Compare MoE routing mechanisms",
        context_bundle=ContextBundle.empty(),
    )

    result = WorkflowResult(
        mode="smart",
        engine_type="langgraph",
        draft_content="### Multi-step agentic analysis on MoE routing",
        tool_outputs=[],
        conversation_history=[],
        user_message="Compare MoE routing mechanisms",
    )

    composed = builder.build(result, ctx)
    assert composed.mode == "smart"
    assert composed.engine_type == "langgraph"

    section_map = {s.name: s for s in composed.sections}
    assert "draft_content" in section_map
    assert section_map["draft_content"].priority == 9
    assert "Multi-step agentic analysis" in section_map["draft_content"].content
