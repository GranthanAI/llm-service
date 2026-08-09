"""
Live Verification Script for Phase 13: Prompt Engine.
Tests loading YAML prompt templates, registry lookups, prioritized section construction,
and OpenAI-compatible message generation.
"""

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
from app.workflow_engine.workflow_result import WorkflowResult


def main():
    print("\n" + "=" * 70)
    print("LIVE VERIFICATION: PHASE 13 PROMPT ENGINE")
    print("=" * 70 + "\n")

    # 1. Initialize Loader and Registry
    print("Stage 1: Scanning and loading YAML prompt templates...")
    loader = PromptLoader()
    registry = PromptRegistry(loader=loader)
    builder = PromptBuilder(registry=registry)

    print(
        f"   Loaded {len(registry.mode_templates)} mode templates and {len(registry.graph_node_templates)} graph node templates.\n"
    )

    # 2. Setup Context and WorkflowResult
    bundle = ContextBundle(
        memory=MemoryContext(
            long_term_facts=[
                Fact(
                    fact_id="f1",
                    statement="User is a Senior Machine Learning Engineer",
                    confidence=0.98,
                ),
                Fact(fact_id="f2", statement="Prefers PyTorch over JAX", confidence=0.95),
            ],
            short_term_messages=[
                Message(role="user", content="Can you help me optimize my deep learning model?"),
                Message(
                    role="assistant", content="Certainly! Which architecture are you training?"
                ),
            ],
        ),
        graph=GraphContext(
            entities=[
                EntityNode(id="e1", name="VisionTransformer", type="ModelArchitecture"),
                EntityNode(id="e2", name="LionOptimizer", type="Optimizer"),
            ],
            relationships=[
                RelationshipEdge(
                    source_id="LionOptimizer",
                    relation_type="accelerates",
                    target_id="VisionTransformer",
                ),
            ],
        ),
        retrieval=RetrievalContext(
            chunks=[
                DocumentChunk(
                    chunk_id="chk_vit_01",
                    file_id="vit_benchmarks.pdf",
                    content="Lion optimizer achieves 0.4% higher top-1 accuracy on ViT-B/16 with 50% lower optimizer state memory.",
                    score=0.92,
                )
            ]
        ),
    )

    ctx = PipelineContext(
        conversation_id="conv_prompt_live_demo",
        user_id="Dr. Elena Rostova",
        message_id="msg_live_1",
        request_id="req_live_1",
        user_message="Provide the architectural and mathematical trade-offs of switching from AdamW to Lion.",
        context_bundle=bundle,
    )

    result = WorkflowResult(
        mode="smart",
        engine_type="langgraph",
        draft_content=(
            "### Empirical Synthesis: AdamW vs. Lion\n"
            "- Lion requires tracking only first-moment momentum, cutting memory in half.\n"
            "- Update rule uses sign(momentum), imparting strong regularization for large batch sizes."
        ),
        tool_outputs=[
            ToolResult(
                tool_name="web_search",
                success=True,
                data={
                    "results": [
                        {
                            "source": "arxiv",
                            "title": "Symbolic Discovery of Optimization Algorithms",
                            "url": "http://arxiv.org/abs/2302.06675",
                            "snippet": "EvoLved Sign Momentum (Lion) discovered by program search.",
                        }
                    ]
                },
            )
        ],
        conversation_history=bundle.memory.short_term_messages,
        user_message="Provide the architectural and mathematical trade-offs of switching from AdamW to Lion.",
    )

    # 3. Compose Prompt
    print("Stage 2: Composing multi-section prompt...")
    composed = builder.build(result, ctx)

    print("\n" + "=" * 70)
    print("COMPOSED PROMPT SECTIONS & TOKEN METRICS")
    print("=" * 70)
    print(f"Mode: {composed.mode}")
    print(f"Engine Type: {composed.engine_type}")
    print(f"Template Version: {composed.template_version}")
    print(f"Total Tokens: {composed.total_tokens}")
    print(f"Total Sections: {len(composed.sections)}\n")

    for idx, s in enumerate(composed.sections, 1):
        trimmable_str = "TRIMMABLE" if s.trimmable else "LOCKED (NEVER TRIMMED)"
        print(
            f"[{idx}] Section: '{s.name}' | Priority: {s.priority}/10 | Tokens: {s.token_count} | {trimmable_str}"
        )
        preview = s.content.replace("\n", " ")[:90]
        print(f"    Preview: {preview}...\n")

    print("=" * 70)
    print("FINAL OPENAI-COMPATIBLE MESSAGES PAYLOAD")
    print("=" * 70)
    for m in composed.messages:
        print(f"\n--- [Role: {m['role'].upper()}] ---")
        print(m["content"])

    print("\n" + "=" * 70)
    print("Prompt Engine live verification completed successfully!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
