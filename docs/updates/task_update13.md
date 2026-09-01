# Task Update 13: Request Pipeline Orchestrator — Wiring the Kafka Round-Trip to Conversation Service

**Service**: GraphGPT LLM Service (`llm-service`)
**Architecture Spec**: HLD v2.0 Section 5.1 (Complete Request Flow / Canonical Pipeline), Section 24 (Communication with Other Services)
**Status**: Completed, Verified Live & Tested (238/238 Tests Passing)

---

## 1. Executive Summary

Prior to this task, every individual pipeline stage existed and was independently tested — `ContextCollector`, `RequestAnalyzer`, `WorkflowEngine`, `PromptBuilder`, `ContextWindowManager`, `GenerationRouter`, and `StreamingEngine` — but nothing chained them together. `ChatConsumer.handle()` called `self.pipeline.run(event)`, yet `app/main.py` constructed `ChatConsumer()` with no `pipeline` argument, so every consumed `chat.message.created` event was silently swallowed as a no-op. The Kafka ingestion path was, in effect, dead code.

This task adds the missing orchestrator and wires it in, so llm-service now genuinely fulfils its side of the Conversation Service integration contract: consume `chat.message.created`, run the full HLD Section 5.1 pipeline, and publish `chat.response.chunk` / `chat.response.generated` / `memory.update.requested` back to Kafka.

1. **`RequestPipeline` (`app/pipeline/request_pipeline.py`)** — new orchestrator with a single `run(event)` entrypoint that executes, in order: input sanitization + PII redaction, Context Collection, Request Analysis, Workflow Dispatch, Prompt Build, Context Window Management, Generation Router streaming, and finally `StreamingEngine.stream()` (unchanged — already publishes chunk/generated/memory-update events). On any exception mid-pipeline, it builds and publishes an **error-variant** `chat.response.generated` event via a new `KafkaPublisher.publish_error_response()` method instead of raising, so Conversation Service is never left waiting indefinitely for a response that will never arrive.
2. **`main.py` lifespan fix** — `ChatConsumer` is now constructed with a real `RequestPipeline` instance built from the same DI-container objects already initialized during startup (`context_collector`, `request_analyzer`, `workflow_engine`, `prompt_builder`, `context_window_manager`, `generation_router`, `streaming_engine`, `publisher`). No new components were introduced — this is purely wiring.
3. **`ChatMessageCreatedEvent` schema** — added `selected_tools: list[str]` so Conversation Service can pass explicit tool selections through the event; `mode_hint` remains the mode carrier (no redundant `mode`/`tool_mode` fields added).
4. **Contract alignment with Conversation Service** — confirmed and preserved: llm-service remains Kafka-only for chat generation (no gRPC server was added for this purpose, per HLD Section 24's explicit "gRPC is synchronous, internal service-to-service calls only" boundary — llm-service is a gRPC *client* to Memory/Graph/Retrieval, never a server for chat).

---

## 2. What Was NOT Changed

- `StreamingEngine`, `KafkaPublisher.publish_chunk` / `publish_response` / `publish_memory_update` — untouched; these already correctly implemented the HLD's egress contract, they just had no caller.
- No gRPC server, no new proto files, no changes to `app/grpc/clients/*` (Memory/Graph/Retrieval clients remain gRPC-client-only, as designed).
- No changes to Kafka topic names or envelope shapes on llm-service's side — Conversation Service was updated separately to adopt llm-service's existing flat (envelope-less) event shapes and topic names (`chat.response.chunk`, `chat.response.generated`), rather than llm-service changing to match a stale contract.

---

## 3. Test Suite Status

```bash
uv run pytest tests -v
```

- `tests/unit/test_request_pipeline.py` (**new**): **3 / 3 passed** — verifies every pipeline stage is invoked in order, `PipelineContext` is populated correctly before streaming, and a mid-pipeline exception results in a published error event rather than a raised exception.
- `tests/kafka/test_kafka_suite.py` (**extended**): added `test_chat_consumer_with_real_pipeline_publishes_chunk_and_generated` — wires `ChatConsumer` with a real `RequestPipeline` (mocked collector/analyzer/workflow/prompt/context-window/router, real `StreamingEngine` + `KafkaPublisher`) and asserts both `chat.response.chunk` and `chat.response.generated` are actually published to the underlying producer — closing the loop that was previously a silent no-op.
- All previously passing suites remain green.

**Total**: **238 / 238 tests passing** (100% success rate), up from 234 pre-task (+3 new pipeline tests, +1 new end-to-end Kafka test).

---

## 4. Next Steps

With the canonical pipeline now actually wired end-to-end, remaining follow-ups (tracked separately, not part of this task):
- Conversation Service's pending-response timeout worker (added on its side) is the current safety net if llm-service is unreachable — no additional retry/backoff was added on llm-service's ingestion side.
- Memory Service still references a stale `chat.response.completed` topic name in its own code; this is a pre-existing inconsistency outside llm-service's scope and was intentionally left untouched.
