# P5: llm-service, report

**Commits (local, not pushed):** `3ab7cbb` P5a · `e322c04` P5b · `8918fe8` P5c · `1b1570f` P5d · `5890c9d` P5e · `e29f86c` P5f · plus the commit that adds this report.
Range: `626020e..HEAD`.

## What was built
- **P5a** `config/models.yaml` + `ModelRegistry` (validated at startup), typed `BillingGrant` (schemas 2.3 and 2.4), per-request group chains
  (`providers/group_chain.py`: only the granted group, never higher), grant cap through the existing context manager (`resolve_context_budget(input_cap=)`).
- **P5b** `AnthropicAdapter`, `OpenAIAdapter` (`anthropic` added to the existing environment with `uv add`; `openai` was already a dependency).
- **P5c** `app/usage` (`CallUsage`, per-request `UsageCollector` in a ContextVar, `usage_purpose`), real usage in every adapter, character-count fallback deleted.
- **P5d** `chat.response.generated` 2.2 / `chat.response.cancelled` billing fields, `memory.update.requested.request_message_id`, one-terminal-event guard.
- **P5e** vendored billing proto + stubs, `BillingClient`, Manim reserve (`off|shadow|enforce`), `artifact.manim.rendered` 1.0.
- **P5f** catalog check, `tests/fixtures/billing_models.json`, README, `.env.example`, `scripts/p5_exit_check.py`.

## Tests run (only these files; the full suite was NOT run)
| Files | Result |
| --- | --- |
| tests/unit/test_group_routing.py, test_catalog_check.py, test_usage_collection.py, test_terminal_events.py, test_manim_billing.py | 75 passed (15 s) |
| tests/providers/test_anthropic_openai.py, test_provider_suite.py; tests/unit/test_streaming_engine.py, test_request_pipeline.py, test_provider_chain.py, test_visualization_jobs.py, test_context_budget.py, test_context_refit.py, test_request_analyzer.py, test_groq_tool_call_retry.py, test_providers.py, test_memory_update_identity.py, test_config.py; tests/kafka/test_kafka_suite.py | 161 passed (34 s) |

No test makes a live provider call (fake SDK streams, a fake gRPC billing server, a fake Kafka producer).

## Exit check: NOT DONE
Billing was not running (nothing listens on 8090; the process on 50051 is **memory-service**, whose gRPC default clashes with billing's), the llm-service on 8003 is an
older process, and the check makes **real Groq / Smart-model calls**, which I did not do without your say-so. To run it:
1. Kafka must advertise `127.0.0.1:9092`. Billing: `PAYMENT_SIMULATION=true`, API + worker, `KAFKA_BOOTSTRAP_SERVERS=127.0.0.1:9092`, gRPC on a free port
   (`GRPC_PORT=50061`); in billing-service: `uv run uvicorn billing.main_api:app --port 8090` and `uv run python -m billing.main_worker`.
2. Restart llm-service on this code with `KAFKA_BOOTSTRAP_SERVERS=127.0.0.1:9092` and `GROQ_API_KEY` set.
3. `uv run python scripts/p5_exit_check.py` prints the 2.2 event, its `calls`, PASS/CHECK and the SQL:
   `docker exec -i graphgpt-postgres psql -U postgres -d billing_db -c "SELECT kind, source_id, model, model_group, input_tokens, cached_input_tokens, output_tokens, cost_micro_inr, hold_id FROM usage_events WHERE source_id = '<request_message_id>';"`
   Expect a row with `cost_micro_inr > 0`. The test user id needs a billing account (billing creates the Free account for an unseen user).

## Call-site coverage (every provider / SDK call)
| Call site | Purpose | How it is covered |
| --- | --- | --- |
| `GroqAnalysisClient.complete` (request analyzer; also reused by the query rewriters) | `request_analysis` / `query_rewrite` | reports `response.usage`; purpose set by the pipeline around `request_analyzer.analyze` and `query_rewriter.rewrite` |
| `RequestAnalyzer` fallbacks: `adapter.execute` (Gemini/Groq) and `nvidia_adapter.execute` | `request_analysis` | the adapters report in `execute` under the same purpose |
| `SearchQueryRewriter` (`tools/query_rewrite.py`: `client.complete`, `adapter.execute`) | `query_rewrite` | as above |
| Retrieval query rewrite (`context/retrieval_query.py`) | `query_rewrite` | the whole `context_collector.collect` runs under `query_rewrite` (its only model call) |
| Answer: `ProviderChain` / `GenerationRouter` -> Groq, Gemini, NVIDIA, Anthropic, OpenAI `stream` | `answer` | each adapter reports at the end of the stream (provider figures), or an estimate if the stream died midway; every attempt that reached the provider counts |
| Cancelled answer | `answer` | the stream is closed and the partial output is recorded as an estimate (`usage_estimated`) |
| Workflow engine / LangGraph smart and deep-research nodes | `workflow` / `research` | **no model call exists in these nodes today** (the planner node reuses the analyzer's plan; `llm_client` is unused). The purposes are wired around `workflow_engine.execute`, so a future call is attributed. |
| Mermaid repair (`POST /v1/visualizations/repair`, browser initiated) | none | **not metered**: no event carries it (cost only; a decision for you) |
| Manim repair (`repair_block`, inline or through `/internal/.../repair`) | `manim_plan` | collected per job and published on `artifact.manim.rendered` |
| Context compaction (`compact_history`) | n/a | heuristic, no model call |

## Terminal-event paths (exactly one per request, per process)
1. Success: `StreamingEngine.stream` -> `publish_response` (then `publish_memory_update`).
2. Cancel: `StreamingEngine.stream`, cancellation branch -> `publish_cancellation` (the stream is closed first, partial usage recorded).
3. Any exception in the pipeline (provider timeout, all providers failed, context overflow, a grant for a group this service cannot serve, a failed prompt build)
   -> `publish_error_response` with the calls made so far.
4. Not terminal events: DLQ messages (deserialization / contract errors, no response is owed) and duplicate deliveries skipped by the consumer's idempotency cache.

A guard in the publisher (per request message id, bounded to 20,000) drops a second terminal event. This fixed a real duplicate: an exception raised by
`publish_memory_update` AFTER the success event used to add an error event as well.

## Model IDs (checked against the providers' docs on 2026-11-09)
Verified: `claude-haiku-5-5`, `claude-sonnet-5-5`, `claude-opus-5-5` (1M window, 128K output); `gpt-6-luna`, `gpt-6.1-sol`, `gpt-6-astra` (1.05M window, 922K max input,
128K output); `gemini-2.5-flash-lite`, `gemini-2.5-flash`, `gemini-3.1-flash-lite`; the Groq ids were already in use.

**Not verified, disabled:** `gemini-3.1-pro`. The docs list the API string `gemini-3.1-pro-preview` (a preview model), not `gemini-3.1-pro`. `models.yaml` carries
`provider_model: gemini-3.1-pro-preview` with `enabled: false` until you decide whether a preview model may serve the paid Best group. Gemini context windows (1,048,576)
are not verified. Also: the docs say **`gemini-2.5-flash` is deprecated and routes to `gemini-3.5-flash-lite` on 1 Dec 2026** (it is in Fast and Smart).

## Deviations
- **Anthropic input tokens** are reported as the TOTAL prompt (uncached + cache write + cache read) so `cached_input_tokens` is a subset of `input_tokens` for every provider.
  The plan says "input from message_start.message.usage", which excludes cache reads. Billing must treat `cached_input_tokens` as part of `input_tokens`.
- **Gemini thinking tokens** (`thoughts_token_count`) are added to `output_tokens` (billed as output, not part of `candidates_token_count`).
- **Gemini is reported as provider `google`** (billing's name in `model_prices.yaml`); the adapter itself is still `gemini`.
- **No `temperature` is sent to Anthropic or OpenAI** (current models restrict it) and `max_completion_tokens` is used for OpenAI. Neither could be checked live.
- **Manim `attempt`** is a new job field `billing_attempt` (the existing `attempt` counts render attempts inside one run); a failed job submitted again is attempt + 1.
- **Fixed a runner bug:** a spec that failed validation before rendering stranded the job in `validating` (VALIDATING -> VALIDATING was refused and read as a cancellation).
  It is now allowed; billing needs the job to end to release the hold and receive the event.
- Two unit assertions of the removed character-count fallback were updated (`test_streaming_engine.py`).

## Decisions needed
1. **Exactly once across a restart.** Within a process there is exactly one terminal event. A crash after publishing and before the offset commit re-delivers the request and
   publishes a second event with the same `request_message_id` (billing dedupes on it). A durable guard needs a store (Redis/Valkey SETNX): say if you want it.
2. The Gemini 3.1 Pro preview id and the 1 Dec 2026 `gemini-2.5-flash` deprecation.
3. Mermaid repair model calls are not metered.
4. The gRPC port clash: memory-service and billing both default to 50051.
5. `tavily_api_key` is unused (left as is).

## `docker system df` (end of session)
```
TYPE            TOTAL     ACTIVE    SIZE      RECLAIMABLE
Images          20        12        11.17GB   2.52GB (22%)
Containers      15        11        366.1MB   6.984MB (1%)
Local Volumes   27        19        8.032GB   131.4MB (1%)
Build Cache     4         0         0B        0B
```
