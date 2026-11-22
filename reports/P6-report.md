# P6: llm-service, quick wins (Q4, Q5, Q7), report

Not pushed. Docs: `docs/quick_wins.md`.

## What changed
- **L1:** `chat.message.created` accepts 2.5 (`think`, `temporary`, `preferences.suggest_followups`, defaults false / false / true). `chat.response.generated` is 2.3 with `think`.
- **L2 temporary chat:** no memory call at all (`_fetch_memory` returns an empty `MemoryContext`), no `memory.update.requested`; graph context unchanged.
- **L3 think longer:** `think:` block per model in `config/models.yaml`; Groq gpt-oss `reasoning_effort` low -> high (medium with a `think_effort_fallback` log when the TPM bucket leaves < 4,096 output tokens), OpenAI `reasoning_effort=high`, Claude `thinking={"type":"enabled","budget_tokens":4096,"display":"omitted"}`, Gemini `ThinkingConfig(thinking_level=HIGH, include_thoughts=False)`. Thinking text is never streamed; reasoning tokens are counted from the provider's usage. Output budget x2 (capped by `think.max_output_tokens`; Groq not doubled). On Auto, think-capable models of the granted group go first; an explicit model is respected; none capable = normal answer with `think: false`. `chat.response.generated.think` = the answering call really reasoned. `catalog_check` reports `think_mismatch`; `tests/fixtures/billing_models.json` now has billing's `think`/`available`.
- **L4 follow-ups:** `app/followups/` + `app/prompts/followups/followups_v1.yaml`; background task after a successful publish (answer >= 200 chars, preference on); `chat.response.suggestions` 1.0; cost on `usage.metered` (feature `followups`); metric `followups_total{outcome}`; settings `FOLLOWUPS_ENABLED`, `FOLLOWUP_MODEL`, `FOLLOWUP_TIMEOUT_MS`, `FOLLOWUP_MAX_TOKENS`.

## Tests
- New `tests/unit/test_quick_wins.py`: 33 passed (event fields, no memory call, no memory event, registry parsing, each adapter's request with and without think, no thinking text in chunks, usage with reasoning tokens, Auto ordering, chain `think` reporting, catalog mismatch, follow-ups published / skipped / timeout / invalid / error / metered / Hinglish kept, streaming-engine ordering, payload shape).
- Whole suite (`tests`, without `integration` and `grpc`): **917 passed, 5 failed before I fixed 3 contract-version tests; the 2 remaining failures fail on untouched HEAD too** (`test_provider_429`, `test_visualization_core`). The 3 touched tests were updated for the new versions (2.3 -> 2.5 request, 2.2 -> 2.3 final event, `think` in the final-event field set, `test_custom_modes` default version).
- Not run: `tests/integration`, `tests/grpc`.

## Not verified
- **No live provider calls** (your rule: no live LLM calls in tests), so no real think call per provider and no real follow-up generation, and no token counts from logs. The parameter shapes come from the installed SDK types (groq 1.6, openai 2.53, anthropic 2.6, google-genai 2.17); whether `claude-*-5-5`, `gpt-6*` and `gemini-3.*` accept them is unconfirmed. A model that rejects the parameter fails before its first token and the chain moves to the next one. Billing flags all ten offered models `think: true` and so does this registry; nothing is `kind: none`.
- Restarted with `make restart-llm` (healthy on 8003). `chat.response.suggestions` needs to exist as a Kafka topic.
- A Groq-only deployment on the free tier will often use `medium`; no counts yet.
