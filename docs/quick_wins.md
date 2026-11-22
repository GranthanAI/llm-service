# Quick wins in llm-service: think longer, follow-ups, temporary chats

## Events
- `chat.message.created` **2.5** (consumed, additive): `think`, `temporary`, `preferences.suggest_followups` (default true).
- `chat.response.generated` **2.3** (published): `think` = the answering call really used provider reasoning.
- NEW `chat.response.suggestions` **1.0** (published, flat JSON, keyed by `conversation_id`): `event_id, occurred_at (…Z),
  conversation_id, user_id, request_message_id, response_message_id, suggestions (2 or 3, each 1..120 chars, no newlines), model`.
  At most once per answer, only after a successful `chat.response.generated`. The topic must exist.
- `memory.update.requested` is NOT published for a temporary chat.

## Temporary chats
`temporary: true` -> `ContextCollector._fetch_memory` returns an empty `MemoryContext` without calling memory-service (no long-term
facts, no short-term window; the event's own history is the conversation context) and no `memory.update.requested` is published.
Graph context is still fetched (graph-service skips the temporary conversation on its side).

## Think longer
`config/models.yaml` -> `think: {kind: none|effort|budget, normal, think, budget_tokens, max_output_tokens}` per model.

| Family | Control | Think value |
| --- | --- | --- |
| Groq gpt-oss | `reasoning_effort` | `low` normally, `high` when thinking; `medium` (logged `think_effort_fallback`) when the TPM bucket leaves < 4,096 output tokens |
| OpenAI GPT-6 | `reasoning_effort` | `high` |
| Claude | `thinking={"type":"enabled","budget_tokens":4096,"display":"omitted"}` (max_tokens is raised above the budget) | |
| Gemini | `ThinkingConfig(thinking_level=HIGH, include_thoughts=False)` (budget kind: `thinking_budget`) | |

The parameter shapes come from the installed SDK types; whether each model accepts them could not be checked without live calls.
Thinking text is never streamed or requested as a summary; reasoning tokens are the provider-reported output tokens and are counted as such.
The output budget of a think call is twice the model's `max_output_tokens`, capped by `think.max_output_tokens` (Groq: no doubling) and
by the prompt-fitting window rules (the prompt is fitted with that larger reservation). On Auto, think-capable models of the granted
group go first (stable order); an explicitly chosen model is respected; with none capable the answer is normal and `think` is false.
`catalog_check` reports `think_mismatch` when billing says `think: true` for a model whose kind is `none` here.

## Follow-ups
Settings: `FOLLOWUPS_ENABLED` (true), `FOLLOWUP_MODEL` (empty = `gemini-3.1-flash-lite` with a Google key, else the Groq request-analysis
model), `FOLLOWUP_TIMEOUT_MS` (4000), `FOLLOWUP_MAX_TOKENS` (300). Prompt: `app/prompts/followups/followups_v1.yaml`.
Runs only for a successful answer of at least 200 characters with `suggest_followups` on; metric `followups_total{outcome=published|skipped|timeout|invalid|error}`;
the call's cost goes out on `usage.metered` (feature `followups`).
