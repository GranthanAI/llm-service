# Task Update 14: NVIDIA Model Swap, Groq Key/Model Fixes, and Request Analyzer NVIDIA Fallback

This update documents the changes made to LLM Service while diagnosing and fixing a full-stack live test that surfaced two dead Groq API keys and one badly-suited Groq model for structured analysis output — plus a new NVIDIA fallback added to Request Analysis at the user's explicit request.

---

## 1. Executive Summary

`deepseek-ai/deepseek-v4-pro-0813` was requested for evaluation as a possible NVIDIA generation model — confirmed via direct testing that it's listed in NVIDIA's model catalog but its inference endpoint never actually responds (hung indefinitely even on a 10-token request with a 25s timeout), while `nvidia/nemotron-3.5-lightning-30b-a3b` responded correctly and quickly. The service was switched to the working model everywhere NVIDIA's model is configured.

Separately, live end-to-end testing surfaced that **both** this service's and Memory Service's Groq API keys had gone invalid (`401 Invalid API Key`) since earlier sessions, and that the configured Groq model (`groq/compound`, an agentic/tool-router model) was returning content that failed JSON parsing for the Request Analyzer's real (larger) system prompt — forcing every single request through a 30+ second NVIDIA fallback instead of Groq's normal ~1s analysis latency.

Finally, per explicit request, added an NVIDIA fallback to Request Analysis itself: if Groq's analysis call fails, the same analysis is retried via NVIDIA before falling back to the heuristic `SafeDefaultPlan`. This is analysis-only — it does not change the Generation Router's provider order for actual chat replies.

---

## 2. NVIDIA Model Switch

Every place NVIDIA's generation model is configured was updated from the now-end-of-life `meta/llama-3.3-70b-instruct` to `nvidia/nemotron-3.5-lightning-30b-a3b`:
- `.env`, `.env.example` (`NVIDIA_MODEL`)
- `app/config/settings.py` (`nvidia_model` default)
- `app/providers/nvidia.py` (`NVIDIAAdapter.__init__`'s `model` default parameter — dead code in practice since `main.py` always passes `config.nvidia_model` explicitly, but kept consistent)
- `ENABLE_NVIDIA` flipped back to `true` in `.env` (was `false` since the old model's end-of-life)

Verified live: a real message sent through the full pipeline produced `"target_model": "nvidia/nemotron-3.5-lightning-30b-a3b"` and a successful `200 OK` from `integrate.api.nvidia.com`, with a correct generated reply.

---

## 3. Groq API Key and Model Fixes

### 3.1 Dead API Key
`GROQ_API_KEY` in `.env` had gone invalid (confirmed directly against Groq's API: `401 Invalid API Key`). Replaced with a new working key.

### 3.2 Wrong Model for Structured Analysis
`GROQ_MODEL` was set to `groq/compound` — an agentic, tool-augmented router model, not a plain structured-completion model. For simple prompts it correctly returns clean JSON in `content`, but for the Request Analyzer's actual (larger) system prompt it sometimes returned whitespace/commentary instead, causing `json.loads()` to fail with `Expecting value: line 1 column 1 (char 0)` on every single request — silently forcing every analysis call through the (much slower) NVIDIA fallback path added in §4.

Switched to `openai/gpt-oss-20b`, which correctly separates its `reasoning` field from `content` and reliably returns pure JSON. Updated in four places for consistency: `.env`, `.env.example`, `app/config/settings.py` (`groq_model` default), `app/request_analyzer/groq_client.py` (`GroqAnalysisClient.__init__`'s default parameter).

**Effect**: Request Analysis latency dropped from ~30-45s (NVIDIA fallback engaged on every turn) to ~1s (direct Groq success), confirmed via live logs (`"provider": "groq", "latency_ms": 1093.96`).

---

## 4. Request Analyzer: NVIDIA Fallback

Per explicit request: `RequestAnalyzer.analyze()` (`app/request_analyzer/analyzer.py`) now tries Groq first as before; if that raises, and an `nvidia_adapter` is configured, it retries the *same* analysis prompt via `NVIDIAAdapter.execute()` before falling back to `SafeDefaultFactory.create()`. Groq's role in Request Analysis and the Generation Router's separate provider order (NVIDIA primary / Gemini fallback for actual chat replies) are unaffected — this fallback exists only inside the analysis step.

Implementation notes:
- `RequestAnalyzer.__init__` gained an optional `nvidia_adapter: NVIDIAAdapter | None` parameter.
- `analyze()` was refactored into `_call_groq()` (unchanged circuit-breaker + retry logic, extracted) and `_finalize_plan()` (shared JSON-parse + validation + safety-override logic, reused by both the Groq and NVIDIA paths).
- Added `_extract_json_object()`: NVIDIA's reasoning-capable models can prepend chain-of-thought text before the actual JSON object (observed directly during testing — a real limerick-generation test streamed a full "Here's a thinking process..." preamble before content). Rather than assume the whole response is valid JSON, this extracts the substring from the first `{` to the last `}` before parsing.
- `app/main.py`: `nvidia_adapter` construction was moved earlier in the lifespan (previously built at Phase 15, alongside Gemini/Groq generation adapters) so it can be injected into `RequestAnalyzer` at Phase 6/7, and the later duplicate construction was removed — the same instance is reused for both the analysis fallback and the Generation Router.

Verified live: with the Groq model still broken (before the §3.2 fix), a real request correctly fell through to `"event": "Request analysis completed via NVIDIA fallback"` and produced a valid `ExecutionPlan`. Also covered by 3 new unit tests in `tests/unit/test_request_analyzer.py`:
- `test_request_analyzer_nvidia_fallback_on_groq_outage` — NVIDIA fallback succeeds when Groq fails.
- `test_request_analyzer_nvidia_fallback_strips_reasoning_text` — reasoning-wrapped JSON is still parsed correctly.
- `test_request_analyzer_falls_back_to_safe_default_when_nvidia_also_fails` — both providers failing still yields a valid `SafeDefaultPlan`, never raises.

---

## 5. Test Suite Status

```bash
uv run pytest tests -q
```
**241 / 241 passed** (238 pre-existing + 3 new NVIDIA-fallback tests), no regressions.

---

## 6. What Was NOT Changed

- Generation Router's provider selection logic and order — untouched.
- No changes to Memory Service, Conversation Service, or any other service's own code (their `.env` Groq key/model fixes are documented in their own task-update logs).
- `deepseek-ai/deepseek-v4-pro-0813` was not adopted anywhere — confirmed non-functional on NVIDIA's serving infrastructure at test time, not a configuration issue on this service's side.
