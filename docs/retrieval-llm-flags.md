# llm-service retrieval flags (settings class: `app/config/settings.py`)

| Env var | Default | Effect |
|---|---|---|
| `RETRIEVAL_TOP_K` | 8 | chunks requested per question (was hard-coded 5; decision recorded in retrieval-service docs) |
| `RETRIEVAL_MAX_TOKENS` | 4000 | token cap sent to retrieval for those chunks |
| `RETRIEVAL_QUERY_REWRITE_ENABLED` | true | standalone + English rewrite of the question (fast models, heuristic fallback) |
| `RETRIEVAL_QUERY_REWRITE_TIMEOUT_S` | 2.0 | per-model timeout of the rewrite call |
| `RETRIEVAL_QUERY_REWRITE_TURNS` | 3 | earlier user/assistant exchanges given to the rewriter (0 = none) |
| `RETRIEVAL_PREVIOUS_CHUNKS_ENABLED` | true | send last turn's chunk ids as `previous_chunk_ids` (process-local store, in memory) |
| `RETRIEVAL_PREVIOUS_CHUNKS_TTL_S` | 3600 | how long a conversation's previous chunk ids are kept |
| `RETRIEVAL_FULL_TEXT_ENABLED` | **false** | small attachments sent in full via `GetDocumentChunks`; enable only after retrieval-service implements it |
| `RETRIEVAL_FULL_TEXT_SHARE` | 0.25 | share of the input budget a full-text attachment may use; larger -> normal retrieval |
| `RETRIEVAL_INPUT_BUDGET_TOKENS` | 0 | input budget used for that share; 0 = min(`PROMPT_TOKEN_BUDGET` if set, `CONTEXT_BUDGET_DEFAULT_TOKENS`) |
| `RETRIEVAL_COVERAGE_PROMPT_ENABLED` | true | WEAK/NONE coverage rendered in the prompt (no change for UNSPECIFIED/STRONG) |
| `LOG_QUERY_TEXT` | false | include query text in the `retrieval_feedback` log line |

Every feature degrades to the previous behaviour when the retrieval server ignores the new fields: no `coverage` signal means
no prompt change, UNIMPLEMENTED `GetDocumentChunks` means normal top-k retrieval.
