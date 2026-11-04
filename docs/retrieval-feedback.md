# Retrieval feedback: from a thumbs-down to a benchmark case

## What is logged
Every document retrieval writes one structured line, event `retrieval_feedback` (logger `context_collector`), with:

| Field | Meaning |
|---|---|
| `trace_id`, `conversation_id`, `message_id` | find the turn |
| `query_hash` | first 16 hex chars of SHA-256 of the user's message (query text is NOT logged) |
| `rewrite_method`, `rewrite_used`, `dense_query_sent` | `model` / `heuristic` / `none` / `disabled`; whether the query differed |
| `previous_chunks_sent` | how many previous-turn chunk ids were offered |
| `full_text` | the whole (small) document was sent instead of top-k |
| `status`, `coverage`, `top_score` | retrieval status and the weak-retrieval signal (`COVERAGE_UNSPECIFIED` from older servers) |
| `chunk_ids`, `scores` | the chunks sent to the model, in order |
| `latency_ms` | per stage: `rewrite_ms`, `search_ms`, `full_text_ms` |

Document content is never logged. With `LOG_QUERY_TEXT=true` (default false) the lines also carry `query_text`,
`retrieval_query_text` and `dense_query_text`; turn it on only briefly in a test environment.

## Turning a thumbs-down answer into a benchmark case
1. From the thumbs-down message take its `conversation_id` / `message_id` (or trace id) and find the `retrieval_feedback` line.
2. Read the question from the conversation (it is stored by the conversation service, not logged here). Check `query_hash`
   (sha256 of the message text, first 16 hex chars) matches, so you are looking at the right turn.
3. Classify the failure with the line: `coverage` WEAK/NONE or low `top_score` = retrieval missed; `chunk_ids` present
   and coverage STRONG but wrong answer = generation problem, not a retrieval case; `rewrite_method=heuristic` on a follow-up =
   rewrite problem (add to `docs/retrieval-rewrite-cases.md`).
4. Open the file the user attached and copy the passage that answers the question verbatim. That text is the gold evidence
   (the benchmark scores by evidence text, so it survives chunker changes).
5. Add a question to `retrieval-service/evaluation/questions.py` (tuning set) or `heldout.py` (held-out set; never tune against it):
   category, the question as typed (keep Hinglish / typos), the file(s), and the evidence string(s). If the document is
   private, write a small invented fixture with the same shape instead of copying it.
6. Run `make eval` there; the case should fail before the fix and pass after. Record the chunk ids from the log to see which
   chunks ranked above the answer.
