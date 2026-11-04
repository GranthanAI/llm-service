# Retrieval rewrite cases (for the retrieval-service benchmark)

Conversations for which llm-service builds a standalone retrieval query (`app/context/retrieval_query.py`). `query` is sent
as `GetRelevantChunksRequest.query` (keyword leg), `dense_query` as the English rewrite (dense leg). Expected values are
what a good rewrite looks like (the model's exact wording varies; the heuristic fallback column is deterministic and is
what the unit tests assert). Corpus: `retrieval-service/evaluation/fixtures/`. Gold is evidence TEXT.

| # | Conversation (earlier turns, then the current message) | Expected `query` | Expected `dense_query` | Heuristic fallback `query` (no model) | Expected evidence (must be in a returned chunk) |
|---|---|---|---|---|---|
| 1 | U: what is phase 6 about? / A: (any) / **U: Check again** | check again what phase 6 is about | what is phase 6 about | `what is phase 6 about? Check again` | "In this phase the team builds webhook ingestion from the payment gateway" (PHASES.md, Phase 6: Payment gateway webhooks) |
| 2 | **U: Phase 6 mein kya karna hai?** (no history) | Phase 6 mein kya karna hai? | what does phase 6 involve | unchanged (message has content words); no `dense_query` | "Verify the webhook signature header on every call" (PHASES.md Phase 6 Tasks) |
| 3 | U: what is phase 6 about? / A: (any) / **U: explain point 3** | explain point 3 of phase 6 | explain task 3 of phase 6 | `what is phase 6 about? explain point 3` | "Map gateway statuses onto internal payment states" |
| 4 | U: What does the runbook say about retries? / A: (any) / **U: and the default?** | what is the default of MAX_RETRY_ATTEMPTS | what is the default value of MAX_RETRY_ATTEMPTS | `What does the runbook say about retries? and the default?` | "The default is 5 and the hard ceiling is 12" (runbook.md) |
| 5 | **U: TXN-102394 kya hai?** | TXN-102394 kya hai? | what is TXN-102394 | unchanged | "traced to transaction TXN-102394" (runbook.md) |
| 6 | U: what is phase 6 about? / A: (wrong guess) / **U: that is wrong, read the document again** | read phase 6 of the document again | what is phase 6 about | `what is phase 6 about? that is wrong, read the document again` | Phase 6 section as in case 1 |
| 7 | U: summarise PHASES.md / A: (any) / **U: ok** | ok summarise PHASES.md | summarise PHASES.md | `summarise PHASES.md ok` | any PHASES.md chunk (smoke only) |

Also useful for the benchmark: previous-turn chunks. Case 3 should be run with `previous_chunk_ids` set to the chunk ids
returned for case 1 and must still return the Phase 6 "Tasks" chunk when the rewrite is poor.
