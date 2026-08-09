# Task Update 4: Generic Tool Framework & Multi-Source Open Web Search Engine

**Service**: GraphGPT LLM Service (`llm-service`)  
**Architecture Spec**: HLD v2.0 (Sections 14, 15) & LLD v2.0 (Sections 15, 16)  
**Phases Covered**: Phase 9 (Tool Framework) & Phase 10 (Multi-Source Open Web Search Tool)  
**Status**: Completed, Fully Tested, and Verified  

---

## 1. Executive Summary

Task Update 4 documents the end-to-end implementation and production verification of **Phase 9 (Generic Tool Framework)** and **Phase 10 (Multi-Source Open Web Search Tool)**.

The LLM Service now features a fully decoupled, asynchronous, schema-driven Generic Tool Framework adhering to strict SOLID and Dependency Injection principles. To eliminate all reliance on paid third-party search APIs, a resilient, multi-source search architecture was developed that queries **7 free public knowledge and search providers** concurrently in parallel:
1. **DuckDuckGo** (Instant Answers & Web Topic Search)
2. **Wikipedia** (MediaWiki REST API with HTML unescaping)
3. **SearXNG** (Open Metasearch Engine API)
4. **Wikidata** (Structured Entity & Concept Knowledge Graph API)
5. **ArXiv** (Scholarly & Academic Paper Repository via Atom XML Feed)
6. **OpenAlex** (Global Open Catalog of Scholarly Articles & Citations)
7. **Stack Exchange / Stack Overflow** (Technical Programming Q&A API)

All 74 automated unit and integration tests across the service are passing, and architectural import boundaries have been strictly verified with 0 AST violations.

---

## 2. Phase 9: Generic Tool Framework Architecture

The tool subsystem in `app/tools/` provides a standardized interface for tool registration, parameter validation, timeout-bounded execution, output normalization, and parallel dispatching.

### 2.1 Core Architectural Components

- **`BaseTool` (`app/tools/base.py`)**:
  - Abstract base class defining the standard contract:
    - `execute(params: ToolParams) -> ToolResult`: Asynchronous execution routine.
    - `validate_params(params: ToolParams) -> ValidationResult`: Static parameter pre-validation.
    - `get_schema() -> ToolSchema`: Self-describing JSON Schema generator for LLM prompts.
    - `is_available() -> bool`: Dynamic availability probe.
  - Zero dependencies on `context/`, `grpc/clients/`, or `workflow_engine/`.

- **`ToolValidator` (`app/tools/validator.py`)**:
  - Validates parameter dictionaries against required keys and Python type signatures before tool execution.

- **`ToolNormalizer` (`app/tools/normalizer.py`)**:
  - Normalizes tool outputs into uniform JSON payloads consumed deterministically by the Prompt Composition Engine.

- **`ToolExecutor` (`app/tools/executor.py`)**:
  - Wraps every tool execution in `asyncio.wait_for()` with configurable per-tool timeout limits.
  - Measures execution latency in milliseconds and maps unexpected exceptions and timeouts into structured error results without crashing the pipeline.

- **`ToolRegistry` (`app/tools/registry.py`)**:
  - Centralized dependency-injected tool catalog.
  - Supports dynamic tool registration, runtime `enable`/`disable` flags, and schema enumeration (`list_tools()`).
  - Raises typed `UnknownToolError` when disabled or unregistered tools are requested.

- **`ToolDispatcher` (`app/tools/dispatcher.py`)**:
  - Implements the Parallel Tool Execution Algorithm (LLD2 Section 15.4).
  - Partitions tool calls into parallel (`parallel=True`) and sequential (`parallel=False`) batches.
  - Concurrently gathers parallel calls via `asyncio.gather(*tasks, return_exceptions=True)`.
  - Executes sequential tool calls in strict deterministic order.
  - Enforces `RequiredToolFailedError` when any tool marked with `required=True` fails or times out.

---

## 3. Phase 10: Multi-Source Open Web Search Engine

To achieve zero-cost, highly resilient search capabilities across diverse domains (web topics, general facts, scholarly papers, and programming code), `WebSearchTool` was re-engineered to aggregate 7 free public search engines simultaneously.

```
                    ┌─────────────────────────────────────────┐
                    │       WebSearchTool.execute(params)     │
                    └────────────────────┬────────────────────┘
                                         │
                         ┌───────────────┴───────────────┐
                         ▼                               ▼
                 [ 60s TTL Cache Hit? ]          [ Perform Search ]
                 ├── Yes ──> Return Cached       │
                 └── No  ──> Parallel Gather ────┘
                                 │
         ┌──────────────┬────────┼────────┬──────────────┬──────────────┬──────────────┐
         ▼              ▼        ▼        ▼              ▼              ▼              ▼
    DuckDuckGo      Wikipedia SearXNG  Wikidata        ArXiv        OpenAlex    StackExchange
    (Web Topics)   (Wiki Pages) (Meta) (Entities)    (Papers XML)  (Citations)   (Code & Q&A)
         │              │        │        │              │              │              │
         └──────────────┴────────┼────────┴──────────────┴──────────────┴──────────────┘
                                 ▼
                     [ Resilient Aggregation ]
                     (Partial success is accepted)
                                 ▼
                     [ Domain Deduplication ]
                     (Max 1 hit per domain)
                                 ▼
                     [ Store in 60s TTL Cache ]
                                 ▼
                     [ Normalized ToolResult ]
```

### 3.1 Individual Provider Implementations (`app/tools/search_providers/`)

1. **`DuckDuckGoProvider` (`duckduckgo.py`)**:
   - Queries the DuckDuckGo Instant Answer API for summaries and extracts structured related topics and links.

2. **`WikipediaProvider` (`wikipedia.py`)**:
   - Queries the MediaWiki Search API, sanitizes HTML formatting tags, unescapes entities, and formats direct Wikipedia article URLs.

3. **`SearXNGProvider` (`searxng.py`)**:
   - Dispatches search queries to open SearXNG metasearch instances, parsing title, URL, and snippet payloads.

4. **`WikidataProvider` (`wikidata.py`)**:
   - Queries Wikidata's `wbsearchentities` API for structured ontology concepts, entity IDs (e.g. `Q104868516`), and descriptions.

5. **`ArxivProvider` (`arxiv.py`)**:
   - Fetches scholarly pre-prints and academic papers directly from the ArXiv Export API by parsing Atom XML feeds (`atom:entry`, `atom:title`, `atom:summary`).

6. **`OpenAlexProvider` (`openalex.py`)**:
   - Queries OpenAlex open scholarly catalog for publication titles, DOI URLs, publication years, venue names, and citation counts.

7. **`StackExchangeProvider` (`stackexchange.py`)**:
   - Searches Stack Overflow via Stack Exchange API 2.3 for relevant programming questions, accepted answers, scores, and tags.

### 3.2 Resilient Multi-Source Aggregation & Deduplication

- **No Single Point of Failure**: Search queries are gathered via `asyncio.gather(*tasks, return_exceptions=True)`. If one or more external providers are slow or unreachable, the tool gracefully aggregates responses from all available providers.
- **Per-Domain Deduplication**: Deduplicates identical URLs and retains only the top-ranked result per domain name to maximize information diversity.
- **60-Second TTL Caching**: Employs an in-memory `TTLCache` to prevent redundant network requests when iterative agentic loops re-query similar topics.

---

## 4. Architectural Boundaries & Clean Architecture

The implementation strictly honors the dependency boundaries outlined in LLD v2.0:
- `app/tools/` does **not** import `app/context/`, `app/grpc/clients/`, or `app/prompts/`.
- `app/workflow_engine/mode_handlers/` consumes `ToolDispatcher` exclusively through dependency injection.
- Container initialization in `app/main.py` serves as the sole Composition Root, registering the tool suite into the DI Container at startup.

---

## 5. Test Suite & Verification Results

All unit and integration tests across the repository pass cleanly:

```bash
uv run pytest tests -v
```

### Summary of Passing Test Suites:
- `tests/unit/test_web_search_multi_provider.py`: 8/8 tests passing (DuckDuckGo, Wikipedia, SearXNG, Wikidata, ArXiv, OpenAlex, StackExchange, and multi-provider partial failure tolerance).
- `tests/unit/test_tools.py`: 6/6 tests passing (Validator, Registry, Executor, Dispatcher, Normalizer, and WebSearch caching).
- `tests/unit/test_workflow_engine.py`: 7/7 tests passing.
- `tests/unit/test_mode_handlers.py`: 5/5 tests passing.
- `tests/unit/test_request_analyzer.py`: 8/8 tests passing.
- `tests/unit/test_models.py`: 8/8 tests passing.
- `tests/unit/test_context.py`: 4/4 tests passing.
- `tests/unit/test_config.py`: 4/4 tests passing.
- `tests/unit/test_health.py`: 7/7 tests passing.
- `tests/grpc/test_grpc_clients.py`: 7/7 tests passing.
- `tests/kafka/test_kafka_infrastructure.py`: 8/8 tests passing.

**Total passing tests**: **74 / 74 passed** (100% success rate).  
**Boundary Checks**: `python scripts/check_import_boundaries.py` -> **0 violations**.

---

## 6. Next Steps

With Phase 9 (Tool Framework) and Phase 10 (Web Search Tool) complete, the project is ready for **Phase 11: Smart Graph (LangGraph Agentic Orchestration)**:
- `SmartGraphState` state schema with reducers.
- `SmartGraphBuilder` constructing agent nodes (`analyze`, `tool_execution`, `synthesize`).
- Conditional routing edges and loop guard iteration limits.
