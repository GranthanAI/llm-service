"""
Prometheus Metrics Definitions.
Implements LLD v2.0 Section 27.3.
"""

from prometheus_client import Counter, Gauge, Histogram

# Request metrics
REQUESTS_TOTAL = Counter(
    "llm_requests_total",
    "Total LLM service requests",
    ["mode", "skill", "engine_type", "provider", "status"],
)

REQUEST_DURATION = Histogram(
    "llm_request_duration_seconds",
    "End-to-end request duration",
    ["mode", "skill", "engine_type", "provider"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)

ERRORS_TOTAL = Counter(
    "llm_errors_total",
    "Total errors classified and handled",
    ["error_type", "stage"],
)

# Workflow Engine dispatch metrics
WORKFLOW_DISPATCH_TOTAL = Counter(
    "llm_workflow_engine_dispatch_total",
    "Workflow Engine dispatch count by mode and engine type",
    ["mode", "engine_type"],
)

LANGGRAPH_NODE_DURATION = Histogram(
    "llm_langgraph_node_duration_seconds",
    "Per-node duration inside SmartGraph / DeepResearchGraph",
    ["graph", "node"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
)

LANGGRAPH_LOOP_ITERATIONS = Histogram(
    "llm_langgraph_loop_iterations",
    "Loop iterations consumed per graph invocation",
    ["graph"],
    buckets=[1, 2, 3, 4, 5, 6, 8, 10],
)

LANGGRAPH_LOOP_CAPPED = Counter(
    "llm_langgraph_loop_capped_total",
    "Count of graph invocations that hit the iteration cap and were forced to proceed",
    ["graph"],
)

TTFT = Histogram(
    "llm_ttft_seconds",
    "Time to first token",
    ["provider", "mode", "engine_type"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
)

TOKENS_TOTAL = Counter(
    "llm_tokens_total",
    "Total tokens processed",
    ["provider", "token_type"],  # token_type: prompt | completion
)

STREAMING_CHUNKS_TOTAL = Counter(
    "llm_streaming_chunks_total",
    "Total streaming token chunks published to Kafka",
    ["provider", "mode"],
)

COST_USD = Counter(
    "llm_cost_usd_total",
    "Total LLM cost in USD",
    ["provider", "mode", "user_tier"],
)

CIRCUIT_BREAKER_STATE = Gauge(
    "llm_circuit_breaker_state",
    "Circuit breaker state (0=closed, 1=half-open, 2=open)",
    ["name"],
)

GENERATION_FALLBACK_TOTAL = Counter(
    "llm_generation_fallback_total",
    "Generation Router fallback events",
    ["from_provider", "to_provider"],
)

TOOL_CALLS = Counter(
    "llm_tool_calls_total",
    "Tool execution count",
    ["tool_name", "status"],
)

TOOL_DURATION = Histogram(
    "llm_tool_duration_seconds",
    "Tool execution duration",
    ["tool_name"],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
)

KAFKA_LAG = Gauge(
    "llm_kafka_consumer_lag",
    "Kafka consumer lag",
    ["topic", "partition"],
)

CONTEXT_FETCH_DURATION = Histogram(
    "llm_context_fetch_duration_seconds",
    "Baseline context provider fetch duration",
    ["source"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
)

CONTEXT_DEGRADED_TOTAL = Counter(
    "llm_context_degraded_total",
    "Count of requests where a baseline context source degraded",
    ["source", "reason"],
)

CONTEXT_TOKENS = Histogram(
    "llm_context_tokens_by_section",
    "Token count per prompt section",
    ["section", "mode"],
)

PROMPT_BUILD_DURATION = Histogram(
    "llm_prompt_build_duration_seconds",
    "Duration of prompt building and section composition",
    ["mode"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25],
)
