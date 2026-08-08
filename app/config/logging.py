"""
Structured logging configuration using Structlog.
Implements LLD v2.0 Section 27.1.
"""

import contextvars
import logging
import sys
from typing import Any

import structlog

# Context variables for correlation IDs across async calls
correlation_request_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default=""
)
correlation_conversation_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "conversation_id", default=""
)
correlation_trace_id: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="")
correlation_span_id: contextvars.ContextVar[str] = contextvars.ContextVar("span_id", default="")
correlation_engine_type: contextvars.ContextVar[str] = contextvars.ContextVar(
    "engine_type", default=""
)


def add_correlation_ids(
    logger: Any, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Structlog processor adding contextvar correlation IDs to all log entries."""
    if req_id := correlation_request_id.get():
        event_dict["request_id"] = req_id
    if conv_id := correlation_conversation_id.get():
        event_dict["conversation_id"] = conv_id
    if trace_id := correlation_trace_id.get():
        event_dict["trace_id"] = trace_id
    if span_id := correlation_span_id.get():
        event_dict["span_id"] = span_id
    if engine := correlation_engine_type.get():
        event_dict["engine_type"] = engine
    return event_dict


def setup_logging(
    log_level: str = "INFO", service_name: str = "llm-service", version: str = "2.0.0"
) -> None:
    """Configures stdlib and Structlog JSON processor chain."""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=numeric_level)

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        add_correlation_ids,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    structlog.configure(
        processors=shared_processors
        + [
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Returns a bound structlog logger."""
    return structlog.get_logger(name)  # type: ignore[return-value]
