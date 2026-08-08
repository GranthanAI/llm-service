"""
OpenTelemetry Tracing Setup and Decorators.
Implements LLD v2.0 Section 27.2.
"""

import functools
import os
from collections.abc import Callable
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.trace import Span, Status, StatusCode

from app.config.logging import (
    correlation_span_id,
    correlation_trace_id,
)

_TRACER = None


def setup_tracing(service_name: str = "llm-service", endpoint: str | None = None) -> trace.Tracer:
    """Configures OpenTelemetry TracerProvider and returns default tracer."""
    global _TRACER
    resource = Resource.create({"service.name": service_name, "service.version": "2.0.0"})
    provider = TracerProvider(resource=resource)

    # In local/test environments or if no otel endpoint, fallback to console or noop
    if os.getenv("ENVIRONMENT") == "development" or not endpoint:
        processor = BatchSpanProcessor(ConsoleSpanExporter())
    else:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

            exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
            processor = BatchSpanProcessor(exporter)
        except Exception:
            processor = BatchSpanProcessor(ConsoleSpanExporter())

    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
    _TRACER = trace.get_tracer(service_name)
    return _TRACER


def get_tracer(name: str = "llm-service") -> trace.Tracer:
    """Returns initialized or default tracer."""
    global _TRACER
    if _TRACER is None:
        _TRACER = trace.get_tracer(name)
    return _TRACER


def trace_span(span_name: str, attributes: dict[str, Any] | None = None) -> Callable:
    """
    Decorator that starts an OTel child span for an async or sync function.
    Records exceptions and updates contextvars with trace_id and span_id.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = get_tracer()
            with tracer.start_as_current_span(span_name) as span:
                _set_span_context(span, attributes)
                try:
                    return await func(*args, **kwargs)
                except Exception as exc:
                    span.record_exception(exc)
                    span.set_status(Status(StatusCode.ERROR, str(exc)))
                    raise

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = get_tracer()
            with tracer.start_as_current_span(span_name) as span:
                _set_span_context(span, attributes)
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    span.record_exception(exc)
                    span.set_status(Status(StatusCode.ERROR, str(exc)))
                    raise

        import inspect

        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def _set_span_context(span: Span, attributes: dict[str, Any] | None = None) -> None:
    ctx = span.get_span_context()
    if ctx and ctx.trace_id:
        correlation_trace_id.set(format(ctx.trace_id, "032x"))
    if ctx and ctx.span_id:
        correlation_span_id.set(format(ctx.span_id, "016x"))
    if attributes:
        for k, v in attributes.items():
            if v is not None:
                span.set_attribute(k, v)
