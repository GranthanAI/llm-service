"""C5: with OTEL_ENABLED=false (or no endpoint) no span exporter is created, so nothing retries a missing collector."""

from opentelemetry import trace

from app.utils import tracing


def processors(provider):
    return list(provider._active_span_processor._span_processors)


def test_disabled_creates_no_exporter(monkeypatch):
    created = []
    monkeypatch.setattr(tracing, "BatchSpanProcessor", lambda e: created.append(e) or object())
    tracing.setup_tracing("t", "http://127.0.0.1:4317", enabled=False)
    assert created == []


def test_no_endpoint_creates_no_exporter(monkeypatch):
    created = []
    monkeypatch.setattr(tracing, "BatchSpanProcessor", lambda e: created.append(e) or object())
    tracing.setup_tracing("t", "", enabled=True)
    assert created == []


def test_enabled_with_an_endpoint_still_builds_an_exporter(monkeypatch):
    created = []
    monkeypatch.setattr(tracing, "BatchSpanProcessor", lambda e: created.append(e) or __import__("opentelemetry.sdk.trace.export", fromlist=["x"]).SimpleSpanProcessor(e))
    monkeypatch.setenv("ENVIRONMENT", "production")
    tracing.setup_tracing("t", "http://127.0.0.1:4317", enabled=True)
    assert len(created) == 1
