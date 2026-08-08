"""
Unit tests for health, readiness, and metrics endpoints.
"""

from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient, None, None]:
    monkeypatch.setenv("GROQ_API_KEY", "test_groq")
    monkeypatch.setenv("NVIDIA_API_KEY", "test_nvidia")
    monkeypatch.setenv("GEMINI_API_KEY", "test_gemini")

    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "llm-service"
    assert data["version"] == "2.0.0"


def test_healthz_alias(client: TestClient):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_readiness_endpoint(client: TestClient):
    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["ready"] is True


def test_readiness_failure():
    app = create_app()
    with TestClient(app) as client:
        # Simulate unready state after boot
        app.state.container.mark_ready(False)
        response = client.get("/ready")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "not_ready"
        assert data["ready"] is False


def test_health_failure():
    app = create_app()
    with TestClient(app) as client:
        # Simulate unhealthy state
        app.state.container.mark_healthy(False)
        response = client.get("/health")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "unhealthy"


def test_metrics_endpoint(client: TestClient):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "llm_requests_total" in response.text
    assert "llm_request_duration_seconds" in response.text


def test_root_endpoint(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "online"
