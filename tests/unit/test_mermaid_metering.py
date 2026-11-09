"""P5x: the browser-initiated Mermaid repair is metered with usage.metered 1.0 (billing plan 4.4.5)."""

import uuid
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import visualizations as viz_api
from app.config.settings import LLMServiceConfig
from app.producers.kafka_producer import KafkaPublisher
from app.providers.usage_hook import report_usage
from app.visualization.jobs import MemoryJobStore
from app.visualization.renderer import MediaStore

OWNER = str(uuid.uuid4())
GOOD = "```mermaid\nflowchart TD\nA-->B\n```"


class Chain:
    """Stands in for the provider chain: reports usage like a real adapter, then returns `reply`."""

    def __init__(self, reply=GOOD, fail=False):
        self.reply, self.fail = reply, fail

    async def generate_stream(self, prompt, params=None):
        report_usage(None, provider="groq", model="openai/gpt-oss-120b", input_tokens=420, output_tokens=60)
        if self.fail:
            raise RuntimeError("provider died after the call was billed")
        yield self.reply


class Publisher:
    def __init__(self, error=None):
        self.events, self.error = [], error

    async def publish_usage_metered(self, **kw):
        if self.error:
            raise self.error
        self.events.append(kw)


def client(tmp_path, chain, publisher):
    config = LLMServiceConfig(manim_enabled=True, mermaid_enabled=True, manim_media_dir=str(tmp_path / "media"))
    store = MemoryJobStore()
    store.inline = False
    container = SimpleNamespace(config=config, generation_router=chain, viz_store=store, viz_tasks=set(),
                                viz_media=MediaStore(config.manim_media_dir), kafka_producer=publisher, catalog_model=None)
    app = FastAPI()
    app.state.container = container
    app.include_router(viz_api.router, prefix="/api")
    return TestClient(app)


def repair(c, kind="mermaid", user=OWNER):
    body = {"kind": kind, "content": "flowchart TD\nA[x", "error": "syntax", "attempt": 2}
    return c.post("/api/v1/visualizations/repair", json=body, headers={"X-User-Id": user})


def test_a_mermaid_repair_publishes_one_usage_metered_event_with_the_call(tmp_path):
    pub = Publisher()
    r = repair(client(tmp_path, Chain(), pub))
    assert r.status_code == 200 and r.json()["ok"] is True
    (e,) = pub.events
    assert e["user_id"] == OWNER and e["feature"] == "mermaid.repair"
    assert e.get("conversation_id") is None and e.get("request_message_id") is None   # not tied to a message
    assert e["calls"] == [{"provider": "groq", "model": "openai/gpt-oss-120b", "purpose": "mermaid.repair",
                           "input_tokens": 420, "cached_input_tokens": 0, "output_tokens": 60, "usage_estimated": False}]


def test_a_repair_that_fails_validation_or_the_provider_is_still_metered(tmp_path):
    pub = Publisher()
    c = client(tmp_path, Chain(reply="flowchart TD\nA[<script>]"), pub)
    assert repair(c).json()["ok"] is False
    assert len(pub.events) == 1
    pub2 = Publisher()
    assert repair(client(tmp_path, Chain(fail=True), pub2)).json()["ok"] is False      # provider died after being billed
    assert len(pub2.events) == 1 and pub2.events[0]["calls"][0]["output_tokens"] == 60


def test_a_publish_failure_does_not_fail_the_repair(tmp_path):
    r = repair(client(tmp_path, Chain(), Publisher(error=ConnectionError("kafka down"))))
    assert r.status_code == 200 and r.json()["ok"] is True


def test_no_event_when_the_repair_is_refused_before_any_model_call(tmp_path):
    pub = Publisher()
    c = client(tmp_path, Chain(), pub)
    c.app.state.container.config = LLMServiceConfig(visualization_repair_enabled=False)
    assert repair(c).json() == {"ok": False, "failure": "disabled"}
    assert pub.events == []


def test_the_manim_repair_endpoint_is_metered_under_its_own_feature(tmp_path):
    pub = Publisher()
    spec = '{"template": "text_steps", "title": "T", "params": {"steps": ["a", "b"]}}'
    repair(client(tmp_path, Chain(reply=spec), pub), kind="manim")
    assert pub.events[0]["feature"] == "manim.repair" and pub.events[0]["calls"][0]["purpose"] == "manim.repair"


@pytest.mark.asyncio
async def test_the_publisher_builds_the_plan_4_4_5_event():
    import json
    from unittest.mock import AsyncMock, MagicMock

    sent = []

    class Producer:
        async def send_and_wait(self, topic, key, value, headers=None):
            sent.append((topic, key, json.loads(value)))

    pub = KafkaPublisher(config=LLMServiceConfig(), producer=Producer())
    pub._started = True

    async def once(fn, **_kw):
        return await fn()

    pub._retry_manager = MagicMock(execute_with_retry=once)
    await pub.publish_usage_metered(user_id=OWNER, feature="mermaid.repair", calls=[{"purpose": "mermaid.repair"}])
    (topic, key, ev), = sent
    assert topic == "usage.metered" and key == OWNER.encode()
    assert set(ev) == {"event_type", "schema_version", "event_id", "occurred_at", "producer", "user_id", "conversation_id",
                       "request_message_id", "feature", "calls", "fixed_cost_micro_inr"}
    assert (ev["event_type"], ev["schema_version"], ev["producer"], ev["fixed_cost_micro_inr"]) == (
        "usage.metered", "1.0", "llm-service", 0)
    assert ev["occurred_at"].endswith("Z") and AsyncMock
