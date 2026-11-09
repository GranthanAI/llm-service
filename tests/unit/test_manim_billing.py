"""P5e: Manim renders reserved with billing (fake gRPC server, no real billing) and artifact.manim.rendered."""

import json
import time
import uuid
from concurrent import futures
from types import SimpleNamespace

import grpc
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import visualizations as viz_api
from app.config.settings import LLMServiceConfig
from app.grpc.clients.billing_client import BillingClient
from app.grpc.proto import billing_pb2, billing_pb2_grpc
from app.providers.usage_hook import report_usage
from app.visualization.billing import ManimBilling
from app.visualization.jobs import JobState, MemoryJobStore, job_id_for
from app.visualization.manim_templates import validate_spec
from app.visualization.policy import FailureCategory
from app.visualization.renderer import MediaStore, RenderResult
from app.visualization.runner import run_job

SPEC = {"template": "text_steps", "title": "T", "params": {"steps": ["a", "b"]}}
OWNER = str(uuid.uuid4())
H = {"X-User-Id": OWNER}


class FakeBilling(billing_pb2_grpc.BillingInternalServicer):
    """behaviour: allow | deny:<DENY_REASON_*> | slow | error"""

    def __init__(self):
        self.behaviour = "allow"
        self.reserves, self.releases, self.metadata = [], [], []
        self.n = 0

    def CheckAndReserve(self, request, context):
        self.reserves.append((request.user_id, request.idempotency_key, request.kind))
        self.metadata.append(dict(context.invocation_metadata()))
        if self.behaviour == "slow":
            time.sleep(0.5)
        if self.behaviour == "error":
            context.abort(grpc.StatusCode.UNAVAILABLE, "down")
        self.n += 1
        if self.behaviour.startswith("deny:"):
            return billing_pb2.CheckAndReserveResponse(
                allowed=False, deny_reason=getattr(billing_pb2, self.behaviour.split(":", 1)[1]), message_key="limit.x")
        return billing_pb2.CheckAndReserveResponse(allowed=True, hold_id=f"hold-{self.n}")

    def Release(self, request, context):
        self.releases.append((request.user_id, request.hold_id, request.reason))
        return billing_pb2.ReleaseResponse(released=True)


@pytest.fixture
def billing():
    servicer = FakeBilling()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    billing_pb2_grpc.add_BillingInternalServicer_to_server(servicer, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    servicer.target = f"127.0.0.1:{port}"
    yield servicer
    server.stop(0)


class FakePublisher:
    def __init__(self):
        self.events = []

    async def publish_manim_rendered(self, **kw):
        self.events.append(kw)


class RepairChain:
    """Stands in for the provider chain: reports usage like a real adapter and returns a corrected spec."""

    async def generate_stream(self, prompt, params=None):
        report_usage(None, provider="groq", model="openai/gpt-oss-120b", input_tokens=700, output_tokens=90)
        yield json.dumps(SPEC)


def make_app(tmp_path, billing, mode="enforce"):
    config = LLMServiceConfig(manim_enabled=True, manim_media_dir=str(tmp_path / "media"), manim_max_active_jobs_per_user=20)
    publisher = FakePublisher()
    client = BillingClient(billing.target, "svc-token") if mode != "off" else None
    store = MemoryJobStore()
    store.inline = False                                   # queue the job; the test runs it
    container = SimpleNamespace(config=config, generation_router=RepairChain(), viz_store=store,
                                viz_media=MediaStore(config.manim_media_dir), viz_tasks=set(),
                                manim_billing=ManimBilling(mode, client, publisher), catalog_model=None)
    app = FastAPI()
    app.state.container = container
    app.include_router(viz_api.router, prefix="/api")
    app.include_router(viz_api.internal_router, prefix="/api")
    return app, container, publisher


def fake_render(tmp_path, results):
    calls = []

    async def render(spec, **kw):
        calls.append(spec.canonical_json())
        r = results[min(len(calls) - 1, len(results) - 1)]
        if r is None:
            f = tmp_path / f"o{len(calls)}.mp4"
            f.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 2000)
            return RenderResult(True, path=f)
        return RenderResult(False, r)

    return render, calls


async def run(container, tmp_path, job_id, results, **kw):
    render, calls = fake_render(tmp_path, results)
    store = container.viz_store
    rec = await run_job(store, container.viz_media, job_id, repair=viz_api._inline_repair(container),
                        final=viz_api._final_hook(container, store), render=render, **kw)
    return rec, calls


def submit(client):
    return client.post("/api/v1/visualizations/manim/jobs", json={"spec": SPEC}, headers=H)


def jid():
    return job_id_for(OWNER, validate_spec(SPEC).canonical_json())


# --- reserve -----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("reason,status,key", [
    ("DENY_REASON_DAILY_LIMIT", 429, "daily_limit"), ("DENY_REASON_MONTHLY_LIMIT", 429, "monthly_limit"),
    ("DENY_REASON_MODEL_NOT_IN_PLAN", 403, "model_not_in_plan"), ("DENY_REASON_ACCOUNT_BLOCKED", 403, "account_blocked"),
    ("DENY_REASON_NO_CREDIT", 402, "no_credit"), ("DENY_REASON_SPEND_CAP", 402, "spend_cap"),
    ("DENY_REASON_MEMBER_SPEND_CAP", 402, "member_spend_cap"),
])
def test_enforced_deny_returns_the_plan_4_6_shape_and_creates_no_job(tmp_path, billing, reason, status, key):
    billing.behaviour = f"deny:{reason}"
    app, container, publisher = make_app(tmp_path, billing)
    with TestClient(app) as client:
        r = submit(client)
    assert r.status_code == status
    detail = r.json()["detail"]
    assert detail["code"] == "billing_denied" and detail["reason"] == key and detail["message_key"] == "limit.x" and detail["message"]
    assert container.viz_store.jobs == {} and container.viz_store.queue == [] and publisher.events == []


def test_allowed_reserve_stores_hold_and_attempt_and_sends_the_right_key_and_metadata(tmp_path, billing):
    app, container, _ = make_app(tmp_path, billing)
    with TestClient(app) as client:
        assert submit(client).status_code == 202
    rec = container.viz_store.jobs[jid()]
    assert (rec.hold_id, rec.billing_attempt) == ("hold-1", 1) and container.viz_store.queue == [jid()]
    assert billing.reserves == [(OWNER, f"{jid()}:1", billing_pb2.RESERVE_KIND_MANIM_VIDEO)]
    md = billing.metadata[0]
    assert md["authorization"] == "Bearer svc-token" and md["x-correlation-id"] == jid()


def test_resubmitting_an_active_job_does_not_reserve_again(tmp_path, billing):
    app, _, _ = make_app(tmp_path, billing)
    with TestClient(app) as client:
        submit(client)
        submit(client)
    assert len(billing.reserves) == 1


def test_off_makes_no_call_and_shadow_never_blocks(tmp_path, billing):
    app, container, _ = make_app(tmp_path, billing, mode="off")
    with TestClient(app) as client:
        assert submit(client).status_code == 202
    assert billing.reserves == [] and container.viz_store.jobs[jid()].hold_id is None

    billing.behaviour = "deny:DENY_REASON_DAILY_LIMIT"
    app, container, _ = make_app(tmp_path, billing, mode="shadow")
    with TestClient(app) as client:
        assert submit(client).status_code == 202
    assert len(billing.reserves) == 1 and container.viz_store.jobs[jid()].hold_id is None   # asked, never blocked


def test_unreachable_billing_lets_the_render_proceed_with_a_null_hold_and_the_breaker_opens(tmp_path, billing):
    billing.behaviour = "error"
    app, container, _ = make_app(tmp_path, billing)
    breaker = container.manim_billing.client.breaker
    with TestClient(app) as client:
        for i in range(5):                                  # 5 consecutive failures open the breaker
            spec = {"template": "text_steps", "title": f"T{i}", "params": {"steps": ["a", str(i)]}}
            r = client.post("/api/v1/visualizations/manim/jobs", json={"spec": spec}, headers=H)
            assert r.status_code == 202
        assert breaker.is_open()
        calls_before = len(billing.reserves)
        spec = {"template": "text_steps", "title": "after", "params": {"steps": ["x", "y"]}}
        assert client.post("/api/v1/visualizations/manim/jobs", json={"spec": spec}, headers=H).status_code == 202
        assert len(billing.reserves) == calls_before         # breaker open: not even called
    assert all(r.hold_id is None for r in container.viz_store.jobs.values()) and len(container.viz_store.jobs) == 6


def test_a_slow_billing_times_out_at_150ms_and_the_render_proceeds(tmp_path, billing):
    billing.behaviour = "slow"
    app, container, _ = make_app(tmp_path, billing)
    with TestClient(app) as client:
        started = time.perf_counter()
        assert submit(client).status_code == 202
        assert time.perf_counter() - started < 0.45
    assert container.viz_store.jobs[jid()].hold_id is None


# --- the event and the hold at the end of a job ----------------------------------------------------------------------

async def test_one_rendered_event_per_attempt_and_a_retry_uses_attempt_2(tmp_path, billing):
    app, container, publisher = make_app(tmp_path, billing)
    with TestClient(app) as client:
        submit(client)
        rec, _ = await run(container, tmp_path, jid(), [FailureCategory.RENDER_ERROR], max_spec_chars=4000,
                           repair_enabled=False)
        assert rec.state == JobState.FAILED
        # the failed job is submitted again (a retry): a new billing attempt with its own hold
        r = submit(client)
        assert r.status_code == 202
    again = container.viz_store.jobs[jid()]
    assert (again.billing_attempt, again.hold_id) == (2, "hold-2")
    assert [k for _, k, _ in billing.reserves] == [f"{jid()}:1", f"{jid()}:2"]
    rec2, _ = await run(container, tmp_path, jid(), [None])
    assert rec2.state == JobState.READY
    # running the final hook again must not duplicate
    await viz_api._final_hook(container, container.viz_store)(rec2)
    assert [(e["attempt"], e["status"], e["hold_id"]) for e in publisher.events] == [(1, "error", "hold-1"), (2, "success", "hold-2")]
    assert all(e["job_id"] == jid() and e["user_id"] == OWNER for e in publisher.events)


async def test_repair_calls_are_attached_to_the_job_as_manim_plan(tmp_path, billing):
    app, container, publisher = make_app(tmp_path, billing)
    with TestClient(app) as client:
        submit(client)
        rec, calls = await run(container, tmp_path, jid(), [FailureCategory.RENDER_ERROR, None], max_spec_chars=4000)
    assert rec.state == JobState.READY and len(calls) == 2
    (event,) = publisher.events
    assert event["status"] == "success"
    assert [(c["purpose"], c["provider"], c["model"], c["input_tokens"], c["output_tokens"]) for c in event["calls"]] == [
        ("manim_plan", "groq", "openai/gpt-oss-120b", 700, 90)]


async def test_a_job_that_fails_before_rendering_releases_its_hold(tmp_path, billing):
    app, container, publisher = make_app(tmp_path, billing)
    with TestClient(app) as client:
        submit(client)
        container.viz_store.jobs[jid()].spec = {"template": "does_not_exist", "params": {}}   # fails validation
    # the gRPC channel above belongs to the TestClient's event loop: the job runs (and releases) in this one
    container.manim_billing.client = BillingClient(billing.target, "svc-token")
    rec, calls = await run(container, tmp_path, jid(), [None], max_spec_chars=4000, repair_enabled=False)
    assert rec.state == JobState.FAILED and calls == []         # never rendered
    assert billing.releases == [(OWNER, "hold-1", "provider_error")]
    assert publisher.events[0]["status"] == "error" and publisher.events[0]["hold_id"] == "hold-1"


async def test_a_job_that_failed_after_rendering_started_is_not_released_by_us(tmp_path, billing):
    app, container, publisher = make_app(tmp_path, billing)
    with TestClient(app) as client:
        submit(client)
        rec, _ = await run(container, tmp_path, jid(), [FailureCategory.RENDER_ERROR], max_spec_chars=4000, repair_enabled=False)
    assert rec.render_started is True and billing.releases == []   # billing decides from status=error + calls
    assert publisher.events[0]["status"] == "error"


async def test_cancel_publishes_the_event_once(tmp_path, billing):
    app, container, publisher = make_app(tmp_path, billing)
    with TestClient(app) as client:
        jid_ = submit(client).json()["id"]
        assert client.delete(f"/api/v1/visualizations/manim/jobs/{jid_}", headers=H).status_code == 202
        assert client.delete(f"/api/v1/visualizations/manim/jobs/{jid_}", headers=H).status_code == 202
    assert [(e["status"], e["attempt"]) for e in publisher.events] == [("error", 1)]


def test_internal_final_requires_the_token_and_publishes_once(tmp_path, billing):
    app, container, publisher = make_app(tmp_path, billing)
    container.config.visualization_internal_token = type(container.config.visualization_internal_token)("secret")
    with TestClient(app) as client:
        jid_ = submit(client).json()["id"]
        container.viz_store.jobs[jid_].state = JobState.FAILED
        body = {"job_id": jid_}
        assert client.post("/api/internal/visualizations/final", json=body).status_code == 401
        for _ in range(2):
            assert client.post("/api/internal/visualizations/final", json=body, headers={"X-Internal-Token": "secret"}).json() == {"ok": True}
    assert len(publisher.events) == 1
