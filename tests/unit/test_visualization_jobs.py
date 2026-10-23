import asyncio
import json
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import visualizations as viz_api
from app.config.settings import LLMServiceConfig
from app.visualization.jobs import JobRecord, JobState, MemoryJobStore, can_transition, job_id_for
from app.visualization.manim_templates import TEMPLATES, validate_spec
from app.visualization.policy import FailureCategory
from app.visualization.renderer import MediaStore, RenderResult, is_valid_mp4
from app.visualization.repair import build_repair_prompt, repair_block, validate_candidate
from app.visualization.runner import run_job

pytestmark = pytest.mark.anyio

SPEC = {"template": "text_steps", "title": "T", "params": {"steps": ["a", "b"]}}
OWNER = str(uuid.uuid4())


@pytest.fixture
def anyio_backend():
    return "asyncio"


def make_record(**kw) -> JobRecord:
    spec = json.loads(validate_spec(SPEC).canonical_json())
    return JobRecord(id=job_id_for(OWNER, json.dumps(spec)), owner=OWNER, spec=spec, **kw)


def fake_mp4(tmp_path: Path) -> Path:
    p = tmp_path / "v.mp4"
    p.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 2000)
    return p


def renderer(results, tmp_path):
    calls = []

    async def render(spec, **kw):
        calls.append(spec.canonical_json())
        r = results[min(len(calls) - 1, len(results) - 1)]
        if r is None:
            f = tmp_path / f"o{len(calls)}.mp4"
            f.write_bytes(fake_mp4(tmp_path).read_bytes())
            return RenderResult(True, path=f)
        return RenderResult(False, r)
    return render, calls


async def submit(store):
    rec, _ = await store.create_if_absent(make_record())
    return rec


async def test_state_machine_rules():
    assert can_transition(JobState.PENDING, JobState.VALIDATING)
    assert not can_transition(JobState.READY, JobState.RENDERING)
    assert not can_transition(JobState.FAILED, JobState.PENDING)


async def test_job_id_is_tenant_scoped_and_idempotent():
    assert job_id_for("u1", "x") != job_id_for("u2", "x")
    store = MemoryJobStore()
    a, created_a = await store.create_if_absent(make_record())
    b, created_b = await store.create_if_absent(make_record())
    assert created_a and not created_b and a is b


async def test_success_first_attempt(tmp_path):
    store, media = MemoryJobStore(), MediaStore(str(tmp_path / "m"))
    rec = await submit(store)
    render, calls = renderer([None], tmp_path)
    out = await run_job(store, media, rec.id, repair=None, render=render)
    assert out.state == JobState.READY and out.attempt == 1 and len(calls) == 1
    assert media.exists(rec.id) and is_valid_mp4(media.path_for(rec.id), 10**7)


async def test_repair_then_success_uses_attempt_two(tmp_path):
    store, media = MemoryJobStore(), MediaStore(str(tmp_path / "m"))
    rec = await submit(store)
    render, calls = renderer([FailureCategory.RENDER_ERROR, None], tmp_path)
    fixed = validate_spec({**SPEC, "params": {"steps": ["c", "d"]}}).canonical_json()
    seen = []

    async def repair(kind, spec, err, attempt):
        seen.append(attempt)
        return fixed

    out = await run_job(store, media, rec.id, repair=repair, render=render)
    assert out.state == JobState.READY and out.attempt == 2 and seen == [2]
    assert calls[1] == fixed


async def test_three_total_attempts_then_failed(tmp_path):
    store, media = MemoryJobStore(), MediaStore(str(tmp_path / "m"))
    rec = await submit(store)
    render, calls = renderer([FailureCategory.RENDER_ERROR], tmp_path)
    repairs = []

    async def repair(kind, spec, err, attempt):
        repairs.append(attempt)
        return spec

    out = await run_job(store, media, rec.id, repair=repair, render=render)
    assert out.state == JobState.FAILED and out.failure == "render_error"
    assert len(calls) == 3 and repairs == [2, 3] and out.attempt == 3
    assert not media.exists(rec.id)


async def test_permanent_failure_stops_immediately(tmp_path):
    store, media = MemoryJobStore(), MediaStore(str(tmp_path / "m"))
    rec = await submit(store)
    render, calls = renderer([FailureCategory.RESOURCE_LIMIT], tmp_path)
    out = await run_job(store, media, rec.id, repair=None, render=render)
    assert out.state == JobState.FAILED and len(calls) == 1


async def test_transient_failure_retries_without_model(tmp_path):
    store, media = MemoryJobStore(), MediaStore(str(tmp_path / "m"))
    rec = await submit(store)
    render, calls = renderer([FailureCategory.TIMEOUT, None], tmp_path)

    async def repair(*a):
        raise AssertionError("model must not be called for transient failures")

    out = await run_job(store, media, rec.id, repair=repair, render=render)
    assert out.state == JobState.READY and len(calls) == 2


async def test_repair_unavailable_still_bounded(tmp_path):
    store, media = MemoryJobStore(), MediaStore(str(tmp_path / "m"))
    rec = await submit(store)
    render, calls = renderer([FailureCategory.RENDER_ERROR], tmp_path)
    out = await run_job(store, media, rec.id, repair=None, render=render)
    assert out.state == JobState.FAILED and len(calls) == 1


async def test_duplicate_worker_is_noop(tmp_path):
    store, media = MemoryJobStore(), MediaStore(str(tmp_path / "m"))
    rec = await submit(store)
    render, calls = renderer([None], tmp_path)
    await asyncio.gather(run_job(store, media, rec.id, repair=None, render=render),
                         run_job(store, media, rec.id, repair=None, render=render))
    assert len(calls) == 1


async def test_cancelled_job_not_rendered(tmp_path):
    store, media = MemoryJobStore(), MediaStore(str(tmp_path / "m"))
    rec = await submit(store)
    await store.update(rec.id, JobState.CANCELLED)
    render, calls = renderer([None], tmp_path)
    out = await run_job(store, media, rec.id, repair=None, render=render)
    assert out.state == JobState.CANCELLED and not calls


def test_media_store_rejects_traversal(tmp_path):
    m = MediaStore(str(tmp_path))
    assert m.path_for("../../etc/passwd") is None and m.path_for("mj_" + "a" * 32) is not None


def test_mp4_validation(tmp_path):
    bad = tmp_path / "b.mp4"
    bad.write_bytes(b"x" * 5000)
    assert not is_valid_mp4(bad, 10**6) and not is_valid_mp4(tmp_path / "none.mp4", 10**6)
    assert is_valid_mp4(fake_mp4(tmp_path), 10**6) and not is_valid_mp4(fake_mp4(tmp_path), 100)


# ---- repair --------------------------------------------------------------------

class Chain:
    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    async def generate_stream(self, prompt, params=None):
        self.prompts.append(prompt)
        yield self.reply


async def test_repair_mermaid_validates_and_is_bounded():
    chain = Chain("```mermaid\nflowchart TD\nA-->B\n```")
    out = await repair_block(chain, "mermaid", "flowchart TD\nA[x", "syntax", 2)
    assert out.ok and out.content.startswith("flowchart")
    assert len(chain.prompts[0].messages) == 2          # only rules + the failed block, no conversation
    bad = await repair_block(Chain("flowchart TD\nA[<script>]"), "mermaid", "x", "syntax", 2)
    assert not bad.ok and bad.category == "unsafe"


async def test_repair_manim_rejects_code_and_unknown_templates():
    assert (await repair_block(Chain("import os"), "manim", "{}", "invalid_spec", 2)).category == "invalid_spec"
    out = await repair_block(Chain(json.dumps({"template": "rm_rf", "params": {}})), "manim", "{}", "x", 2)
    assert out.category == "unsupported_template"
    good = await repair_block(Chain(json.dumps(SPEC)), "manim", "{}", "x", 2)
    assert good.ok


async def test_repair_provider_failure_is_contained():
    class Boom:
        async def generate_stream(self, p, params=None):
            raise RuntimeError("down")
            yield ""
    assert (await repair_block(Boom(), "mermaid", "x", "syntax", 2)).category == "worker_unavailable"
    assert (await repair_block(None, "mermaid", "x", "syntax", 2)).ok is False


def test_repair_prompt_truncates_input():
    p = build_repair_prompt("mermaid", "x" * 50_000, "e" * 5000, 2)
    assert len(p.messages[1]["content"]) < 4000


# ---- API -----------------------------------------------------------------------

def make_client(tmp_path, **cfg):
    config = LLMServiceConfig(manim_enabled=True, manim_media_dir=str(tmp_path / "media"), **cfg)
    container = SimpleNamespace(config=config, generation_router=Chain(json.dumps(SPEC)))
    app = FastAPI()
    app.state.container = container
    app.include_router(viz_api.router, prefix="/api")
    app.include_router(viz_api.internal_router, prefix="/api")
    return TestClient(app), container


def H(user=OWNER):
    return {"X-User-Id": user}


def test_api_requires_identity(tmp_path):
    c, _ = make_client(tmp_path)
    assert c.post("/api/v1/visualizations/manim/jobs", json={"spec": SPEC}).status_code == 401
    assert c.post("/api/v1/visualizations/manim/jobs", json={"spec": SPEC}, headers={"X-User-Id": "nope"}).status_code == 401


def test_api_rejects_bad_specs_without_leaking(tmp_path):
    c, _ = make_client(tmp_path)
    r = c.post("/api/v1/visualizations/manim/jobs", json={"spec": {"template": "x; import os"}}, headers=H())
    assert r.status_code == 422 and r.json()["detail"]["failure"] == "unsupported_template"
    assert "import os" not in r.text


def test_api_disabled_by_default(tmp_path):
    config = LLMServiceConfig(manim_enabled=False, manim_media_dir=str(tmp_path))
    app = FastAPI()
    app.state.container = SimpleNamespace(config=config, generation_router=None)
    app.include_router(viz_api.router, prefix="/api")
    c = TestClient(app)
    assert c.post("/api/v1/visualizations/manim/jobs", json={"spec": SPEC}, headers=H()).status_code == 403
    assert c.get("/api/v1/visualizations/config").json()["manim"] is False


def test_api_job_lifecycle_idempotent_and_tenant_isolated(tmp_path, monkeypatch):
    c, container = make_client(tmp_path)
    started = []
    monkeypatch.setattr(viz_api, "_start_inline", lambda cont, jid: started.append(jid))
    r1 = c.post("/api/v1/visualizations/manim/jobs", json={"spec": SPEC}, headers=H())
    r2 = c.post("/api/v1/visualizations/manim/jobs", json={"spec": SPEC}, headers=H())
    assert r1.status_code == 202 and r1.json()["id"] == r2.json()["id"] and len(started) == 1
    jid = r1.json()["id"]
    assert "spec" not in r1.json() and "owner" not in r1.json()
    assert c.get(f"/api/v1/visualizations/manim/jobs/{jid}", headers=H()).json()["state"] == "pending"
    other = H(str(uuid.uuid4()))
    assert c.get(f"/api/v1/visualizations/manim/jobs/{jid}", headers=other).status_code == 404
    assert c.get(f"/api/v1/visualizations/manim/jobs/{jid}/media", headers=other).status_code == 404
    assert c.get(f"/api/v1/visualizations/manim/jobs/{jid}/media", headers=H()).status_code == 404   # not ready yet
    assert c.delete(f"/api/v1/visualizations/manim/jobs/{jid}", headers=other).status_code == 404
    assert c.delete(f"/api/v1/visualizations/manim/jobs/{jid}", headers=H()).json()["state"] == "cancelled"
    assert c.get("/api/v1/visualizations/manim/jobs/not-an-id", headers=H()).status_code == 404


def test_api_media_served_when_ready(tmp_path, monkeypatch):
    c, container = make_client(tmp_path)
    monkeypatch.setattr(viz_api, "_start_inline", lambda cont, jid: None)
    jid = c.post("/api/v1/visualizations/manim/jobs", json={"spec": SPEC}, headers=H()).json()["id"]
    store, media = container.viz_store, container.viz_media

    async def finish():
        await store.update(jid, JobState.VALIDATING)
        await store.update(jid, JobState.RENDERING)
        media.save(jid, fake_mp4(tmp_path))
        await store.update(jid, JobState.READY, media_key=jid)
    asyncio.run(finish())
    r = c.get(f"/api/v1/visualizations/manim/jobs/{jid}/media", headers=H())
    assert r.status_code == 200 and r.headers["content-type"] == "video/mp4" and "private" in r.headers["cache-control"]


def test_api_active_job_limit(tmp_path, monkeypatch):
    c, _ = make_client(tmp_path, manim_max_active_jobs_per_user=1)
    monkeypatch.setattr(viz_api, "_start_inline", lambda cont, jid: None)
    assert c.post("/api/v1/visualizations/manim/jobs", json={"spec": SPEC}, headers=H()).status_code == 202
    other = {**SPEC, "params": {"steps": ["x", "y"]}}
    assert c.post("/api/v1/visualizations/manim/jobs", json={"spec": other}, headers=H()).status_code == 429


def test_api_repair_attempt_bounds(tmp_path):
    c, _ = make_client(tmp_path, mermaid_enabled=True)
    body = {"kind": "manim", "content": "{}", "error": "x", "attempt": 2}
    assert c.post("/api/v1/visualizations/repair", json=body, headers=H()).json()["ok"] is True
    assert c.post("/api/v1/visualizations/repair", json={**body, "attempt": 1}, headers=H()).status_code == 422
    assert c.post("/api/v1/visualizations/repair", json={**body, "attempt": 4}, headers=H()).json()["failure"] == "budget_exhausted"
    assert c.post("/api/v1/visualizations/repair", json={**body, "content": "x" * 9000}, headers=H()).status_code == 422


def test_internal_repair_requires_token(tmp_path):
    c, _ = make_client(tmp_path, visualization_internal_token="s3cret")
    body = {"kind": "manim", "content": "{}", "error": "x", "attempt": 2}
    assert c.post("/api/internal/visualizations/repair", json=body).status_code == 401
    assert c.post("/api/internal/visualizations/repair", json=body, headers={"X-Internal-Token": "wrong"}).status_code == 401
    assert c.post("/api/internal/visualizations/repair", json=body, headers={"X-Internal-Token": "s3cret"}).json()["ok"]
    c2, _ = make_client(tmp_path)          # no token configured -> endpoint closed
    assert c2.post("/api/internal/visualizations/repair", json=body, headers={"X-Internal-Token": ""}).status_code == 401


def test_candidate_validation_helper():
    assert validate_candidate("mermaid", "flowchart TD\nA-->B").ok
    assert not validate_candidate("manim", "not json").ok
    assert set(TEMPLATES)


def test_api_repair_accepts_long_browser_error_messages(tmp_path):
    c, _ = make_client(tmp_path)
    body = {"kind": "manim", "content": "{}", "error": "Parse error on line 2: " + "x" * 500, "attempt": 2}
    assert c.post("/api/v1/visualizations/repair", json=body, headers=H()).status_code == 200


# ---- artifact-registry contract (conversation-service fingerprints with this) and repository hygiene ----------------

def test_config_exposes_the_render_identity_used_in_artifact_fingerprints(tmp_path):
    from app.visualization.manim_templates import RENDERER_VERSION
    c, _ = make_client(tmp_path, manim_quality="m")
    body = c.get("/api/v1/visualizations/config").json()
    assert body["renderer_version"] == RENDERER_VERSION and body["quality"] == "m"


def test_default_media_directory_is_outside_the_source_tree():
    import pathlib
    repo = pathlib.Path(__file__).resolve().parents[2]
    media = pathlib.Path(LLMServiceConfig().manim_media_dir).resolve()
    assert repo not in media.parents and media != repo, "renders must never default to a path inside the repository"


def test_rendered_media_is_git_ignored():
    import pathlib
    ignore = (pathlib.Path(__file__).resolve().parents[2] / ".gitignore").read_text()
    assert ".media/" in ignore and "*.mp4" in ignore


# ---- thumbnails ------------------------------------------------------------------------------------------------

def test_media_store_keeps_an_optional_thumbnail_next_to_the_video(tmp_path):
    store = MediaStore(str(tmp_path / "m"))
    jid = "mj_" + "a" * 32
    src = tmp_path / "t.png"
    src.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 20)
    store.save_thumbnail(jid, src)
    assert store.thumb_path_for(jid).is_file() and store.thumb_path_for("../evil") is None
    store.delete(jid)
    assert not store.thumb_path_for(jid).exists()


@pytest.mark.asyncio
async def test_runner_stores_the_thumbnail_and_cleans_every_temp_file(tmp_path):
    store, media = MemoryJobStore(), MediaStore(str(tmp_path / "m"))
    jid = job_id_for(OWNER, validate_spec(SPEC).canonical_json())
    await store.create_if_absent(JobRecord(id=jid, owner=OWNER, spec=json.loads(validate_spec(SPEC).canonical_json())))
    video, thumb = tmp_path / "v.mp4", tmp_path / "t.png"
    video.write_bytes(b"x" * 2000)
    thumb.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 50)

    async def fake_render(spec, **kw):
        return RenderResult(True, path=video, thumbnail=thumb)

    rec = await run_job(store, media, jid, repair=None, render=fake_render)
    assert rec.state == JobState.READY and media.exists(jid) and media.thumb_path_for(jid).is_file()
    assert not video.exists() and not thumb.exists()
