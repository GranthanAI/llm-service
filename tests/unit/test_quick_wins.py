"""Quick wins in llm-service: event fields (L1), temporary chats (L2), think longer (L3), follow-ups (L4).

No network and no keys: provider SDK clients and Kafka are faked.
"""

import asyncio
import json
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.context.collector import ContextCollector
from app.context.schemas import MemoryContext
from app.followups import FollowupService, parse_suggestions
from app.models.billing import BillingGrant
from app.models.generation_preferences import GenerationPreferences, normalize_preferences
from app.models.pipeline_context import PipelineContext
from app.models.request import ChatMessageCreatedEvent
from app.models.response import ChatResponseGeneratedEvent
from app.providers.anthropic import AnthropicAdapter
from app.providers.catalog_check import compare
from app.providers.chain import ChainEntry, ProviderChain
from app.providers.gemini import GeminiAdapter
from app.providers.group_chain import GroupChainBuilder
from app.providers.groq import THINK_HIGH_MIN_OUTPUT, GroqAdapter
from app.providers.openai import OpenAIAdapter
from app.providers.registry import ModelRegistry, ModelSpec, RegistryError, ThinkSpec, think_output_tokens
from app.usage.collector import UsageCollector
from app.utils.metrics import FOLLOWUPS_TOTAL

THINK_EFFORT = ThinkSpec(kind="effort", think="high")
THINK_GROQ = ThinkSpec(kind="effort", normal="low", think="high", max_output_tokens=8192)
THINK_BUDGET = ThinkSpec(kind="budget", budget_tokens=4096)


def event(**over):
    base = dict(message_id="m1", conversation_id="c1", user_id="u1", content="hi", response_message_id="r1")
    base.update(over)
    return ChatMessageCreatedEvent(**base)


class Sink:
    def __init__(self):
        self.calls = []

    def record_provider_usage(self, **kw):
        self.calls.append(kw)


class Stream:
    def __init__(self, items):
        self.items = items

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for i in self.items:
            yield i


async def collect(agen):
    return [t async for t in agen]


MSGS = [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}]


# ================================================================================================== L1 events

def test_request_event_new_fields_default_off_and_accept_2_5():
    old = event()
    assert (old.think, old.temporary) is not None and old.think is False and old.temporary is False
    new = event(schema_version="2.5", think=True, temporary=True, preferences={"suggest_followups": False})
    assert new.think and new.temporary
    ctx = PipelineContext.from_event(new, request_id="req")
    assert ctx.think and ctx.temporary and ctx.think_used is False and ctx.preferences.suggest_followups is False


def test_suggest_followups_defaults_true_and_bad_values_fall_back():
    assert GenerationPreferences().suggest_followups is True
    assert normalize_preferences({"suggest_followups": False}).suggest_followups is False
    assert normalize_preferences({"suggest_followups": "garbage"}).suggest_followups is True
    assert GenerationPreferences().is_default


def test_generated_event_is_2_3_with_think_false_by_default():
    ev = ChatResponseGeneratedEvent(response_id="r", conversation_id="c", user_id="u", request_message_id="m")
    assert ev.schema_version == "2.3" and ev.think is False


# ============================================================================================ L2 temporary chats

async def test_temporary_chat_reads_no_memory_at_all():
    memory_client = MagicMock()
    memory_client.get_memory_context = AsyncMock()
    c = ContextCollector.__new__(ContextCollector)
    c.memory_client = memory_client
    ctx = PipelineContext.from_event(event(temporary=True), request_id="r")
    out = await c._fetch_memory(ctx)
    assert isinstance(out, MemoryContext) and not out.long_term_facts and not out.short_term_messages
    memory_client.get_memory_context.assert_not_called()
    normal = PipelineContext.from_event(event(), request_id="r")
    memory_client.get_memory_context.return_value = MemoryContext()
    await c._fetch_memory(normal)
    memory_client.get_memory_context.assert_awaited_once()


async def test_temporary_chat_publishes_no_memory_update_but_a_normal_one_does():
    from app.producers.kafka_producer import KafkaPublisher
    pub = KafkaPublisher.__new__(KafkaPublisher)
    pub.logger = MagicMock()
    pub._send = AsyncMock()
    await pub.publish_memory_update(PipelineContext.from_event(event(temporary=True), request_id="r"), "answer")
    pub._send.assert_not_called()
    await pub.publish_memory_update(PipelineContext.from_event(event(), request_id="r"), "answer")
    assert pub._send.await_args.kwargs["topic"] == "memory.update.requested"


# ============================================================================================ L3 registry / config

def test_models_yaml_think_blocks_load_and_match_billing_flags():
    reg = ModelRegistry.load("config/models.yaml")
    kinds = {k: v.think.kind for k, v in reg.models.items()}
    assert kinds["openai/gpt-oss-120b"] == "effort" and reg.models["openai/gpt-oss-120b"].think.normal == "low"
    assert kinds["claude-sonnet-5-5"] == "budget" and kinds["gpt-6-luna"] == "effort" and kinds["gemini-3.8-flash"] == "effort"
    billing = json.load(open("tests/fixtures/billing_models.json", encoding="utf-8"))["models"]
    flagged = [m for m in billing if m.get("think")]
    assert flagged and all(kinds[m["id"]] != "none" for m in flagged if m["id"] in kinds)
    assert compare(billing, reg, lambda p: True, lambda s: True).get("think_mismatch") is None


def test_catalog_check_flags_think_true_for_a_model_that_cannot_think():
    reg = ModelRegistry.load("config/models.yaml")
    spec = reg.models["claude-haiku-5-5"]
    from dataclasses import replace
    reg.models["claude-haiku-5-5"] = replace(spec, think=ThinkSpec())
    billing = [{"id": "claude-haiku-5-5", "group": "smart", "think": True}]
    assert compare(billing, reg, lambda p: True, lambda s: True)["think_mismatch"] == ["claude-haiku-5-5"]
    billing[0]["think"] = False
    assert "think_mismatch" not in compare(billing, reg, lambda p: True, lambda s: True)


def test_registry_rejects_a_bad_think_block():
    base = {"provider": "groq", "provider_model": "m", "group": "fast", "context_window": 100000, "max_output_tokens": 100}
    for think in ({"kind": "magic"}, {"kind": "effort"}, {"kind": "budget", "budget_tokens": 10}, "yes"):
        with pytest.raises(RegistryError):
            ModelRegistry.from_dict({"models": {"m": {**base, "think": think}}})
    ok = ModelRegistry.from_dict({"models": {"m": {**base, "think": {"kind": "effort", "think": "high"}}}})
    assert ok.models["m"].think.kind == "effort"


def test_think_output_budget_is_doubled_but_capped_and_untouched_for_models_that_cannot():
    spec = lambda think, out=8192: ModelSpec("m", "openai", "m", "smart", 100000, out, True, think)  # noqa: E731
    assert think_output_tokens(spec(THINK_EFFORT)) == 16384
    assert think_output_tokens(spec(THINK_GROQ)) == 8192                       # Groq: the TPM bucket decides, no doubling
    assert think_output_tokens(spec(ThinkSpec())) == 8192
    assert think_output_tokens(spec(ThinkSpec(kind="effort", think="high", max_output_tokens=10000))) == 10000


# ============================================================================================ L3 adapters (requests)

def o_chunks(parts=("a", "b"), completion=900):
    ch = [NS(choices=[NS(delta=NS(content=p), finish_reason=None)], usage=None) for p in parts]
    ch.append(NS(choices=[NS(delta=NS(content=None, reasoning_content="private reasoning"), finish_reason="stop")], usage=None))
    ch.append(NS(choices=[], usage=NS(prompt_tokens=50, completion_tokens=completion,
                                      completion_tokens_details=NS(reasoning_tokens=800),
                                      prompt_tokens_details=NS(cached_tokens=0))))
    return ch


def o_client(chunks):
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)
        return Stream(list(chunks))

    return NS(chat=NS(completions=NS(create=create))), calls


async def test_openai_sends_reasoning_effort_only_when_thinking_and_counts_reasoning_tokens():
    client, calls = o_client(o_chunks())
    adapter = OpenAIAdapter(api_key="k", model="gpt-6-luna", client=client)
    adapter.think_spec = THINK_EFFORT
    sink = Sink()
    params = {"think": True, "max_tokens": 16000, "_usage": sink}
    text = await collect(adapter.stream(MSGS, params))
    assert calls[0]["reasoning_effort"] == "high" and calls[0]["max_completion_tokens"] == 16000
    assert "".join(text) == "ab"                                                # no reasoning text in the chunks
    assert sink.calls[0]["output_tokens"] == 900                                # provider-reported, reasoning included
    assert params["_think_applied"] is True
    client2, calls2 = o_client(o_chunks())
    adapter2 = OpenAIAdapter(api_key="k", model="gpt-6-luna", client=client2)
    adapter2.think_spec = THINK_EFFORT
    params2 = {"max_tokens": 100}
    await collect(adapter2.stream(MSGS, params2))
    assert "reasoning_effort" not in calls2[0] and "_think_applied" not in params2


async def test_openai_model_without_a_control_ignores_the_think_flag():
    client, calls = o_client(o_chunks())
    adapter = OpenAIAdapter(api_key="k", model="gpt-6-luna", client=client)
    adapter.think_spec = ThinkSpec()
    params = {"think": True}
    await collect(adapter.stream(MSGS, params))
    assert "reasoning_effort" not in calls[0] and "_think_applied" not in params


def a_events():
    return [NS(type="message_start", message=NS(usage=NS(input_tokens=100, cache_creation_input_tokens=0,
                                                         cache_read_input_tokens=0, output_tokens=1))),
            NS(type="content_block_delta", delta=NS(type="thinking_delta", thinking="secret thoughts")),
            NS(type="content_block_delta", delta=NS(type="signature_delta", signature="sig")),
            NS(type="content_block_delta", delta=NS(type="text_delta", text="Hi")),
            NS(type="message_delta", delta=NS(stop_reason="end_turn"), usage=NS(output_tokens=1500))]


async def test_anthropic_thinking_budget_omitted_display_and_no_thinking_text_and_usage():
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)
        return Stream(a_events())

    adapter = AnthropicAdapter(api_key="k", model="claude-sonnet-5-5", client=NS(messages=NS(create=create)))
    adapter.think_spec = THINK_BUDGET
    sink = Sink()
    params = {"think": True, "max_tokens": 2000, "_usage": sink}
    text = await collect(adapter.stream(MSGS, params))
    assert calls[0]["thinking"] == {"type": "enabled", "budget_tokens": 4096, "display": "omitted"}
    assert calls[0]["max_tokens"] > 4096                                         # the budget counts inside max_tokens
    assert text == ["Hi"] and "secret" not in "".join(text)
    assert sink.calls[0]["output_tokens"] == 1500                                # thinking tokens included, as reported
    calls.clear()
    await collect(adapter.stream(MSGS, {"max_tokens": 2000}))
    assert "thinking" not in calls[0] and calls[0]["max_tokens"] == 2000


def gem_client(captured):
    async def gen():
        for piece in ("x", "y"):
            chunk = MagicMock()
            chunk.text = piece
            chunk.usage_metadata = None
            yield chunk
        last = MagicMock()
        last.text = ""
        last.usage_metadata = NS(prompt_token_count=40, candidates_token_count=10, thoughts_token_count=700,
                                 cached_content_token_count=0)
        yield last

    async def stream(model, contents, config):
        captured.append(config)
        return gen()

    return NS(aio=NS(models=NS(generate_content_stream=stream)))


async def test_gemini_thinking_level_budget_include_thoughts_false_and_thought_tokens_counted():
    from google.genai import types
    cap = []
    adapter = GeminiAdapter(api_key="k", model="gemini-3.8-flash", client=gem_client(cap))
    adapter.think_spec = THINK_EFFORT
    sink = Sink()
    params = {"think": True, "_usage": sink}
    text = await collect(adapter.stream(MSGS, params))
    assert "".join(text) == "xy"
    cfg = cap[0].thinking_config
    assert cfg.thinking_level == types.ThinkingLevel.HIGH and cfg.include_thoughts is False
    assert sink.calls[0]["output_tokens"] == 710                                 # candidates + thoughts
    cap.clear()
    adapter.think_spec = ThinkSpec(kind="budget", budget_tokens=2048)
    await collect(adapter.stream(MSGS, {"think": True}))
    assert cap[0].thinking_config.thinking_budget == 2048
    cap.clear()
    await collect(adapter.stream(MSGS, {}))
    assert cap[0].thinking_config is None


def groq_client(calls):
    async def create(**kwargs):
        calls.append(kwargs)
        usage = NS(prompt_tokens=60, completion_tokens=1200, total_tokens=1260, prompt_tokens_details=None)
        chunks = [NS(choices=[NS(delta=NS(content="ok", reasoning="deep thought"), finish_reason=None)], usage=None, x_groq=None),
                  NS(choices=[NS(delta=NS(content=None), finish_reason="stop")], usage=usage, x_groq=None)]
        return Stream(chunks)

    return NS(chat=NS(completions=NS(create=create)))


async def test_groq_effort_low_normally_high_when_thinking_and_medium_when_the_bucket_is_too_small():
    calls = []
    adapter = GroqAdapter(api_key="k", model="openai/gpt-oss-120b", client=groq_client(calls), tpm_limit=0)
    adapter.think_spec = THINK_GROQ
    await collect(adapter.stream(MSGS, {}))
    await collect(adapter.stream(MSGS, {"think": True}))
    assert [c["reasoning_effort"] for c in calls] == ["low", "high"]
    # free tier: 8,000 TPM per request leaves < 4,096 output tokens after a 5,000-token prompt -> medium
    calls.clear()
    small = GroqAdapter(api_key="k", model="openai/gpt-oss-120b", client=groq_client(calls), tpm_limit=8000)
    small.think_spec = THINK_GROQ
    big = [{"role": "user", "content": "word " * 5200}]
    params = {"think": True}
    text = await collect(small.stream(big, params))
    assert calls[0]["reasoning_effort"] == "medium" and "".join(text) == "ok" and params["_think_applied"] is True
    assert THINK_HIGH_MIN_OUTPUT == 4096
    calls.clear()
    roomy = GroqAdapter(api_key="k", model="openai/gpt-oss-120b", client=groq_client(calls), tpm_limit=8000)
    roomy.think_spec = THINK_GROQ
    await collect(roomy.stream([{"role": "user", "content": "short"}], {"think": True}))
    assert calls[0]["reasoning_effort"] == "high"


# ============================================================================================ L3 routing

class FakeAdapter:
    def __init__(self, name, tokens=("t",), error=None):
        self.name, self.tokens, self.error = name, tokens, error
        self.think_spec = None

    def wants_think(self, params):
        if params.get("think") and self.think_spec is not None and self.think_spec.kind != "none":
            params["_think_applied"] = True
            return True
        return False

    async def stream(self, messages, params=None, **kw):
        self.wants_think(params or {})
        if self.error:
            raise self.error
        for t in self.tokens:
            yield t


def _spec(mid, group, think):
    return ModelSpec(mid, "groq", mid, group, 100000, 4096, True, think)


def _prompt():
    from app.context_window.models import PromptSection, TrimmedPrompt
    return TrimmedPrompt(sections=[PromptSection(name="q", content="x", priority=1, token_count=1, trimmable=False)],
                         messages=[{"role": "user", "content": "x"}], total_tokens=1, mode="default", target_model="m")


async def test_chain_reports_think_true_only_when_the_answering_entry_really_reasoned():
    capable, plain = FakeAdapter("a"), FakeAdapter("b")
    capable.think_spec, plain.think_spec = THINK_EFFORT, ThinkSpec()
    for first, second, expected in ((capable, plain, True), (plain, capable, False)):
        chain = ProviderChain([ChainEntry(first.name, first, None, 1.0, spec=_spec(first.name, "smart", first.think_spec)),
                               ChainEntry(second.name, second, None, 1.0, spec=_spec(second.name, "smart", second.think_spec))])
        params = {"think": True}
        [t async for t in chain.generate_stream(_prompt(), params)]
        assert params["_route"].get("think", False) is expected


async def test_a_failed_capable_attempt_does_not_leak_think_to_the_model_that_answers():
    failing, plain = FakeAdapter("a", error=RuntimeError("boom")), FakeAdapter("b")
    failing.think_spec, plain.think_spec = THINK_EFFORT, ThinkSpec()
    chain = ProviderChain([ChainEntry("a", failing, None, 1.0, spec=_spec("a", "smart", THINK_EFFORT)),
                           ChainEntry("b", plain, None, 1.0, spec=_spec("b", "smart", ThinkSpec()))])
    params = {"think": True}
    [t async for t in chain.generate_stream(_prompt(), params)]
    assert params["_route"]["model_id"] == "b" and not params["_route"].get("think")


def _registry(order, kinds):
    models = {m: _spec(m, "smart", THINK_EFFORT if kinds[m] else ThinkSpec()) for m in order}
    return ModelRegistry(models, {"smart": order})


def _builder(reg):
    return GroupChainBuilder(reg, NS(generation_idle_timeout_ms=1000, groq_first_token_timeout_ms=1000,
                                                  gemini_timeout_ms=1000, generation_first_token_timeout_ms=1000),
                             None, lambda spec: FakeAdapter(spec.id), lambda p: True)


def test_auto_prefers_the_first_think_capable_model_and_never_leaves_the_group(monkeypatch):
    import app.providers.group_chain as gc
    monkeypatch.setattr(gc, "build_group_fitter", lambda *a, **k: None)
    reg = _registry(["a", "b", "c"], {"a": False, "b": True, "c": True})
    auto = BillingGrant(granted_group="smart", granted_model=None)
    chain = _builder(reg).build(auto, think=True)
    assert [e.spec.id for e in chain.entries] == ["b", "c", "a"]                 # capable first, stable, nothing dropped
    assert [e.spec.id for e in _builder(reg).build(auto, think=False).entries] == ["a", "b", "c"]
    none_capable = _registry(["a", "b"], {"a": False, "b": False})
    assert [e.spec.id for e in _builder(none_capable).build(auto, think=True).entries] == ["a", "b"]   # answers normally
    explicit = BillingGrant(granted_group="smart", granted_model="a")
    assert _builder(reg).build(explicit, think=True).entries[0].spec.id == "a"   # the user's own pick is respected


def test_group_chain_gives_each_adapter_its_registry_think_spec(monkeypatch):
    import app.providers.group_chain as gc
    monkeypatch.setattr(gc, "build_group_fitter", lambda *a, **k: None)
    chain = _builder(_registry(["a", "b"], {"a": True, "b": False})).build(BillingGrant(granted_group="smart"))
    assert [e.adapter.think_spec.kind for e in chain.entries] == ["effort", "none"]


async def test_pipeline_tracks_think_used_from_the_route():
    from app.pipeline.request_pipeline import RequestPipeline
    ctx = PipelineContext.from_event(event(think=True), request_id="r")
    params = {"_route": {"provider": "groq:m", "model_id": "m", "think": True}}

    async def tokens():
        yield "x"

    [t async for t in RequestPipeline._track_route(tokens(), params, ctx)]
    assert ctx.think_used is True
    ctx2 = PipelineContext.from_event(event(think=True), request_id="r")
    [t async for t in RequestPipeline._track_route(tokens(), {"_route": {"provider": "p", "think": False}}, ctx2)]
    assert ctx2.think_used is False


async def test_generated_event_carries_think_from_the_context():
    from app.producers.kafka_producer import KafkaPublisher
    pub = KafkaPublisher.__new__(KafkaPublisher)
    pub.logger, pub.config = MagicMock(), NS(kafka_output_topic="chat.response.generated")
    pub._send = AsyncMock()
    pub._terminal_already_published = lambda ctx, kind: False
    pub._mark_terminal_published = lambda ctx: None
    ctx = PipelineContext.from_event(event(think=True), request_id="req-12345678901234")
    ctx.think_used = True
    await pub.publish_response(ctx, "answer")
    assert pub._send.await_args.kwargs["payload"].think is True
    ctx.think_used = False
    await pub.publish_response(ctx, "answer")
    assert pub._send.await_args.kwargs["payload"].think is False


# ============================================================================================ L4 follow-ups

LONG = "A long explanation of photosynthesis. " * 12


class FakePublisher:
    def __init__(self):
        self.suggestions, self.metered = [], []

    async def publish_suggestions(self, ctx, suggestions, model):
        self.suggestions.append((ctx.response_message_id, suggestions, model))

    async def publish_usage_metered(self, **kw):
        self.metered.append(kw)


class FakeModel:
    def __init__(self, reply="", error=None, delay=0.0, record=True):
        self.reply, self.error, self.delay, self.record = reply, error, delay, record
        self.calls = []

    async def execute(self, messages, params=None):
        self.calls.append((messages, params))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.record:
            params["_usage"].record_provider_usage(provider="gemini", model="gemini-3.1-flash-lite", input_tokens=300,
                                                   output_tokens=40)
        if self.error:
            raise self.error
        return NS(content=self.reply)


def fctx(**over):
    ev = event(preferences=over.pop("preferences", {}), **over)
    return PipelineContext.from_event(ev, request_id="req")


def service(model, pub=None, **kw):
    pub = pub or FakePublisher()
    return FollowupService(pub, model, "gemini-3.1-flash-lite", **kw), pub


def outcome(name):
    return FOLLOWUPS_TOTAL.labels(outcome=name)._value.get()


async def test_followups_are_published_after_a_good_answer_and_the_cost_is_metered():
    svc, pub = service(FakeModel(json.dumps({"suggestions": ["Why does it need light?", "What is chlorophyll?", "Where does it happen?"]})))
    before = outcome("published")
    svc.maybe_start(fctx(), LONG)
    await svc.drain()
    assert pub.suggestions == [("r1", ["Why does it need light?", "What is chlorophyll?", "Where does it happen?"], "gemini-3.1-flash-lite")]
    assert outcome("published") == before + 1
    (m,) = pub.metered
    assert m["feature"] == "followups" and m["user_id"] == "u1" and m["request_message_id"] == "m1"
    assert m["calls"][0]["purpose"] == "followups" and m["calls"][0]["input_tokens"] == 300


async def test_followups_hinglish_is_kept_as_is_and_the_prompt_asks_for_the_same_language():
    q = ["Isme chlorophyll ka kya role hai?", "Light na ho toh kya hoga?"]
    model = FakeModel(json.dumps({"suggestions": q}))
    svc, pub = service(model)
    svc.maybe_start(fctx(content="Photosynthesis kya hota hai?", preferences={"language": "hinglish", "language_name": "Hinglish"}), LONG)
    await svc.drain()
    assert pub.suggestions[0][1] == q
    system, user = model.calls[0][0][0]["content"], model.calls[0][0][1]["content"]
    assert "same language and script" in system and "Hinglish" in system and "Hinglish" in user
    assert "Photosynthesis kya hota hai?" in user and model.calls[0][1]["max_tokens"] == 300


async def test_followups_tutor_mode_prompt_and_answer_tail_only():
    model = FakeModel(json.dumps({"suggestions": ["a?", "b?"]}))
    svc, _ = service(model)
    ctx = fctx()
    ctx.plan = NS(mode=NS(value="tutor"))
    answer = "x" * 5000 + "THE-END"
    svc.maybe_start(ctx, answer)
    await svc.drain()
    user = model.calls[0][0][1]["content"]
    assert "Mode: tutor" in user and "THE-END" in user and user.count("x") <= 1500
    assert "go deeper" in model.calls[0][0][0]["content"]


@pytest.mark.parametrize("answer,prefs,enabled", [
    ("short", {}, True),                                             # under 200 characters
    (LONG, {"suggest_followups": False}, True),                     # the user's preference is off
    (LONG, {}, False),                                               # disabled in settings
])
async def test_followups_skipped_when_short_or_off(answer, prefs, enabled):
    model = FakeModel(json.dumps({"suggestions": ["a?", "b?"]}))
    svc, pub = service(model, enabled=enabled)
    before = outcome("skipped")
    assert svc.maybe_start(fctx(preferences=prefs), answer) is None
    await svc.drain()
    assert pub.suggestions == [] and model.calls == [] and outcome("skipped") == before + 1


async def test_followups_never_start_without_a_response_id():
    svc, pub = service(FakeModel("{}"))
    ctx = fctx()
    ctx.response_message_id = None
    assert svc.maybe_start(ctx, LONG) is None


async def test_followups_nothing_published_on_timeout_invalid_json_or_too_few():
    pub = FakePublisher()
    slow, _ = service(FakeModel(json.dumps({"suggestions": ["a?", "b?"]}), delay=0.5), pub, timeout_ms=50)
    before = outcome("timeout")
    slow.maybe_start(fctx(), LONG)
    await slow.drain()
    assert outcome("timeout") == before + 1 and pub.suggestions == []
    assert len(pub.metered) == 0                                     # the call never reported usage before it timed out
    for reply in ("not json at all", json.dumps({"suggestions": ["only one?"]}), json.dumps({"suggestions": "nope"}),
                  json.dumps({"suggestions": ["x" * 130, "two\nlines", "", "ok?"]})):
        before = outcome("invalid")
        bad, _ = service(FakeModel(reply), pub)
        bad.maybe_start(fctx(), LONG)
        await bad.drain()
        assert outcome("invalid") == before + 1
    assert pub.suggestions == []
    boom, _ = service(FakeModel(error=RuntimeError("provider down")), pub)
    before = outcome("error")
    boom.maybe_start(fctx(), LONG)
    await boom.drain()                                               # the task swallows the failure
    assert outcome("error") == before + 1 and pub.suggestions == []
    assert pub.metered                                               # a failed call that used tokens is still metered


def test_parse_suggestions_rules():
    assert parse_suggestions('```json\n{"suggestions": ["a?", " b? ", "a?", "", "c?", "d?"]}\n```') == ["a?", "b?", "c?"]
    assert parse_suggestions("[]") == [] and parse_suggestions("") == [] and parse_suggestions('{"suggestions": [1, 2]}') == []
    assert parse_suggestions(json.dumps({"suggestions": ["x" * 120, "x" * 121]})) == ["x" * 120]


async def test_the_streaming_engine_starts_followups_only_after_the_answer_was_published_and_survives_their_failure():
    from app.services.streaming_service import StreamingEngine
    order = []
    pub = MagicMock()
    pub.publish_chunk = AsyncMock()
    pub.publish_response = AsyncMock(side_effect=lambda *a: order.append("response"))
    pub.publish_memory_update = AsyncMock(side_effect=lambda *a: order.append("memory"))
    follow = MagicMock()
    follow.maybe_start = MagicMock(side_effect=lambda *a: order.append("followups") or (_ for _ in ()).throw(RuntimeError("x")))
    engine = StreamingEngine(publisher=pub, followups=follow)

    async def tokens():
        yield "hello"

    out = await engine.stream(tokens(), PipelineContext.from_event(event(), request_id="r"))
    assert out == "hello" and order == ["response", "memory", "followups"]       # a failure to start never breaks the answer


async def test_cancelled_stream_never_starts_followups():
    from app.services.streaming_service import StreamingEngine
    from app.utils.cancellation import CancellationToken
    pub = MagicMock()
    pub.publish_chunk = AsyncMock()
    pub.publish_cancellation = AsyncMock()
    follow = MagicMock()
    engine = StreamingEngine(publisher=pub, followups=follow)
    token = CancellationToken()
    token.cancel("client_disconnected") if hasattr(token, "cancel") else None

    async def tokens():
        yield "hello"

    await engine.stream(tokens(), PipelineContext.from_event(event(), request_id="r"), cancellation_token=token)
    follow.maybe_start.assert_not_called()


def test_publish_suggestions_payload_matches_the_contract():
    from app.producers.kafka_producer import KafkaPublisher
    pub = KafkaPublisher.__new__(KafkaPublisher)
    pub._send = AsyncMock()
    ctx = PipelineContext.from_event(event(), request_id="req")
    asyncio.run(pub.publish_suggestions(ctx, ["a?", "b?"], "gemini-3.1-flash-lite"))
    kw = pub._send.await_args.kwargs
    p = kw["payload"]
    assert kw["topic"] == "chat.response.suggestions" and kw["key"] == "c1"
    assert p["event_type"] == "chat.response.suggestions" and p["schema_version"] == "1.0"
    assert p["occurred_at"].endswith("Z") and len(p["event_id"]) == 36
    assert (p["conversation_id"], p["user_id"], p["request_message_id"], p["response_message_id"]) == ("c1", "u1", "m1", "r1")
    assert p["suggestions"] == ["a?", "b?"] and p["model"] == "gemini-3.1-flash-lite"


def test_usage_collector_knows_the_followups_purpose_and_the_prompt_file_loads():
    from app.followups.service import load_prompt
    from app.usage.collector import PURPOSES
    assert "followups" in PURPOSES
    p = load_prompt()
    assert p["name"] == "followups" and "{question}" in p["user_template"] and "{answer_tail}" in p["user_template"]
