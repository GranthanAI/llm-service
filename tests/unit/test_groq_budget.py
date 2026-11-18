"""C1: the shared per-model Groq token budget (headers in, answers pick a model with budget, analysis never blocks)."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.exceptions.provider import AllProvidersFailedError
from app.providers import chain as chain_module
from app.providers.chain import ChainEntry, ProviderChain
from app.providers.groq import GroqAdapter
from app.providers.groq_budget import ANALYSIS_SKIPPED, BUDGET_REMAINING, GroqBudget, parse_reset_seconds, read_headers
from app.request_analyzer.analyzer import RequestAnalyzer
from app.request_analyzer.groq_client import AnalysisBudgetSkipped, GroqAnalysisClient

pytestmark = pytest.mark.asyncio


class FakeRedis:
    """The few commands the budget uses, with TTLs counted down by `tick`."""

    def __init__(self):
        self.kv, self.ttls, self.z = {}, {}, {}

    async def get(self, k):
        return self.kv.get(k)

    async def set(self, k, v, ex=None):
        self.kv[k] = str(v).encode()
        self.ttls[k] = ex

    async def ttl(self, k):
        return self.ttls.get(k, -2) if k in self.kv else -2

    async def exists(self, k):
        return 1 if k in self.kv else 0

    async def decrby(self, k, n):
        self.kv[k] = str(int(self.kv[k]) - n).encode()

    async def incrby(self, k, n):
        self.kv[k] = str(int(self.kv[k]) + n).encode()

    async def expire(self, k, s):
        return 1

    async def zadd(self, k, mapping):
        self.z.setdefault(k, {}).update(mapping)

    async def zremrangebyscore(self, k, lo, hi):
        self.z[k] = {m: s for m, s in self.z.get(k, {}).items() if not lo <= s <= hi}

    async def zrange(self, k, a, b, withscores=False):
        return sorted(((m.encode(), s) for m, s in self.z.get(k, {}).items()), key=lambda x: x[1])

    def expire_key(self, k):
        self.kv.pop(k, None)


M20, M120 = "openai/gpt-oss-20b", "openai/gpt-oss-120b"


def headers(remaining, reset):
    return {"x-ratelimit-remaining-tokens": str(remaining), "x-ratelimit-reset-tokens": reset}


async def test_headers_are_parsed_and_stored_with_the_reset_as_ttl():
    assert parse_reset_seconds("7.66s") == pytest.approx(7.66)
    assert parse_reset_seconds("1m30.5s") == pytest.approx(90.5)
    assert parse_reset_seconds("120ms") == pytest.approx(0.12)
    assert read_headers(headers(1234, "6.5s")) == (1234, 6.5)
    assert read_headers({}) is None and read_headers(None) is None
    r = FakeRedis()
    b = GroqBudget(r)
    assert await b.record_headers(M20, headers(1234, "6.5s"))
    assert r.kv[f"groq:budget:{M20}:remaining"] == b"1234" and r.ttls[f"groq:budget:{M20}:remaining"] == 7
    assert BUDGET_REMAINING.labels(model=M20)._value.get() == 1234


async def test_known_headers_decide_and_the_claim_is_deducted_for_everyone():
    b = GroqBudget(FakeRedis(), answer_reserve=4000)
    await b.record_headers(M20, headers(6000, "20s"))
    assert await b.acquire(M20, 3000, "answer") == 0.0          # an answer may use the whole bucket
    assert (await b.known_remaining(M20))[0] == 3000            # and the next caller sees what is left
    wait = await b.acquire(M20, 3500, "answer")
    assert 0 < wait <= 20                                       # does not fit: wait for the reset


async def test_the_answer_reserve_is_respected_by_analysis_and_memory():
    b = GroqBudget(FakeRedis(), answer_reserve=4000)
    await b.record_headers(M20, headers(5000, "30s"))
    assert await b.acquire(M20, 1500, "analysis") > 0           # 5000 - 1500 < 4000: leaves too little for an answer
    assert await b.acquire(M20, 1000, "memory") == 0.0          # 5000 - 1000 >= 4000
    assert await b.acquire(M20, 3500, "answer") == 0.0          # the answer still fits what is left (4000)


async def test_without_headers_a_sliding_window_against_the_tpm_limit_is_used():
    now = [1000.0]
    b = GroqBudget(FakeRedis(), tpm_limit=8000, answer_reserve=4000, clock=lambda: now[0])
    assert await b.acquire(M20, 5000, "answer") == 0.0
    wait = await b.acquire(M20, 5000, "answer")
    assert wait == pytest.approx(60.0)                          # the first claim leaves the window in 60 s
    now[0] += 61
    assert await b.acquire(M20, 5000, "answer") == 0.0


async def test_redis_down_falls_back_to_the_local_window_and_never_raises():
    class Broken(FakeRedis):
        async def get(self, k):
            raise ConnectionError("down")

        async def zremrangebyscore(self, *a):
            raise ConnectionError("down")

        async def zadd(self, *a, **k):
            raise ConnectionError("down")

    b = GroqBudget(Broken(), tpm_limit=8000)
    assert await b.acquire(M20, 6000, "answer") == 0.0
    assert await b.acquire(M20, 6000, "answer") > 0


# --- the chain -------------------------------------------------------------------------------------------------------

class Adapter:
    def __init__(self, name, waits=(0.0,), text="ok"):
        self.name, self.waits, self.text, self.calls = name, list(waits), text, 0
        self.provider_name = self.model_name = name

    async def budget_wait(self, messages, params=None):
        return self.waits.pop(0) if len(self.waits) > 1 else self.waits[0]

    async def stream(self, messages, params=None):
        self.calls += 1
        yield self.text


def chain(*adapters):
    entries = [ChainEntry(a.name, a, None, None) for a in adapters]
    return ProviderChain(entries, idle_timeout_s=None)


PROMPT = SimpleNamespace(messages=[{"role": "user", "content": "hi"}])


async def run(c):
    params = {}
    out = [t async for t in c.generate_stream(PROMPT, params)]
    return out, params


async def test_an_answer_moves_to_the_next_model_that_has_budget():
    first, second = Adapter("groq:openai/gpt-oss-20b", waits=(24.0,)), Adapter("groq:openai/gpt-oss-120b", text="from 120b")
    out, params = await run(chain(first, second))
    assert out == ["from 120b"] and first.calls == 0
    assert params["_route"]["provider"] == "groq:openai/gpt-oss-120b" and params["_route"]["fallback_used"] is True


async def test_with_no_budget_anywhere_a_short_gap_is_waited_out_once(monkeypatch):
    slept = []

    async def fake_sleep(s):
        slept.append(s)

    monkeypatch.setattr(chain_module.asyncio, "sleep", fake_sleep)
    a, b = Adapter("a", waits=(6.0, 0.0), text="a after the gap"), Adapter("b", waits=(9.0, 9.0))
    out, _ = await run(chain(a, b))
    assert out == ["a after the gap"] and len(slept) == 1 and 6.0 <= slept[0] <= 6.5


async def test_with_no_budget_and_a_long_gap_the_turn_fails_so_billing_releases_the_hold(monkeypatch):
    monkeypatch.setattr(chain_module.asyncio, "sleep", AsyncMock())
    a, b = Adapter("a", waits=(24.0,)), Adapter("b", waits=(40.0,))
    with pytest.raises(AllProvidersFailedError, match="no token budget"):
        await run(chain(a, b))
    assert a.calls == 0 and b.calls == 0
    chain_module.asyncio.sleep.assert_not_awaited()


async def test_adapters_without_a_budget_hook_behave_as_before():
    class Plain:
        name = provider_name = model_name = "plain"

        async def stream(self, messages, params=None):
            yield "plain"

    out, _ = await run(chain(Plain()))
    assert out == ["plain"]


# --- the Groq adapter and the analysis client --------------------------------------------------------------------

async def test_the_adapter_records_the_streams_headers_after_a_call():
    b = GroqBudget(FakeRedis())

    class Stream:
        response = SimpleNamespace(headers=headers(2222, "11s"))

        def __aiter__(self):
            async def gen():
                yield SimpleNamespace(usage=None, x_groq=None, choices=[SimpleNamespace(delta=SimpleNamespace(content="hi"), finish_reason=None)])
            return gen()

    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=Stream())
    adapter = GroqAdapter(api_key="k", model=M120, tpm_limit=8000, client=client, budget=b)
    params = {}
    assert await adapter.budget_wait([{"role": "user", "content": "hello"}], params) == 0.0   # claimed in the window
    assert params["_budget_claimed"] == M120
    assert [t async for t in adapter.stream([{"role": "user", "content": "hello"}], params)] == ["hi"]
    assert (await b.known_remaining(M120))[0] == 2222


async def test_the_adapter_reports_a_short_budget_instead_of_calling_groq():
    b = GroqBudget(FakeRedis(), tpm_limit=8000)
    await b.record_headers(M20, headers(100, "20s"))
    adapter = GroqAdapter(api_key="k", model=M20, tpm_limit=8000, client=MagicMock(), budget=b)
    assert await adapter.budget_wait([{"role": "user", "content": "hello"}], {}) > 0




async def test_analysis_is_skipped_not_blocking_when_the_bucket_is_short():
    b = GroqBudget(FakeRedis(), answer_reserve=4000)
    await b.record_headers(M20, headers(4200, "30s"))           # an analysis (~1.5k + 512) would leave < 4000
    fake_groq = MagicMock()
    fake_groq.chat.completions.create = AsyncMock(side_effect=AssertionError("analysis must not call Groq"))
    client = GroqAnalysisClient("k", model=M20, client=fake_groq, budget=b, max_tokens=512)
    with pytest.raises(AnalysisBudgetSkipped):
        await client.reserve([{"role": "user", "content": "x" * 3000}])
    fake_groq.chat.completions.create.assert_not_awaited()

    analyzer = RequestAnalyzer.__new__(RequestAnalyzer)
    analyzer.groq_client, analyzer.fallback_clients, analyzer.fallback_adapters, analyzer.nvidia_adapter = client, [], [], None
    analyzer.logger = MagicMock()
    analyzer.prompt_builder = SimpleNamespace(build_prompt=lambda ctx, tools: [{"role": "user", "content": "x" * 3000}])
    before = ANALYSIS_SKIPPED._value.get()
    plan = await analyzer.analyze(SimpleNamespace(conversation_id="c1"), None)
    assert plan is not None and ANALYSIS_SKIPPED._value.get() == before + 1
    assert any(c.args and c.args[0] == "analysis_skipped_budget" for c in analyzer.logger.info.call_args_list)


async def test_analysis_asks_for_512_tokens():
    b = GroqBudget(FakeRedis(), tpm_limit=0)
    create = AsyncMock(return_value=SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20, total_tokens=120),
        choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))]))
    client_obj = MagicMock()
    client_obj.chat.completions.create = create
    client = GroqAnalysisClient("k", model=M20, client=client_obj, budget=b, max_tokens=512)
    assert await client.complete([{"role": "user", "content": "x"}]) == "{}"
    assert create.await_args.kwargs["max_tokens"] == 512
