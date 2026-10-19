"""Configurable context budget, automatic compaction, per-provider budgets, fallback re-validation and 429 handling.
All local: no model or network calls."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from app.config.settings import LLMServiceConfig
from app.context.schemas import ContextBundle, MemoryContext, Message, Role
from app.context_window.budget_policy import ContextBudgetError, resolve_context_budget
from app.context_window.compaction import NOTE_HEADER, compact_history
from app.context_window.fit import build_chain_fitter, fit_prompt
from app.context_window.manager import ContextWindowManager
from app.context_window.models import ModelLimits
from app.context_window.token_counter import TokenCounter
from app.exceptions.analysis import ContextOverflowError
from app.prompts.schemas import ComposedPrompt, PromptSection

COUNTER = TokenCounter()


def cfg(**over) -> LLMServiceConfig:
    # no legacy cap and no tokens-per-minute cap unless a test asks for them
    base = dict(_env_file=None, prompt_token_budget=0, groq_tpm_limit=0)
    base.update(over)
    return LLMServiceConfig(**base)


# ------------------------------------------------------------------ configuration


def test_documented_defaults():
    c = LLMServiceConfig(_env_file=None)
    assert (c.context_budget_enabled, c.context_budget_default_tokens, c.context_budget_groq_tokens,
            c.context_budget_gemini_tokens) == (True, 150_000, 150_000, 150_000)
    assert (c.context_compaction_enabled, c.context_compaction_trigger_ratio, c.context_compaction_target_ratio,
            c.context_compaction_min_recent_messages) == (True, 0.85, 0.65, 6)
    assert (c.context_token_safety_margin, c.context_output_token_reserve) == (0.05, 8192)


def test_environment_variables_override_the_defaults(monkeypatch):
    for k, v in {"CONTEXT_BUDGET_DEFAULT_TOKENS": "200000", "CONTEXT_BUDGET_GROQ_TOKENS": "120000",
                 "CONTEXT_COMPACTION_TRIGGER_RATIO": "0.9", "CONTEXT_COMPACTION_TARGET_RATIO": "0.5",
                 "CONTEXT_COMPACTION_MIN_RECENT_MESSAGES": "10", "CONTEXT_TOKEN_SAFETY_MARGIN": "0.1",
                 "CONTEXT_OUTPUT_TOKEN_RESERVE": "4096", "CONTEXT_COMPACTION_ENABLED": "false",
                 "CONTEXT_BUDGET_MODEL_OVERRIDES": '{"groq:openai/gpt-oss-120b": 90000}',
                 "CONTEXT_MODEL_WINDOWS": '{"my/model": 64000}'}.items():
        monkeypatch.setenv(k, v)
    c = LLMServiceConfig(_env_file=None)
    assert (c.context_budget_default_tokens, c.context_budget_groq_tokens, c.context_budget_gemini_tokens) == (200_000, 120_000, 150_000)
    assert (c.context_compaction_trigger_ratio, c.context_compaction_target_ratio, c.context_compaction_min_recent_messages) == (0.9, 0.5, 10)
    assert (c.context_token_safety_margin, c.context_output_token_reserve, c.context_compaction_enabled) == (0.1, 4096, False)
    assert c.context_budget_model_overrides == {"groq:openai/gpt-oss-120b": 90_000} and c.context_model_windows == {"my/model": 64_000}


@pytest.mark.parametrize("bad", [
    {"context_compaction_trigger_ratio": 0}, {"context_compaction_trigger_ratio": 1.01}, {"context_compaction_trigger_ratio": -0.1},
    {"context_compaction_target_ratio": 0}, {"context_compaction_target_ratio": 0.9},                 # target must be < trigger (0.85)
    {"context_compaction_trigger_ratio": 0.6, "context_compaction_target_ratio": 0.6},
    {"context_token_safety_margin": -0.01}, {"context_token_safety_margin": 0.5},
    {"context_budget_default_tokens": -1}, {"context_budget_default_tokens": 10}, {"context_budget_groq_tokens": 0},
    {"context_output_token_reserve": 0}, {"context_output_token_reserve": 150_000},
    {"context_compaction_min_recent_messages": 0}, {"prompt_token_budget": -5},
    {"context_budget_model_overrides": {"m": 5}}, {"context_model_windows": {"": 100000}},
])
def test_invalid_values_fail_at_startup(bad):
    with pytest.raises(ValidationError):
        LLMServiceConfig(_env_file=None, **bad)


def test_every_documented_context_variable_in_env_example_is_a_real_setting():
    text = (Path(__file__).resolve().parents[2] / ".env.example").read_text(encoding="utf-8")
    names = {line.split("=", 1)[0].strip() for line in text.splitlines() if line.startswith("CONTEXT_")}
    assert {"CONTEXT_BUDGET_DEFAULT_TOKENS", "CONTEXT_COMPACTION_TRIGGER_RATIO", "CONTEXT_OUTPUT_TOKEN_RESERVE"} <= names
    fields = {f.upper() for f in LLMServiceConfig.model_fields}
    assert names <= fields, names - fields
    LLMServiceConfig(_env_file=None, **{n.lower(): v for n, v in
                                       (line.split("=", 1) for line in text.splitlines()
                                        if line.startswith("CONTEXT_") and "{" not in line)})     # the shipped values validate


# ------------------------------------------------------------------ budget resolution


KNOWN = ModelLimits(model="m", context_window=200_000, max_output_tokens=8192, provider="groq")


def test_the_formula_is_total_times_margin_minus_reserve():
    b = resolve_context_budget(cfg(), "groq", "m", KNOWN)
    assert b.app_budget == 150_000 and b.total == 150_000 and b.app_budget_source == "provider"
    assert b.input_budget == int(150_000 * 0.95) - 8192 and not b.window_assumed


def test_the_model_window_is_a_second_ceiling():
    small = ModelLimits(model="m", context_window=32_000, max_output_tokens=4096, provider="groq")
    b = resolve_context_budget(cfg(), "groq", "m", small)
    assert b.total == 32_000 and b.input_budget == int(32_000 * 0.95) - 4096


def test_resolution_order_model_override_then_provider_then_default():
    c = cfg(context_budget_default_tokens=100_000, context_budget_groq_tokens=120_000, context_budget_gemini_tokens=None,
            context_budget_model_overrides={"groq:special": 50_000, "bare-model": 60_000})
    assert resolve_context_budget(c, "groq", "special", KNOWN).app_budget == 50_000
    assert resolve_context_budget(c, "gemini", "bare-model", KNOWN).app_budget == 60_000
    assert resolve_context_budget(c, "groq", "other", KNOWN).app_budget == 120_000
    assert resolve_context_budget(c, "gemini", "g", KNOWN).app_budget == 100_000          # gemini unset -> default
    assert resolve_context_budget(c, "nvidia", "n", KNOWN).app_budget == 100_000


def test_gemini_can_have_its_own_budget_without_changing_the_default():
    c = cfg(context_budget_gemini_tokens=300_000)
    huge = ModelLimits(model="g", context_window=1_048_576, max_output_tokens=8192, provider="gemini")
    assert resolve_context_budget(c, "gemini", "g", huge).app_budget == 300_000
    assert resolve_context_budget(c, "groq", "m", KNOWN).app_budget == 150_000
    # a huge window alone never raises the budget
    assert resolve_context_budget(cfg(), "gemini", "g", huge).total == 150_000


def test_an_unknown_model_gets_the_conservative_window_and_is_flagged():
    b = resolve_context_budget(cfg(), "groq", "mystery", None)
    assert b.window_assumed and b.model_window == 32_768 and b.total == 32_768


def test_requested_output_is_reserved_and_the_legacy_cap_only_lowers():
    b = resolve_context_budget(cfg(), "groq", "m", KNOWN, requested_output_tokens=2048)
    assert b.output_reserve == 2048 and b.input_budget == int(150_000 * 0.95) - 2048
    capped = resolve_context_budget(cfg(prompt_token_budget=5000), "groq", "m", KNOWN)
    assert capped.input_budget == 5000 and capped.legacy_cap_applied
    assert resolve_context_budget(cfg(prompt_token_budget=10_000_000), "groq", "m", KNOWN).legacy_cap_applied is False


def test_disabling_the_budget_leaves_only_the_model_window():
    b = resolve_context_budget(cfg(context_budget_enabled=False), "groq", "m", KNOWN)
    assert b.total == 200_000 and b.app_budget_source == "disabled"


def test_a_budget_too_small_for_any_request_is_a_configuration_error():
    tiny = ModelLimits(model="t", context_window=4_000, max_output_tokens=4_000, provider="groq")
    with pytest.raises(ContextBudgetError):
        resolve_context_budget(cfg(), "groq", "t", tiny, requested_output_tokens=3_900)


# ------------------------------------------------------------------ compaction (pure)


def convo(n: int, size: int = 60) -> list[Message]:
    return [Message(role=Role.USER if i % 2 == 0 else Role.ASSISTANT, content=f"turn {i} " + "word " * size) for i in range(n)]


def test_history_that_fits_is_returned_unchanged():
    msgs = convo(4)
    r = compact_history(msgs, token_budget=10_000, min_recent=2, counter=COUNTER, provider="groq")
    assert r.messages == msgs and not r.changed and r.tokens_after == r.tokens_before


def test_older_turns_are_condensed_and_recent_ones_kept_verbatim_in_order():
    msgs = convo(30)
    r = compact_history(msgs, token_budget=1500, min_recent=6, counter=COUNTER, provider="groq")
    assert r.changed and r.tokens_after < r.tokens_before
    kept = [m for m in r.messages if m.role != Role.SYSTEM]
    assert kept == msgs[-len(kept):] and len(kept) >= 6                      # verbatim tail, original order and roles
    note = r.messages[0]
    assert note.role == Role.SYSTEM and note.content.startswith(NOTE_HEADER)
    assert r.condensed + r.omitted + len(kept) == 30                         # nothing unaccounted for
    assert "turn 29" not in note.content or len(kept) < 30                   # newest turns live in the verbatim tail


def test_the_minimum_recent_messages_are_kept_even_when_over_budget():
    msgs = convo(12, size=200)
    r = compact_history(msgs, token_budget=300, min_recent=6, counter=COUNTER, provider="groq")
    assert [m for m in r.messages if m.role != Role.SYSTEM][-6:] == msgs[-6:]


def test_compaction_is_idempotent_on_the_canonical_history_and_never_stacks_notes():
    msgs = convo(40)
    once = compact_history(msgs, token_budget=1500, min_recent=6, counter=COUNTER, provider="groq")
    again = compact_history(msgs, token_budget=1500, min_recent=6, counter=COUNTER, provider="groq")
    assert [(m.role, m.content) for m in once.messages] == [(m.role, m.content) for m in again.messages]   # deterministic
    assert sum(m.content.startswith(NOTE_HEADER) for m in once.messages) == 1
    longer = msgs + convo(2)
    grown = compact_history(longer, token_budget=1500, min_recent=6, counter=COUNTER, provider="groq")
    assert sum(m.content.startswith(NOTE_HEADER) for m in grown.messages) == 1  # new messages arrive, still one note
    assert grown.messages[-1] is longer[-1]


def test_a_memory_summary_keeps_the_condensed_note_small():
    msgs = convo(40)
    with_summary = compact_history(msgs, token_budget=2000, min_recent=6, counter=COUNTER, provider="groq", has_memory_summary=True)
    without = compact_history(msgs, token_budget=2000, min_recent=6, counter=COUNTER, provider="groq")
    assert with_summary.condensed <= without.condensed


# ------------------------------------------------------------------ fitting a whole prompt


class FakeBuilder:
    """Builds a prompt shaped like PromptBuilder's: locked system + user_query, trimmable history and graph."""

    def __init__(self, graph_words=0, system_words=20, query="what did we decide about caching?"):
        self.calls = 0
        self.graph_words, self.system_words, self.query = graph_words, system_words, query

    def build(self, result, ctx):
        self.calls += 1
        history = result.conversation_history or []
        hist_text = "\n".join(f"{m.role.value}: {m.content}" for m in history)
        parts = {"system": "sys " * self.system_words, "graph_context": "g " * self.graph_words,
                 "conversation_history": hist_text, "user_query": self.query}
        prio = {"system": 10, "graph_context": 5, "conversation_history": 8, "user_query": 10}
        locked = {"system", "user_query"}
        sections = [PromptSection(name=n, content=t, priority=prio[n], token_count=COUNTER.count_text(t, "groq"),
                                  trimmable=n not in locked) for n, t in parts.items() if t]
        system = "\n\n".join(s.content for s in sections if s.name == "system")
        user = "\n\n".join(s.content for s in sections if s.name != "system")
        return ComposedPrompt(sections=sections, total_tokens=sum(s.token_count for s in sections),
                              messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                              mode="default", engine_type="mode_handler")


def fit(config, builder, history, model="m-test", max_out=1000, summary=""):
    mgr = ContextWindowManager(logger_instance=MagicMock())
    mgr.apply_config(config)
    mgr.register_model_limits(model, ModelLimits(model=model, context_window=50_000, max_output_tokens=2000, provider="groq"))
    result = SimpleNamespace(conversation_history=list(history))
    memory = MemoryContext(short_term_messages=list(history), summary=summary)
    ctx = SimpleNamespace(context_bundle=ContextBundle(memory=memory))
    composed = builder.build(result, ctx)
    trimmed, report = fit_prompt(config=config, manager=mgr, prompt_builder=builder, result=result, ctx=ctx, composed=composed,
                                 provider="groq", model=model, max_output_tokens=max_out, logger=MagicMock())
    return trimmed, report, result, ctx


SMALL = dict(context_budget_groq_tokens=6_000, context_output_token_reserve=1_000)       # total 6000 -> input 6000*0.95-1000 = 4700; trigger 3995; target 3055


def test_a_request_below_the_trigger_is_not_compacted():
    builder = FakeBuilder()
    trimmed, report, result, _ = fit(cfg(**SMALL), builder, convo(4, size=30))
    assert report.compaction == "skipped" and builder.calls == 1 and report.tokens_after < 3995
    assert len(result.conversation_history) == 4


def test_a_request_above_the_trigger_is_compacted_and_recounted():
    builder = FakeBuilder()
    history = convo(40, size=120)
    trimmed, report, result, ctx = fit(cfg(**SMALL), builder, history)
    assert report.compaction == "compacted" and report.messages_condensed + report.messages_omitted > 0
    assert builder.calls == 2                                                # rebuilt after compaction = recounted
    assert report.tokens_before > 3995 and report.tokens_after <= report.budget.input_budget
    assert report.tokens_after == COUNTER.count_messages(trimmed.messages, "groq")       # final serialized payload counted
    assert trimmed.messages[1]["content"].endswith("what did we decide about caching?")  # the latest request is verbatim
    assert result.conversation_history[-6:] == history[-6:] and ctx.context_bundle.memory.short_term_messages[-6:] == history[-6:]


def test_a_request_over_the_hard_budget_compacts_even_if_the_trigger_ratio_is_not_met():
    config = cfg(**SMALL, context_compaction_trigger_ratio=1.0, context_compaction_target_ratio=0.5)
    trimmed, report, *_ = fit(config, FakeBuilder(), convo(60, size=120))
    assert report.compaction == "compacted" and report.tokens_after <= report.budget.input_budget


def test_the_output_reserve_and_safety_margin_are_applied():
    _, report, *_ = fit(cfg(**SMALL), FakeBuilder(), convo(2), max_out=1000)
    assert report.budget.output_reserve == 1000 and report.budget.input_budget == int(6000 * 0.95) - 1000


def test_optional_context_is_trimmed_when_history_alone_cannot_fix_it():
    builder = FakeBuilder(graph_words=20_000)
    trimmed, report, *_ = fit(cfg(**SMALL), builder, convo(8, size=20))
    assert "graph_context" in report.trimmed_sections and report.tokens_after <= report.budget.input_budget


def test_a_request_that_cannot_fit_is_rejected_not_sent_and_the_system_and_user_stay_intact():
    builder = FakeBuilder(query="q " * 20_000)                               # the locked user request alone is too big
    with pytest.raises(ContextOverflowError):
        fit(cfg(**SMALL), builder, convo(4))


def test_compaction_disabled_still_enforces_the_budget_by_trimming():
    builder = FakeBuilder(graph_words=20_000)
    _, report, result, _ = fit(cfg(**SMALL, context_compaction_enabled=False), builder, convo(40, size=120))
    assert report.compaction == "skipped" and builder.calls == 1 and report.tokens_after <= report.budget.input_budget
    assert len(result.conversation_history) == 40                            # history was not compacted


def test_a_compaction_failure_never_breaks_the_request(monkeypatch):
    import app.context_window.fit as fit_module

    monkeypatch.setattr(fit_module, "compact_history", MagicMock(side_effect=RuntimeError("boom")))
    _, report, *_ = fit(cfg(**SMALL), FakeBuilder(graph_words=20_000), convo(40, size=120))
    assert report.compaction == "failed" and report.tokens_after <= report.budget.input_budget


def test_unknown_model_uses_the_conservative_window_and_still_fits():
    config = cfg()
    mgr = ContextWindowManager(logger_instance=MagicMock())
    mgr.apply_config(config)
    builder = FakeBuilder(graph_words=60_000)
    result, ctx = SimpleNamespace(conversation_history=[]), SimpleNamespace(context_bundle=ContextBundle())
    trimmed, report = fit_prompt(config=config, manager=mgr, prompt_builder=builder, result=result, ctx=ctx,
                                 composed=builder.build(result, ctx), provider="groq", model="never-seen", max_output_tokens=2048,
                                 logger=MagicMock())
    assert report.budget.window_assumed and report.tokens_after <= int(32_768 * 0.95) - 2048


def test_repeated_requests_on_a_growing_conversation_do_not_stack_summaries():
    history = convo(40, size=120)
    notes = []
    for extra in range(3):
        _, _, result, _ = fit(cfg(**SMALL), FakeBuilder(), history + convo(extra, size=120))
        notes.append(sum(m.content.startswith(NOTE_HEADER) for m in result.conversation_history))
    assert notes == [1, 1, 1]


# ------------------------------------------------------------------ fallback re-validation


class Adapter:
    def __init__(self, name, fail=None):
        self.name, self.fail, self.seen = name, fail, None

    async def stream(self, messages, params=None):
        self.seen = messages
        if self.fail:
            raise self.fail
        yield f"answer from {self.name}"


def chain(entries, config):
    from app.providers.chain import ChainEntry, ProviderChain

    mgr = ContextWindowManager(logger_instance=MagicMock())
    mgr.apply_config(config)
    mgr.register_model_limits("big", ModelLimits(model="big", context_window=1_000_000, max_output_tokens=8192, provider="gemini"))
    mgr.register_model_limits("gem", ModelLimits(model="gem", context_window=1_000_000, max_output_tokens=8192, provider="gemini"))
    mgr.register_model_limits("gq", ModelLimits(model="gq", context_window=1_000_000, max_output_tokens=8192, provider="groq"))
    mgr.register_model_limits("gq2", ModelLimits(model="gq2", context_window=1_000_000, max_output_tokens=8192, provider="groq"))
    chain_entries = [ChainEntry(name=n, adapter=a, breaker=None, first_token_timeout_s=None) for n, a in entries]
    return ProviderChain(chain_entries, idle_timeout_s=None, fitter=build_chain_fitter(config, mgr, MagicMock())), mgr


async def collect(c, prompt):
    return [t async for t in c.generate_stream(prompt, {})]


def trimmed_of(builder, history, mgr_config):
    mgr = ContextWindowManager(logger_instance=MagicMock())
    mgr.apply_config(mgr_config)
    mgr.register_model_limits("x", ModelLimits(model="x", context_window=5_000_000, max_output_tokens=8192, provider="groq"))
    composed = builder.build(SimpleNamespace(conversation_history=history), None)
    return mgr.manage(composed, target_model="x", custom_input_budget=10**6)


@pytest.mark.asyncio
async def test_a_rate_limited_groq_falls_back_to_gemini_after_revalidating_the_budget():
    from app.exceptions.provider import ProviderRateLimitError

    config = cfg(context_budget_groq_tokens=1_000_000, context_budget_gemini_tokens=20_000, gemini_model="gem")
    groq, gemini = Adapter("groq", fail=ProviderRateLimitError("429", provider="groq")), Adapter("gemini")
    c, _ = chain([("groq:gq", groq), ("gemini", gemini)], config)
    prompt = trimmed_of(FakeBuilder(graph_words=40_000), convo(10, size=40), config)       # ~45k tokens: fine for groq
    out = await collect(c, prompt)
    assert out == ["answer from gemini"]
    sent = COUNTER.count_messages(gemini.seen, "gemini")
    assert sent <= int(20_000 * 0.95) - config.gemini_max_tokens               # gemini's OWN (smaller) budget applied


@pytest.mark.asyncio
async def test_a_provider_whose_budget_cannot_hold_the_mandatory_content_is_skipped_not_sent_an_oversized_request():
    config = cfg(context_budget_groq_tokens=1_000_000, context_budget_gemini_tokens=12_000, gemini_model="gem")
    groq, gemini = Adapter("groq", fail=RuntimeError("down")), Adapter("gemini")
    c, _ = chain([("groq:gq", groq), ("gemini", gemini)], config)
    prompt = trimmed_of(FakeBuilder(query="q " * 6_000), [], config)           # locked content > what gemini's 12k budget leaves for input
    from app.exceptions.provider import AllProvidersFailedError

    with pytest.raises(AllProvidersFailedError) as exc:
        await collect(c, prompt)
    assert gemini.seen is None and "context_too_large" in str(exc.value)


@pytest.mark.asyncio
async def test_a_larger_gemini_window_does_not_disable_the_application_budget():
    config = cfg(context_budget_groq_tokens=1_000_000, context_budget_gemini_tokens=150_000, gemini_model="big")
    c, mgr = chain([("gemini", Adapter("gemini"))], config)
    huge = trimmed_of(FakeBuilder(graph_words=400_000), [], config)           # far over 150k although gemini's window is 1M
    assert COUNTER.count_messages(huge.messages, "gemini") > 150_000
    out = await collect(c, huge)
    assert out == ["answer from gemini"]
    assert COUNTER.count_messages(c.entries[0].adapter.seen, "gemini") <= int(150_000 * 0.95) - config.gemini_max_tokens


@pytest.mark.asyncio
async def test_fallback_never_switches_after_output_started():
    class Partial:
        async def stream(self, messages, params=None):
            yield "partial"
            raise RuntimeError("dropped")

    config = cfg(gemini_model="gem")
    gem = Adapter("gemini")
    c, _ = chain([("groq:gq", Partial()), ("gemini", gem)], config)
    prompt = trimmed_of(FakeBuilder(), [], config)
    seen = []
    with pytest.raises(RuntimeError):
        async for t in c.generate_stream(prompt, {}):
            seen.append(t)
    assert seen == ["partial"] and gem.seen is None                           # no spliced second answer


# ------------------------------------------------------------------ rate limits are not context overflows


def test_rate_limit_parsing_prefers_retry_after_and_ignores_context_errors():
    from app.providers.groq import is_context_limit_error, rate_limit_wait_s

    class Err(Exception):
        def __init__(self, msg, headers=None):
            super().__init__(msg)
            self.response = SimpleNamespace(headers=headers or {})

    assert rate_limit_wait_s(Err("Error code: 429 - rate limit", {"retry-after": "7"})) == 7.0
    assert rate_limit_wait_s(Err("429 Rate limit reached. Please try again in 2.5s")) == 2.5
    assert rate_limit_wait_s(Err("429 Rate limit reached")) == 60.0                       # unknown wait: treated as long
    ctx_err = Err("Error code: 400 - context_length_exceeded: maximum context length is 131072 tokens")
    assert is_context_limit_error(ctx_err) and rate_limit_wait_s(ctx_err) is None
    assert rate_limit_wait_s(Err("500 internal")) is None


def test_a_context_size_rejection_is_not_reported_as_a_rate_limit():
    from app.exceptions.provider import ProviderError, ProviderRateLimitError
    from app.providers.groq import GroqAdapter

    adapter = GroqAdapter.__new__(GroqAdapter)
    adapter.logger = MagicMock()
    with pytest.raises(ProviderError) as ctx_exc:
        adapter._handle_exception(Exception("400 context_length_exceeded"))
    assert not isinstance(ctx_exc.value, ProviderRateLimitError)
    with pytest.raises(ProviderRateLimitError):
        adapter._handle_exception(Exception("429 rate limit reached"))


def test_by_default_nothing_caps_the_input_below_the_context_budget_except_a_paid_tier_having_no_tpm_limit():
    c = LLMServiceConfig(_env_file=None, groq_tpm_limit=0)
    assert c.prompt_token_budget == 0
    b = resolve_context_budget(c, "groq", "m", KNOWN)
    assert b.legacy_cap_applied is False and b.tpm_cap_applied is False and b.input_budget == int(150_000 * 0.95) - 8192
    shipped = {l.split("=", 1)[0]: l.split("=", 1)[1].strip() for l in
               (Path(__file__).resolve().parents[2] / ".env.example").read_text(encoding="utf-8").splitlines()
               if l and not l.startswith("#") and "=" in l}
    assert shipped["PROMPT_TOKEN_BUDGET"] == "0"


def test_a_tokens_per_minute_limit_keeps_room_for_the_whole_answer():
    """Groq counts prompt + max_tokens against the per-minute limit: without this the answer is squeezed to the floor
    and cut off (observed: max_tokens=1024 on an 8000 TPM account)."""
    c = LLMServiceConfig(_env_file=None, groq_tpm_limit=8000)
    b = resolve_context_budget(c, "groq", "m", KNOWN, requested_output_tokens=2048)
    assert b.tpm_cap_applied and b.input_budget == 8000 - 2048 - 200 and b.input_budget + b.output_reserve < 8000
    gemini = resolve_context_budget(c, "gemini", "g", KNOWN, requested_output_tokens=2048)
    assert gemini.tpm_cap_applied is False and gemini.input_budget > 100_000          # the limit is Groq's, not Gemini's
    paid = resolve_context_budget(LLMServiceConfig(_env_file=None, groq_tpm_limit=0), "groq", "m", KNOWN, requested_output_tokens=2048)
    assert paid.tpm_cap_applied is False and paid.input_budget > 100_000


def test_the_prompt_estimate_for_the_tpm_arithmetic_is_not_wildly_pessimistic():
    from app.providers.groq import estimate_prompt_tokens

    text = "The quick brown fox jumps over the lazy dog. " * 400
    real = COUNTER.count_text(text, "groq")
    assert real <= estimate_prompt_tokens([{"role": "user", "content": text}]) <= real * 1.2
