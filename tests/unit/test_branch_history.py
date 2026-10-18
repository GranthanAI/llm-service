"""Ancestor turns inherited by a branch reach the model, ahead of the branch's own turns."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from app.context.schemas import ContextBundle, MemoryContext, Message, Role
from app.models.request import HistoryMessage
from app.pipeline.request_pipeline import RequestPipeline


def _pipeline():
    p = RequestPipeline.__new__(RequestPipeline)
    p.logger = MagicMock()
    return p


def _ctx(bundle, inherited=(), own=()):
    return SimpleNamespace(
        context_bundle=bundle, inherited_history=list(inherited), event_history=list(own),
        conversation_id="c", request_id="r")


def _texts(ctx):
    return [m.content for m in ctx.context_bundle.memory.short_term_messages]


def test_event_history_marks_inherited_turns():
    h = HistoryMessage(role="user", content="x", inherited=True, source_conversation_id="a")
    assert h.inherited and h.source_conversation_id == "a"
    assert HistoryMessage(role="user", content="y").inherited is False   # older producers


def test_inherited_turns_come_before_the_memory_services_own_turns():
    mem = MemoryContext(short_term_messages=[Message(role=Role.USER, content="own question")])
    ctx = _ctx(ContextBundle(memory=mem), inherited=[("user", "A1"), ("assistant", "A2")])
    _pipeline()._ensure_conversation_history(ctx)
    assert _texts(ctx) == ["A1", "A2", "own question"]


def test_without_memory_history_the_event_supplies_inherited_then_own():
    ctx = _ctx(None, inherited=[("user", "A1")], own=[("user", "own")])
    _pipeline()._ensure_conversation_history(ctx)
    assert _texts(ctx) == ["A1", "own"]


def test_a_conversation_without_ancestors_is_unchanged():
    mem = MemoryContext(short_term_messages=[Message(role=Role.USER, content="own")])
    ctx = _ctx(ContextBundle(memory=mem))
    _pipeline()._ensure_conversation_history(ctx)
    assert _texts(ctx) == ["own"]
