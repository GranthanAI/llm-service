"""Smart mode searches the web only when the question needs current or external facts (no model call)."""

from datetime import UTC, datetime

import pytest

from app.workflow_engine.langgraph_workflows.smart.freshness import needs_web_search

NOW = datetime(2026, 10, 28, tzinfo=UTC)


@pytest.mark.parametrize("q", [
    "Embedding model kya hota hai, explain with example",       # the real question that was wrongly searched
    "What is a Python dictionary?",
    "Explain how backpropagation works step by step",
    "difference between a list and a tuple in python code",
    "Why is the sky blue?",
    "Summarize this paragraph for me please",
    "",
])
def test_stable_questions_do_not_search(q):
    assert needs_web_search(q, NOW) == (False, "none")


@pytest.mark.parametrize("q,reason", [
    ("What changed in the latest Python release?", "current"),
    ("current pricing of the OpenAI API", "current"),
    ("who is the current CEO of Nvidia", "current"),
    ("news about Mars missions this week", "current"),
    ("search for GPU shortages", "explicit"),
    ("give me sources and links for transformers", "explicit"),
    ("is https://example.com/a safe", "explicit"),
    ("Tell me about GPT-6 Astra", "fast_moving"),
    ("how does Claude 5 handle tools", "fast_moving"),
    ("latest arxiv papers on diffusion", "current"),
    ("survey of long-context benchmarks", "papers"),
    ("EU AI Act rules in 2026", "recent_year"),
])
def test_time_sensitive_or_explicit_questions_search(q, reason):
    search, why = needs_web_search(q, NOW)
    assert search is True and why == reason


def test_old_years_do_not_trigger():
    assert needs_web_search("what happened in 2015 in chess", NOW) == (False, "none")
