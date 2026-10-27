"""Smart mode: when does a request need the web?

A deterministic, testable policy (no model call). Stable, self-contained questions ("what is an embedding model", "explain
Python dictionaries") are answered without browsing. A web search is planned only when the answer depends on facts that change
or that the model may not know:

  * the user asks for current information (latest, recent, today, this year, news, price, release, status, a current office-holder),
  * the user explicitly asks to search / look up / cite sources / give links, or pastes a URL,
  * the topic is a fast-moving named product or model family (GPT, Claude, Gemini, ... with a version) or a research paper,
  * the text mentions a recent year (the model's knowledge may predate it).

Returns the decision together with a short reason code for logs and metrics (never user text).
"""

import re
from datetime import UTC, datetime

_CURRENT = re.compile(
    r"\b(latest|newest|current(ly)?|recent(ly)?|right now|as of|up[- ]to[- ]date|today|tonight|yesterday|tomorrow|"
    r"this (week|month|year|quarter)|last (week|month|year)|breaking|trending|news|headlines?|"
    r"price|pricing|cost of|how much (is|does|do)|release[sd]?|launch(ed)?|announce[ds]?|changelog|roadmap|"
    r"outage|status of|down for|score|weather|stock|exchange rate|election|who is the (current )?"
    r"(ceo|president|prime minister|founder|head|owner)|new version|what'?s new)\b",
    re.IGNORECASE,
)
_EXPLICIT = re.compile(
    r"(\b(search|google|look ?up|browse|web|online|internet|source|sources|cite|citation|reference|references|link|links|"
    r"url|verify|fact[- ]?check)\b|https?://)",
    re.IGNORECASE,
)
# Fast-moving model/product families (a version number makes the question time-sensitive).
_FAST_MOVING = re.compile(
    r"\b(gpt|claude|gemini|llama|qwen|deepseek|mistral|grok|o[134]|sora|copilot|chatgpt|nemotron|gemma|phi)[- ]?\d",
    re.IGNORECASE,
)
_PAPERS = re.compile(r"\b(arxiv|papers?|preprints?|benchmarks?|sota|state of the art)\b", re.IGNORECASE)
_YEAR = re.compile(r"\b(20\d{2})\b")


def needs_web_search(message: str, now: datetime | None = None) -> tuple[bool, str]:
    """(search?, reason). reason is one of: none, current, explicit, fast_moving, papers, recent_year."""
    text = (message or "").strip()
    if not text:
        return False, "none"
    if _EXPLICIT.search(text):
        return True, "explicit"
    if _CURRENT.search(text):
        return True, "current"
    if _FAST_MOVING.search(text):
        return True, "fast_moving"
    if _PAPERS.search(text):
        return True, "papers"
    this_year = (now or datetime.now(UTC)).year
    if any(int(y) >= this_year - 1 for y in _YEAR.findall(text)):
        return True, "recent_year"
    return False, "none"
