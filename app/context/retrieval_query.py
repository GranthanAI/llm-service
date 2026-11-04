"""
Standalone retrieval query for document questions.

"Check again" or "Phase 6 mein kya karna hai?" embed badly on their own. This builds, from the current message and the last
few turns:
  query        the question in the user's language, made standalone (keyword leg of retrieval)
  dense_query  an English version of that question (dense/embedding leg); None when it adds nothing

A fast model does the rewrite under a short timeout; any failure falls back to a heuristic (short or content-free message ->
append the previous user question). Never raises. Message text is never logged.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any

import structlog

from app.config.logging import get_logger

_MAX_TURN_CHARS = 400
_MAX_QUERY_CHARS = 300
_SHORT_WORDS = 4   # fewer words than this = "short"

# Words that carry no topic: English function words, follow-up verbs, and common romanised-Hindi function words.
_STOP = frozenset("""
a an the is are was were be been am do does did done have has had it its this that these those they them he she we you i me my
of in on at to for from by with about as and or but so then also too if not no yes ok okay please pls can could would should will
what which who whom whose when where why how tell me show give explain describe check recheck re-check again more another
other same previous above earlier last next continue go ahead try once one point points part thing things stuff
kya hai hain ho tha thi the mein me main se ka ki ke ko par pe aur ya nahi nahin kaise kaun kab kahan kyun kyu karna karo
kar kare batao bata bolo dikhao samjhao samjha fir phir se wapas dobara ek aur iska iski iske uska uski uske yeh ye woh wo
""".split())

_SYSTEM = (
    "You rewrite a user's last message into a standalone document-search query. Use the conversation only to resolve "
    "references (it, again, that, point 3). Keep names, numbers, identifiers and technical terms exactly. Reply with JSON "
    'only: {"standalone": "<the question in the user\'s own language, understandable without the conversation>", '
    '"english": "<the same question in English>"}. Maximum 25 words each. Do not answer the question.'
)


@dataclass
class RetrievalQuery:
    query: str                      # sent as GetRelevantChunksRequest.query
    dense_query: str | None = None  # sent as dense_query (English); None = not sent
    method: str = "none"            # none | model | heuristic | disabled
    rewritten: bool = False         # True when query differs from the user's message or dense_query is set


def content_words(text: str) -> list[str]:
    words = re.findall(r"[\w][\w\-]*", text.lower())
    return [w for w in words if w not in _STOP and (len(w) > 2 or w.isdigit() or any(c.isdigit() for c in w))]


def needs_context(message: str) -> bool:
    """Short, or nothing in it names a topic: the message leans on the earlier turns."""
    words = re.findall(r"\w+", message)
    return len(words) < _SHORT_WORDS or not content_words(message)


def _clean(text: object) -> str:
    return " ".join(str(text).replace("\n", " ").split())[:_MAX_QUERY_CHARS].strip()


def heuristic_query(message: str, history: list[tuple[str, str]]) -> str:
    """Append the previous (different) user question when the message cannot stand alone."""
    message = message.strip()
    if not needs_context(message):
        return message
    for role, content in reversed(history):
        text = " ".join(content.split())
        if role == "user" and text and text != message:
            return _clean(f"{text[:200]} {message}")
    return message


class RetrievalQueryRewriter:
    def __init__(
        self,
        clients: list[Any] | None = None,
        adapters: list[Any] | None = None,
        timeout_s: float = 2.0,
        turns: int = 3,
        logger: structlog.stdlib.BoundLogger | None = None,
    ):
        self.clients = [c for c in (clients or []) if c is not None]
        self.adapters = list(adapters or [])
        self.timeout_s = timeout_s
        self.turns = turns
        self.logger = logger or get_logger("retrieval_query_rewriter")

    def _window(self, history: list[tuple[str, str]]) -> list[tuple[str, str]]:
        turns = [(r, c) for r, c in history if r in ("user", "assistant") and c.strip()]
        return turns[-2 * self.turns:] if self.turns else []

    async def build(self, message: str, history: list[tuple[str, str]]) -> RetrievalQuery:
        message = message.strip()
        window = self._window(history)
        prompt = None
        if message and (self.clients or self.adapters):
            convo = "\n".join(f"{r}: {c.strip()[:_MAX_TURN_CHARS]}" for r, c in window)
            prompt = [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": f"Conversation:\n{convo or '(none)'}\n\nLast user message: {message}"},
            ]
        if prompt:
            parsed = await self._ask_model(prompt)
            if parsed:
                standalone, english = parsed
                standalone = standalone or message
                dense = english if english and english.lower() != standalone.lower() else None
                if dense is None and standalone.lower() != message.lower():
                    dense = standalone
                return RetrievalQuery(standalone, dense, "model", standalone != message or dense is not None)

        query = heuristic_query(message, window)
        return RetrievalQuery(query, None, "heuristic", query != message)

    async def _ask_model(self, prompt: list[dict]) -> tuple[str, str] | None:
        for client in self.clients:
            try:
                raw = await asyncio.wait_for(client.complete(messages=prompt), timeout=self.timeout_s)
                got = self._parse(raw)
                if got:
                    return got
            except Exception as exc:  # timeout, rate limit, bad JSON: next model, then the heuristic
                self.logger.warning("retrieval_rewrite_failed", model=getattr(client, "model", "?"), error=type(exc).__name__)
        for adapter in self.adapters:
            try:
                resp = await asyncio.wait_for(adapter.execute(prompt), timeout=self.timeout_s)
                got = self._parse(resp.content)
                if got:
                    return got
            except Exception as exc:
                self.logger.warning("retrieval_rewrite_failed", provider=getattr(adapter, "provider_name", "?"), error=type(exc).__name__)
        return None

    @staticmethod
    def _parse(raw: Any) -> tuple[str, str] | None:
        text = raw if isinstance(raw, str) else str(raw)
        data = json.loads(text[text.find("{"): text.rfind("}") + 1])
        standalone, english = _clean(data.get("standalone", "")), _clean(data.get("english", ""))
        return (standalone, english) if (standalone or english) else None
