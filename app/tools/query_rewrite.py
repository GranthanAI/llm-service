"""
Turn a conversational follow-up into a standalone web-search query.

"Check again and tell" is meaningless to a search engine on its own; with the conversation it means "look
up <the topic we were discussing>". Self-contained questions are passed through untouched (no model call).
"""

import asyncio
import json
import re
from typing import Any

import structlog

from app.config.logging import get_logger
from app.context.schemas import Message

# Words/phrases that signal the message leans on earlier turns.
_FOLLOWUP = re.compile(
    r"\b(this|that|these|those|it|its|they|them|again|more|previous|above|earlier|same|continue|also|"
    r"instead|else|too|another|recheck|re-check|check)\b|^(and|but|so|then|also|ok|okay)\b",
    re.IGNORECASE,
)
_MAX_HISTORY_TURNS = 6
_MAX_TURN_CHARS = 400
_MAX_QUERY_CHARS = 200

_SYSTEM = (
    "You write web search queries. Given a conversation and the user's last message, rewrite the last message "
    "as ONE standalone search query that makes sense without the conversation. Keep it under 15 words, keep "
    "names and technical terms, add the topic being discussed when the message refers back to it. "
    'Reply with JSON only: {"query": "<search query>"}.'
)


def needs_context(message: str) -> bool:
    """True when the message is too short or too referential to search for as-is."""
    text = message.strip()
    return len(text.split()) < 7 or bool(_FOLLOWUP.search(text))


def _clean(query: str) -> str:
    return " ".join(str(query).replace("\n", " ").split())[:_MAX_QUERY_CHARS].strip()


def _topic(text: str) -> str:
    """The first sentence of an earlier user message, capped: that is where people state their topic."""
    first = re.split(r"(?<=[?.!])\s", " ".join(text.split()), maxsplit=1)[0]
    return first[:100].strip()


def fallback_query(message: str, history: list[Message]) -> str:
    """No model available: prefix the last different user message (the topic) to the follow-up."""
    for m in reversed(history):
        if m.role.value == "user" and m.content.strip() and m.content.strip() != message.strip():
            return _clean(f"{_topic(m.content)} {message.strip()}")
    return _clean(message)


class SearchQueryRewriter:
    def __init__(
        self,
        groq_client: Any | None,
        timeout_s: float = 3.0,
        logger: structlog.stdlib.BoundLogger | None = None,
        fallback_clients: list[Any] | None = None,
        fallback_adapters: list[Any] | None = None,
    ):
        self.groq_client = groq_client
        self.fallback_clients = list(fallback_clients or [])
        self.fallback_adapters = list(fallback_adapters or [])
        self.timeout_s = timeout_s
        self.logger = logger or get_logger("search_query_rewriter")

    async def rewrite(self, message: str, history: list[Message]) -> str:
        """Always returns a usable query; never raises."""
        message = message.strip()
        turns = [m for m in history if m.role.value in ("user", "assistant") and m.content.strip()]
        if not turns or not needs_context(message):
            return message

        clients = [c for c in [self.groq_client, *self.fallback_clients] if c is not None]
        if clients:
            convo = "\n".join(f"{m.role.value}: {m.content.strip()[:_MAX_TURN_CHARS]}" for m in turns[-_MAX_HISTORY_TURNS:])
            prompt = [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": f"Conversation:\n{convo}\n\nLast user message: {message}"},
            ]
            for client in clients:
                try:
                    raw = await asyncio.wait_for(client.complete(messages=prompt), timeout=self.timeout_s)
                    query = _clean(json.loads(raw).get("query", ""))
                    if query:
                        self.logger.info("search_query_rewritten", method="model", model=getattr(client, "model", "?"), changed=query != message)
                        return query
                except Exception as exc:  # timeout, rate limit, bad JSON: try the next model, then the heuristic
                    self.logger.warning("search_query_rewrite_failed", model=getattr(client, "model", "?"), error=type(exc).__name__)

        if self.fallback_adapters and turns:
            convo = "\n".join(f"{m.role.value}: {m.content.strip()[:_MAX_TURN_CHARS]}" for m in turns[-_MAX_HISTORY_TURNS:])
            prompt = [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": f"Conversation:\n{convo}\n\nLast user message: {message}"},
            ]
            for adapter in self.fallback_adapters:
                try:
                    resp = await asyncio.wait_for(adapter.execute(prompt), timeout=self.timeout_s * 2)
                    raw = resp.content
                    query = _clean(json.loads(raw[raw.find("{") : raw.rfind("}") + 1]).get("query", ""))
                    if query:
                        self.logger.info("search_query_rewritten", method="model", model=getattr(resp, "model", "?"), changed=query != message)
                        return query
                except Exception as exc:
                    self.logger.warning("search_query_rewrite_failed", provider=getattr(adapter, "provider_name", "?"), error=type(exc).__name__)

        query = fallback_query(message, turns)
        self.logger.info("search_query_rewritten", method="heuristic", changed=query != message)
        return query
