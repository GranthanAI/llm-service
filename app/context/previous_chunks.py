"""Chunk ids shown in the last assistant turn of each conversation, so "check again" / "explain point 3" can reuse them.

Process-local and bounded (LRU + TTL). A restart or a different llm-service replica simply means no previous ids for one
turn: retrieval then behaves exactly as before. Only ids are kept, never document text."""

from __future__ import annotations

import time
from collections import OrderedDict


class PreviousChunkStore:
    def __init__(self, ttl_s: float = 3600.0, max_conversations: int = 2000, max_ids: int = 20):
        self.ttl_s, self.max_conversations, self.max_ids = ttl_s, max_conversations, max_ids
        self._items: OrderedDict[str, tuple[float, list[str]]] = OrderedDict()

    def record(self, conversation_id: str, chunk_ids: list[str]) -> None:
        ids = list(dict.fromkeys(i for i in chunk_ids if i))[: self.max_ids]
        if not conversation_id or not ids:      # an empty turn keeps the earlier ids
            return
        self._items[conversation_id] = (time.monotonic(), ids)
        self._items.move_to_end(conversation_id)
        while len(self._items) > self.max_conversations:
            self._items.popitem(last=False)

    def get(self, conversation_id: str) -> list[str]:
        item = self._items.get(conversation_id)
        if not item:
            return []
        if time.monotonic() - item[0] > self.ttl_s:
            del self._items[conversation_id]
            return []
        return list(item[1])
