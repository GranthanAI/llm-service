"""
History compaction: shrink the conversation history of ONE request so it fits a token budget, without deleting
anything permanently.

Design (see docs/CONTEXT_BUDGET.md):
  * Pure and deterministic. It works on the canonical history that was loaded for this request and returns a NEW list;
    nothing is persisted and no model is called, so there is no extra latency or failure mode on the request path and
    repeated requests never stack summaries on top of summaries (each request re-derives the condensed view from the
    canonical messages).
  * The most recent `min_recent` messages are kept verbatim (more when the budget allows). The latest user request is
    not part of the history and is never touched.
  * Older turns are not blindly dropped: they are condensed into one compact "earlier turns" note (one clipped line per
    message, oldest lines dropped first when even the note must shrink). When Memory Service already supplies a
    conversation summary, the note is kept small because the summary covers the older part.
  * Roles and message order are preserved; the note is a `system`-role message placed before the kept messages.
"""

from dataclasses import dataclass, field

from app.context.schemas import Message, Role
from app.context_window.token_counter import TokenCounter

NOTE_HEADER = "Earlier in this conversation (condensed; some detail is omitted):"
LINE_CLIP_CHARS = 180


@dataclass
class CompactionResult:
    messages: list[Message]
    kept: int = 0                    # verbatim messages
    condensed: int = 0               # older messages represented by the note
    omitted: int = 0                 # older messages that did not fit even in the note
    tokens_before: int = 0
    tokens_after: int = 0
    note_added: bool = False
    changed: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self.changed = self.condensed > 0 or self.omitted > 0


def _tokens(counter: TokenCounter, provider: str, message: Message) -> int:
    return counter.count_text(message.content, provider) + 4


def _clip(text: str, limit: int) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "…"


def compact_history(
    messages: list[Message],
    *,
    token_budget: int,
    min_recent: int,
    counter: TokenCounter,
    provider: str,
    note_share: float = 0.3,
    has_memory_summary: bool = False,
) -> CompactionResult:
    """Fit `messages` into `token_budget` tokens (best effort; the caller re-counts and trims further if needed)."""
    before = sum(_tokens(counter, provider, m) for m in messages)
    if before <= token_budget or not messages:
        return CompactionResult(list(messages), kept=len(messages), tokens_before=before, tokens_after=before)

    # 1. Keep the newest messages verbatim: at least min_recent, then as many more as fit.
    keep_from = len(messages)
    used = 0
    note_reserve = int(token_budget * (0.1 if has_memory_summary else note_share))
    for index in range(len(messages) - 1, -1, -1):
        cost = _tokens(counter, provider, messages[index])
        recent_count = len(messages) - index
        if recent_count > min_recent and used + cost > token_budget - note_reserve:
            break
        used += cost
        keep_from = index
    kept = messages[keep_from:]
    older = messages[:keep_from]
    if not older:
        return CompactionResult(list(messages), kept=len(messages), tokens_before=before, tokens_after=before)

    # 2. Condense the older turns into one note that fits what is left.
    note_budget = max(0, token_budget - used)
    lines = [f"{m.role.value.capitalize()}: {_clip(m.content, LINE_CLIP_CHARS)}" for m in older if m.role in (Role.USER, Role.ASSISTANT)]
    chosen: list[str] = []
    header_cost = counter.count_text(NOTE_HEADER, provider) + 4
    spent = header_cost
    for line in reversed(lines):                         # newest older turns first; the oldest are dropped first
        cost = counter.count_text(line, provider) + 1
        if spent + cost > note_budget:
            break
        chosen.append(line)
        spent += cost
    chosen.reverse()

    result_messages = list(kept)
    note_added = False
    if chosen and spent <= note_budget:
        note = Message(role=Role.SYSTEM, content=NOTE_HEADER + "\n" + "\n".join(chosen))
        result_messages = [note, *kept]
        note_added = True
    after = sum(_tokens(counter, provider, m) for m in result_messages)
    return CompactionResult(
        result_messages, kept=len(kept), condensed=len(chosen), omitted=len(older) - len(chosen),
        tokens_before=before, tokens_after=after, note_added=note_added)
