"""Turns prompt sections into OpenAI-style chat messages.

Without `history_as_messages` the result is the original shape: one system message and one user message holding every
other section (history flattened into text). With it, the conversation history is sent as real role-separated
messages, followed by one user message holding the remaining sections (context, retrieved passages, the question).
"""
from typing import Any

from app.prompts.schemas import PromptSection

HISTORY_SECTION = "conversation_history"
_ROLES = ("user", "assistant")


def _append(messages: list[dict[str, Any]], role: str, content: str) -> None:
    """Appends a message, merging it into the previous one of the same role (providers reject or mangle two
    consecutive messages of one role)."""
    if messages and messages[-1]["role"] == role:
        messages[-1]["content"] += "\n\n" + content
    else:
        messages.append({"role": role, "content": content})


def history_messages(section: PromptSection) -> list[dict[str, Any]]:
    """The section's messages as chat messages, oldest first. A leading assistant message is dropped: chat APIs expect the
    conversation to open with the user, and the oldest turn is the least valuable one."""
    out: list[dict[str, Any]] = []
    for item in section.items or []:
        role = item.get("role", "").lower()
        content = (item.get("content") or "").strip()
        if role in _ROLES and content:
            _append(out, role, content)
    while out and out[0]["role"] != "user":
        out.pop(0)
    return out


def compose_messages(
    sections: list[PromptSection], history_as_messages: bool
) -> tuple[list[dict[str, Any]], str, str]:
    """Returns (messages, system_prompt, user_prompt). `user_prompt` is always the flat text of every non-system section."""
    system_prompt = "\n\n".join(s.content for s in sections if s.name == "system")
    user_prompt = "\n\n".join(s.content for s in sections if s.name != "system")

    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    history = next((s for s in sections if s.name == HISTORY_SECTION and s.items), None)
    if history_as_messages and history is not None:
        for m in history_messages(history):
            messages.append(m)
        rest = "\n\n".join(s.content for s in sections if s.name not in ("system", HISTORY_SECTION))
        if rest:
            _append(messages, "user", rest)
    elif user_prompt:
        messages.append({"role": "user", "content": user_prompt})
    return messages, system_prompt, user_prompt
