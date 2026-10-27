"""
Versioned response-block contract.

An answer is an ORDERED list of typed blocks. The wire/storage form stays plain Markdown text (so the existing streaming,
persistence, replay and older clients keep working): visual blocks are fenced code blocks with the language tag `mermaid`
or `manim`. This module is the single, strict parser/validator of that form:

    text      free Markdown
    mermaid   a Mermaid diagram source (rendered in the browser after validation + sanitisation)
    manim     a JSON animation specification (rendered by an isolated worker from approved templates)
    invalid   a block that was recognised but malformed: it never invalidates its text siblings and renders as a fallback

Nothing here trusts the model: sizes, counts and kinds are bounded, ids are assigned by the parser (never taken from the
model), unknown fence languages stay ordinary code (text) and an unterminated fence is text (it is still streaming).
"""

import json
import re
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0"
MAX_TITLE_CHARS = 120

_FENCE_OPEN = re.compile(r"^(?P<indent> {0,3})(?P<fence>`{3,}|~{3,})\s*(?P<info>[^`\n]*)$")


class TextBlock(BaseModel):
    id: str
    type: Literal["text"] = "text"
    content: str


class MermaidBlock(BaseModel):
    id: str
    type: Literal["mermaid"] = "mermaid"
    title: str | None = None
    source: str


class ManimBlock(BaseModel):
    id: str
    type: Literal["manim"] = "manim"
    title: str | None = None
    spec: dict


class InvalidBlock(BaseModel):
    id: str
    type: Literal["invalid"] = "invalid"
    requested_type: Literal["mermaid", "manim"]
    reason: Literal["too_large", "malformed", "disabled", "limit_exceeded", "empty"]
    original: str          # the fenced text as the model wrote it (shown as a plain code block fallback)


Block = Annotated[Union[TextBlock, MermaidBlock, ManimBlock, InvalidBlock], Field(discriminator="type")]


class ParsedResponse(BaseModel):
    schema_version: str = SCHEMA_VERSION
    blocks: list[Block]

    @property
    def visual_blocks(self) -> list[Block]:
        return [b for b in self.blocks if b.type in ("mermaid", "manim")]


def _clean_title(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    flat = " ".join("".join(ch for ch in value if ch.isprintable()).split())[:MAX_TITLE_CHARS]
    return flat or None



def _is_manim_spec(lines: list[str], start: int, fence: str) -> bool:
    """A ```json (or untagged) fence whose body is exactly a Manim spec: an object with a known template and params."""
    close = next((j for j in range(start + 1, len(lines))
                  if lines[j].strip().startswith(fence[0] * len(fence)) and set(lines[j].strip()) == {fence[0]}), None)
    if close is None:
        return False
    try:
        spec = json.loads("\n".join(lines[start + 1:close]))
    except (ValueError, RecursionError):
        return False
    from app.visualization.manim_templates import TEMPLATES
    return isinstance(spec, dict) and spec.get("template") in TEMPLATES and isinstance(spec.get("params"), dict)

def parse_response_blocks(
    text: str,
    *,
    max_visual_blocks: int = 5,
    mermaid_enabled: bool = True,
    manim_enabled: bool = True,
    max_mermaid_chars: int = 6000,
    max_manim_chars: int = 4000,
) -> ParsedResponse:
    """Split Markdown into ordered blocks. Never raises: anything unexpected stays text."""
    lines = (text or "").split("\n")
    blocks: list = []
    buffer: list[str] = []
    visuals = 0

    def flush_text() -> None:
        content = "\n".join(buffer)
        buffer.clear()
        if content.strip():
            blocks.append(TextBlock(id=f"b{len(blocks) + 1}", content=content))

    i = 0
    while i < len(lines):
        m = _FENCE_OPEN.match(lines[i])
        lang = (m.group("info").strip().split() or [""])[0].lower() if m else ""
        if m and lang in ("json", "") and _is_manim_spec(lines, i, m.group("fence")):
            lang = "manim"       # models sometimes tag the spec ```json; it is still a Manim spec and must render
        if not m or lang not in ("mermaid", "manim"):
            buffer.append(lines[i])
            i += 1
            continue

        fence = m.group("fence")
        close = next((j for j in range(i + 1, len(lines))
                      if lines[j].strip().startswith(fence[0] * len(fence)) and set(lines[j].strip()) == {fence[0]}), None)
        if close is None:                                   # unterminated: still streaming (or broken) -> plain text
            buffer.extend(lines[i:])
            break
        original = "\n".join(lines[i:close + 1])
        body = "\n".join(lines[i + 1:close])
        flush_text()
        block_id = f"b{len(blocks) + 1}"
        kind = lang
        enabled = mermaid_enabled if kind == "mermaid" else manim_enabled
        limit = max_mermaid_chars if kind == "mermaid" else max_manim_chars

        def invalid(reason: str) -> None:
            blocks.append(InvalidBlock(id=block_id, requested_type=kind, reason=reason, original=original))  # type: ignore[arg-type]

        if not enabled:
            invalid("disabled")
        elif visuals >= max_visual_blocks:
            invalid("limit_exceeded")
        elif not body.strip():
            invalid("empty")
        elif len(body) > limit:
            invalid("too_large")
        elif kind == "mermaid":
            title = None
            first = body.lstrip().split("\n", 1)[0]
            tm = re.match(r"^%%\s*title:\s*(.+)$", first, re.IGNORECASE)
            if tm:
                title = _clean_title(tm.group(1))
                body = body.lstrip().split("\n", 1)[1] if "\n" in body.lstrip() else ""
            if not body.strip():
                invalid("empty")
            else:
                blocks.append(MermaidBlock(id=block_id, title=title, source=body.strip("\n")))
                visuals += 1
        else:
            try:
                spec = json.loads(body)
                if not isinstance(spec, dict):
                    raise ValueError
            except (ValueError, RecursionError):
                invalid("malformed")
            else:
                blocks.append(ManimBlock(id=block_id, title=_clean_title(spec.get("title")), spec=spec))
                visuals += 1
        i = close + 1
    flush_text()
    return ParsedResponse(blocks=blocks)
