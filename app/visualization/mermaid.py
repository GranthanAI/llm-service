"""Mermaid source validation (server side).

This is a SAFETY and sanity gate, not a full Mermaid parser (the browser's Mermaid is the authoritative syntax checker):
it rejects oversized/over-complex sources, unsupported diagram types and anything that could carry markup, scripts, links
or external resources, and catches the most common syntax slips (unbalanced brackets) so a repair can be attempted
before the diagram ever reaches the browser."""

import re
from dataclasses import dataclass

ALLOWED_TYPES = (
    "flowchart", "graph", "sequenceDiagram", "classDiagram", "stateDiagram-v2", "stateDiagram", "erDiagram", "gantt",
    "pie", "journey", "mindmap", "timeline", "gitGraph", "quadrantChart", "requirementDiagram", "xychart-beta",
)
MAX_LINES = 400
_UNSAFE = [
    (re.compile(r"<\s*/?\s*(script|iframe|object|embed|img|a|svg|style|link|meta|form|input)\b", re.I), "markup tag"),
    (re.compile(r"javascript\s*:|vbscript\s*:|data\s*:", re.I), "script or data URL"),
    (re.compile(r"\bon[a-z]+\s*=", re.I), "event handler"),
    (re.compile(r"%%\s*\{"), "directive"),
    (re.compile(r"^\s*(click|link|callback)\b", re.I | re.M), "interaction/link directive"),
    (re.compile(r"\bhref\s*=|url\s*\(|@import", re.I), "external resource"),
    (re.compile(r"https?://|//\S+\.\w", re.I), "external URL"),
]
_PAIRS = {"[": "]", "(": ")", "{": "}"}


@dataclass(frozen=True)
class MermaidValidation:
    ok: bool
    category: str = ""     # empty | too_large | too_complex | unsafe | unsupported_type | syntax
    message: str = ""


def _strip_quoted(line: str) -> str:
    return re.sub(r'"[^"\n]*"', '""', line)


def validate_mermaid(source: str, max_chars: int = 6000) -> MermaidValidation:
    text = (source or "").strip("\n")
    if not text.strip():
        return MermaidValidation(False, "empty", "The diagram is empty.")
    if len(text) > max_chars:
        return MermaidValidation(False, "too_large", f"The diagram is longer than {max_chars} characters.")
    lines = text.split("\n")
    if len(lines) > MAX_LINES:
        return MermaidValidation(False, "too_complex", f"The diagram has more than {MAX_LINES} lines.")
    for pattern, label in _UNSAFE:
        if pattern.search(text):
            return MermaidValidation(False, "unsafe", f"Not allowed: {label}.")

    body = [ln for ln in lines if ln.strip() and not ln.strip().startswith("%%")]
    first = body[0].strip() if body else ""
    if first == "---":                                       # YAML front matter is not allowed (config smuggling)
        return MermaidValidation(False, "unsafe", "Front matter is not allowed.")
    keyword = re.split(r"[\s{;:]", first, maxsplit=1)[0]
    if keyword not in ALLOWED_TYPES:
        return MermaidValidation(False, "unsupported_type", f"Unsupported or missing diagram type: {keyword[:30]!r}.")

    if keyword not in ("flowchart", "graph"):      # free-text diagram types legitimately contain stray brackets
        return MermaidValidation(True)
    for number, line in enumerate(body, 1):
        stack: list[str] = []
        for ch in _strip_quoted(line):
            if ch in _PAIRS:
                stack.append(_PAIRS[ch])
            elif ch in _PAIRS.values():
                if not stack or stack.pop() != ch:
                    return MermaidValidation(False, "syntax", f"Unbalanced brackets near line {number}.")
        if stack:
            return MermaidValidation(False, "syntax", f"Unclosed bracket on line {number}.")
    return MermaidValidation(True)
