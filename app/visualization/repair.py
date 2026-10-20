"""
Targeted repair of ONE failed visual block.

The repair prompt contains only: the failed block's source/spec, a short content-free error description and the format rules.
It never includes the conversation, other blocks or any other user data, so the cost is bounded (a few hundred tokens) and
a repair cannot rewrite the surrounding answer. The reply is parsed and re-validated with the same validators as the original.
"""

import json
import re
from dataclasses import dataclass

from app.context_window.models import TrimmedPrompt
from app.visualization.manim_templates import SpecError, template_catalog, validate_spec
from app.visualization.mermaid import validate_mermaid

MAX_FAILED_INPUT = 3000
MAX_ERROR = 300
MAX_REPLY_CHARS = 8000
_FENCE = re.compile(r"```[a-zA-Z]*\s*\n?(.*?)```", re.S)

_MERMAID_RULES = (
    "You fix ONE Mermaid diagram. Reply with ONLY the corrected diagram source, no prose, no code fence. Rules: first line is the "
    "diagram type (flowchart TD, sequenceDiagram, classDiagram, stateDiagram-v2, erDiagram, gantt, pie...). Wrap any label that "
    "has spaces or punctuation in double quotes. No HTML, no links, no click/callback lines, no %% directives, no init blocks."
)


def _manim_rules() -> str:
    names = ", ".join(t["name"] for t in template_catalog())
    return (
        "You fix ONE animation specification. Reply with ONLY a corrected JSON object, no prose, no code fence: "
        '{"template": <name>, "title": <short>, "params": {...}}. Python code is never allowed. '
        f"Allowed templates: {names}. Keep parameters small (lists of at most 10 short items)."
    )


@dataclass(frozen=True)
class RepairOutcome:
    ok: bool
    kind: str
    content: str = ""            # corrected mermaid source, or canonical manim spec JSON
    category: str = ""           # validation category when not ok


def build_repair_prompt(kind: str, failed: str, error: str, attempt: int) -> TrimmedPrompt:
    rules = _MERMAID_RULES if kind == "mermaid" else _manim_rules()
    user = (
        f"Attempt {attempt}. The previous {kind} {'source' if kind == 'mermaid' else 'specification'} failed.\n"
        f"Error: {error[:MAX_ERROR]}\n\nPrevious:\n{failed[:MAX_FAILED_INPUT]}\n\nReturn the corrected one now."
    )
    messages = [{"role": "system", "content": rules}, {"role": "user", "content": user}]
    return TrimmedPrompt(messages=messages, mode="visualization_repair")


def _extract(reply: str) -> str:
    text = (reply or "")[:MAX_REPLY_CHARS].strip()
    m = _FENCE.search(text)
    return (m.group(1) if m else text).strip()


def validate_candidate(kind: str, candidate: str, *, max_mermaid: int = 6000, max_manim: int = 4000) -> RepairOutcome:
    if kind == "mermaid":
        v = validate_mermaid(candidate, max_mermaid)
        return RepairOutcome(v.ok, kind, candidate if v.ok else "", "" if v.ok else v.category)
    try:
        spec = validate_spec(json.loads(candidate), max_manim)
    except SpecError as exc:
        return RepairOutcome(False, kind, category="unsupported_template" if exc.unsupported else "invalid_spec")
    except (ValueError, RecursionError):
        return RepairOutcome(False, kind, category="invalid_spec")
    return RepairOutcome(True, kind, spec.canonical_json())


async def repair_block(chain, kind: str, failed: str, error: str, attempt: int, *, max_mermaid: int = 6000,
                       max_manim: int = 4000) -> RepairOutcome:
    """Ask the generation chain for a corrected block and validate it. Never raises for model/provider problems."""
    if chain is None or kind not in ("mermaid", "manim"):
        return RepairOutcome(False, kind, category="worker_unavailable")
    prompt = build_repair_prompt(kind, failed, error, attempt)
    params = {"max_tokens": 700, "temperature": 0.2}
    parts: list[str] = []
    size = 0
    try:
        async for token in chain.generate_stream(prompt, params):
            parts.append(token)
            size += len(token)
            if size > MAX_REPLY_CHARS:
                break
    except Exception:
        return RepairOutcome(False, kind, category="worker_unavailable")
    return validate_candidate(kind, _extract("".join(parts)), max_mermaid=max_mermaid, max_manim=max_manim)
