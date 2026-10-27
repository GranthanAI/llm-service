"""Short prompt addendum that teaches the model when and how to emit visual blocks. Empty (zero tokens) when the feature is
off for the active mode. Kept to a few hundred tokens: the account is token-per-minute limited."""

from app.config.settings import LLMServiceConfig
from app.visualization.manim_templates import TEMPLATES

_MERMAID = (
    "VISUALS. The app renders diagrams for the user. For something with structure or flow (architecture, process, lifecycle, "
    "call sequence, hierarchy, relationships, timeline) or when a diagram is asked for, include ONE fenced ```mermaid block; "
    "NEVER draw ASCII/box-and-arrow art. Block rules: the first line is the type (flowchart TD, sequenceDiagram, classDiagram, "
    "stateDiagram-v2, erDiagram, gantt, pie); put EVERY node label in double quotes, e.g. A[\"Producer (App 1)\"]; no HTML, "
    "links, click lines or %% directives. The written answer must stand without the diagram. Skip diagrams for simple, "
    "factual or chat questions."
)


_HINTS = {
    "binary_search": '{"array":[sorted ints],"target":int}',
    "sorting": '{"algorithm":"bubble|insertion|selection","array":[ints 1-99, max 10]}',
    "recursion_stack": '{"function":"factorial|fibonacci","n":1-6}',
    "graph_traversal": '{"algorithm":"bfs|dfs","nodes":["A","B"],"edges":[["A","B"]],"start":"A"}',
    "message_flow": '{"actors":["Client","Server"],"messages":[{"from":"Client","to":"Server","label":"request"}]}',
    "state_machine": '{"states":["Idle","Run"],"transitions":[{"from":"Idle","to":"Run","label":"start"}],"path":["Idle","Run"]}',
    "text_steps": '{"steps":["line 1","line 2"]}',
}


def _manim() -> str:
    hints = "; ".join(f"{name} params {_HINTS[name]}" for name in TEMPLATES if name in _HINTS)
    return (
        " ANIMATIONS: when asked for an animation (or motion clarifies an algorithm), include ONE fenced ```manim block holding "
        'ONLY JSON {"template": <name>, "title": <short>, "params": {...}}; never Python; the fence tag is exactly manim, never '
        "json. Pick the CLOSEST template: components talking (architectures, protocols, request flows) -> message_flow; algorithms -> "
        "their template; else text_steps. Params exactly: " + hints + ". Text is ASCII only (plain hyphen, no en/em dashes or "
        "curly quotes); actor names max 16 chars (2-5 actors), message labels max 28, step lines max 48. "
        "HARD RULES: the user only sees the finished video, never your JSON. So do NOT tell them to paste, load, run, render or "
        "customize it; do NOT mention manim, templates, JSON or tools; do NOT describe or announce the visual; NO how-to-use or "
        "next-step text. Text around the block is only the explanation they asked for."
    )


_REQUESTED = (
    " The user switched on 'Include diagram if necessary' for this message: if a diagram or animation would make the answer "
    "clearer, include one (a diagram of the structure, flow or steps involved); if it would not help, do not force one."
)


def visualization_guidance(config: LLMServiceConfig, mode: str, requested: bool = False) -> str:
    """`requested` is the user's per-message opt-in: it offers the visual blocks in every mode (the mode list only
    decides what is offered by default) and tells the model the user wants one when it helps. The feature flags still apply."""
    if not config.visualization_enabled or (not requested and str(mode).lower() not in config.visualization_mode_list):
        return ""
    text = _MERMAID if config.mermaid_enabled else ""
    if config.manim_enabled:
        text = (text or "VISUALS (optional).") + _manim()
    return text + _REQUESTED if requested and text else text
