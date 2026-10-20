"""Short prompt addendum that teaches the model when and how to emit visual blocks. Empty (zero tokens) when the feature is
off for the active mode. Kept to a few hundred tokens: the account is token-per-minute limited."""

from app.config.settings import LLMServiceConfig
from app.visualization.manim_templates import TEMPLATES

_MERMAID = (
    "VISUALS. You can draw diagrams that the app renders for the user. When the user asks to explain, compare or design "
    "something that has structure or flow (an architecture, a process or pipeline, a lifecycle, a sequence of calls, a hierarchy, "
    "relationships between entities, a timeline), or asks for a diagram, include ONE fenced ```mermaid block instead of an ASCII-art "
    "drawing. NEVER draw ASCII/box-and-arrow diagrams. Rules for the block: the first line is the diagram type (flowchart TD, "
    "sequenceDiagram, classDiagram, stateDiagram-v2, erDiagram, gantt, pie); put EVERY node label in double quotes, e.g. A[\"Producer (App 1)\"] - unquoted parentheses, slashes or colons break the diagram; "
    "no HTML, links, click lines or %% directives. Keep the written explanation complete, so the answer also works "
    "without the diagram. Skip diagrams for simple, factual or conversational questions."
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
        " When the user asks for an animation (or an algorithm walkthrough is clearer with motion), include ONE fenced ```manim block "
        'containing ONLY a JSON object {"template": <name>, "title": <short>, "params": {...}}; never Python code. Pick the CLOSEST '
        "template: components talking to each other (architectures, protocols, request flows) -> message_flow with the components as "
        "actors; algorithms -> their template; anything else -> text_steps. Params must follow exactly: " + hints + ". Labels are "
        "short plain text (letters, digits, simple punctuation)."
    )


def visualization_guidance(config: LLMServiceConfig, mode: str) -> str:
    if not config.visualization_enabled or str(mode).lower() not in config.visualization_mode_list:
        return ""
    text = _MERMAID if config.mermaid_enabled else ""
    if config.manim_enabled:
        text = (text or "VISUALS (optional).") + _manim()
    return text
