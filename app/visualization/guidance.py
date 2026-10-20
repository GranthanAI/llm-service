"""Short prompt addendum that teaches the model when and how to emit visual blocks. Empty (zero tokens) when the feature is
off for the active mode. Kept to a few hundred tokens: the account is token-per-minute limited."""

from app.config.settings import LLMServiceConfig
from app.visualization.manim_templates import TEMPLATES

_MERMAID = (
    "VISUALS. You can draw diagrams that the app renders for the user. When the user asks to explain, compare or design "
    "something that has structure or flow (an architecture, a process or pipeline, a lifecycle, a sequence of calls, a hierarchy, "
    "relationships between entities, a timeline), or asks for a diagram, include ONE fenced ```mermaid block instead of an ASCII-art "
    "drawing. NEVER draw ASCII/box-and-arrow diagrams. Rules for the block: the first line is the diagram type (flowchart TD, "
    "sequenceDiagram, classDiagram, stateDiagram-v2, erDiagram, gantt, pie); put every label that has spaces or punctuation in double "
    "quotes; no HTML, links, click lines or %% directives. Keep the written explanation complete, so the answer also works "
    "without the diagram. Skip diagrams for simple, factual or conversational questions."
)


def _manim() -> str:
    names = ", ".join(TEMPLATES)
    return (
        " When the user asks for an animation, or a step-by-step algorithm is clearer with motion, you may also include ONE fenced ```manim block "
        "containing ONLY a JSON object {\"template\": <name>, \"title\": <short>, \"params\": {...}}; never Python code. "
        f"Templates: {names}. Keep parameters small (at most 10 short items)."
    )


def visualization_guidance(config: LLMServiceConfig, mode: str) -> str:
    if not config.visualization_enabled or str(mode).lower() not in config.visualization_mode_list:
        return ""
    text = _MERMAID if config.mermaid_enabled else ""
    if config.manim_enabled:
        text = (text or "VISUALS (optional).") + _manim()
    return text
