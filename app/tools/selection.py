"""
Explicit tool selection: discovery shape, validation against the registry, and call construction.

`selected_tools` arrives on the chat event as plain strings from an untrusted client. Nothing in it is
executed unless it names a registered AND enabled tool, and a tool that cannot be run from the user's
message alone is rejected instead of being run with guessed parameters.
"""

from typing import Any

from app.exceptions.user_facing import InvalidToolSelectionError
from app.models.execution_plan import ToolCall
from app.models.tool import ToolSchema
from app.tools.registry import ToolRegistry

MAX_SELECTED_TOOLS = 5

# Parameter names that mean "the user's request text" for a single-input tool.
_TEXT_PARAMS = ("query", "prompt", "text", "input", "question")


def describe_tool(schema: ToolSchema) -> dict[str, Any]:
    """Public, UI-safe description of a tool. Contains the schema the tool already publishes to the
    planner; never configuration, credentials, endpoints or provider details."""
    props = (schema.parameters or {}).get("properties", {})
    required = list((schema.parameters or {}).get("required", []))
    return {
        "id": schema.name,
        "name": schema.name.replace("_", " ").replace("-", " ").title(),
        "description": schema.user_description or schema.description,
        "version": schema.version,
        "parameters": [
            {
                "name": name,
                "type": spec.get("type", "string"),
                "description": spec.get("description", ""),
                "required": name in required,
            }
            for name, spec in props.items()
        ],
        # True when the tool can be run directly from the user's message (no extra input needed).
        "auto_runnable": all(r in _TEXT_PARAMS for r in required),
    }


def list_enabled_tools(registry: ToolRegistry) -> list[dict[str, Any]]:
    """Tools a user can pick. Capabilities that a response mode already guarantees (Web search) or that
    only the planner should use are not listed, so the UI never shows the same capability twice."""
    return [describe_tool(s) for s in registry.list_tools() if s.user_selectable]


def normalize_selection(selected: list[str] | None) -> list[str]:
    """Trim, drop empties, de-duplicate (order kept)."""
    seen: dict[str, None] = {}
    for raw in selected or []:
        name = str(raw).strip()
        if name:
            seen.setdefault(name, None)
    return list(seen)


def filter_default_tools(selected: list[str] | None, registry: ToolRegistry | None) -> tuple[list[str], list[str]]:
    """Saved *default* tools are applied only where they still work: unknown, disabled,
    unconfigured or not-runnable-from-chat tools are dropped (not an error, the user did
    not choose them for this message), and the live MAX_SELECTED_TOOLS cap applies.
    Returns (kept, dropped)."""
    names = normalize_selection(selected)
    if not names or registry is None:
        return [], names
    kept: list[str] = []
    dropped: list[str] = []
    for name in names:
        ok = registry.is_user_selectable(name)
        if ok:
            tool = registry.get(name)
            ok = tool.is_available()
            if ok:
                required = list((tool.get_schema().parameters or {}).get("required", []))
                ok = all(r in _TEXT_PARAMS for r in required)
        if ok and len(kept) < MAX_SELECTED_TOOLS:
            kept.append(name)
        else:
            dropped.append(name)
    return kept, dropped


def drop_calls_covered_by_mode(calls: list[ToolCall], mode: Any) -> list[ToolCall]:
    """A mode that always runs a capability (Web search mode runs web_search) must not have it run a
    second time because the same tool was also selected explicitly."""
    covered = {"web_search"} if getattr(mode, "value", mode) == "web_search" else set()
    return [c for c in calls if c.tool_name not in covered]


def build_selected_calls(
    selected: list[str] | None, registry: ToolRegistry | None, user_message: str
) -> list[ToolCall]:
    """Validate the user's explicit selection and turn it into required ToolCalls.

    Raises InvalidToolSelectionError (shown to the user, nothing executed) when a name is unknown or
    disabled, when more than MAX_SELECTED_TOOLS are chosen, or when a tool needs input that cannot
    be derived from the message.
    """
    names = normalize_selection(selected)
    if not names:
        return []
    if registry is None:
        raise InvalidToolSelectionError(names, "tools are not available right now")
    if len(names) > MAX_SELECTED_TOOLS:
        raise InvalidToolSelectionError(names[MAX_SELECTED_TOOLS:], f"at most {MAX_SELECTED_TOOLS} tools can be selected")

    unavailable = [n for n in names if not registry.is_enabled(n)]
    if unavailable:
        raise InvalidToolSelectionError(unavailable, "unknown or disabled")

    calls: list[ToolCall] = []
    for name in names:
        tool = registry.get(name)
        if not tool.is_available():
            raise InvalidToolSelectionError([name], "it is not configured on this server")
        schema = tool.get_schema()
        required = list((schema.parameters or {}).get("required", []))
        params: dict[str, Any] = {}
        for r in required:
            if r in _TEXT_PARAMS:
                params[r] = user_message
            else:
                raise InvalidToolSelectionError([name], f"it needs a '{r}' value that cannot be provided from chat")
        calls.append(ToolCall(tool_name=name, params=params, parallel=True, required=True))
    return calls
