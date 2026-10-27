import json

import pytest

from app.visualization.blocks import parse_response_blocks
from app.visualization.manim_templates import (
    SpecError, TEMPLATES, build_scene_source, validate_spec,
)
from app.visualization.mermaid import validate_mermaid
from app.visualization.policy import (
    AttemptsExhausted, AttemptTracker, FailureCategory, text_fallback,
)


# ---- blocks ----------------------------------------------------------------

def test_plain_markdown_is_one_text_block():
    r = parse_response_blocks("Hello\n\n```python\nprint(1)\n```\n")
    assert [b.type for b in r.blocks] == ["text"]


def test_interleaved_blocks_keep_order():
    spec = json.dumps({"template": "text_steps", "params": {}})
    text = f"intro\n```mermaid\nflowchart TD\nA-->B\n```\nmid\n```manim\n{spec}\n```\nend"
    r = parse_response_blocks(text)
    assert [b.type for b in r.blocks] == ["text", "mermaid", "text", "manim", "text"]


def test_unterminated_fence_is_text():
    r = parse_response_blocks("a\n```mermaid\nflowchart TD\nA-->B")
    assert [b.type for b in r.blocks] == ["text"]


def test_malformed_manim_and_disabled_and_limits():
    r = parse_response_blocks("```manim\nnot json\n```")
    assert r.blocks[0].type == "invalid" and r.blocks[0].reason == "malformed"
    r = parse_response_blocks("```mermaid\nflowchart TD\nA-->B\n```", mermaid_enabled=False)
    assert r.blocks[0].reason == "disabled"
    r = parse_response_blocks("```mermaid\nflowchart TD\nA-->B\n```\n```mermaid\nflowchart TD\nC-->D\n```", max_visual_blocks=1)
    assert [b.type for b in r.blocks] == ["mermaid", "invalid"]
    r = parse_response_blocks("```mermaid\n" + "x" * 300 + "\n```", max_mermaid_chars=100)
    assert r.blocks[0].reason == "too_large"


def test_mermaid_title_extracted():
    r = parse_response_blocks("```mermaid\n%% title: Flow\nflowchart TD\nA-->B\n```")
    assert r.blocks[0].title == "Flow" and r.blocks[0].source.startswith("flowchart")


# ---- mermaid ---------------------------------------------------------------

@pytest.mark.parametrize("src", [
    "flowchart TD\nA[Start] --> B{Ok?}", "sequenceDiagram\nA->>B: hi", "pie\n\"a\": 1",
])
def test_valid_mermaid(src):
    assert validate_mermaid(src).ok


@pytest.mark.parametrize("src,cat", [
    ("", "empty"),
    ("flowchart TD\nA[<script>x</script>]", "unsafe"),
    ("flowchart TD\nA-->B\nclick A href \"x\"", "unsafe"),
    ("%%{init: {}}%%\nflowchart TD\nA-->B", "unsafe"),
    ("---\ntitle: x\n---\nflowchart TD\nA-->B", "unsafe"),
    ("flowchart TD\nA[x](javascript:alert(1))", "unsafe"),
    ("notadiagram\nA-->B", "unsupported_type"),
    ("flowchart TD\nA[Start --> B", "syntax"),
])
def test_invalid_mermaid(src, cat):
    v = validate_mermaid(src)
    assert not v.ok and v.category == cat


def test_mermaid_size_limits():
    assert validate_mermaid("flowchart TD\n" + "A-->B\n" * 2000).category == "too_large"
    assert validate_mermaid("flowchart TD\n" + "A-->B\n" * 450, max_chars=10_000).category == "too_complex"


# ---- policy ----------------------------------------------------------------

def test_exactly_three_total_attempts():
    t = AttemptTracker()
    assert [t.begin_attempt() for _ in range(3)] == [1, 2, 3]
    with pytest.raises(AttemptsExhausted):
        t.begin_attempt()


def test_next_step_rules():
    t = AttemptTracker()
    t.begin_attempt()
    assert t.next_step(FailureCategory.SYNTAX) == "repair"
    assert t.next_step(FailureCategory.TIMEOUT) == "retry"
    assert t.next_step(FailureCategory.DISABLED) == "give_up"
    t.begin_attempt(); t.begin_attempt()
    assert t.next_step(FailureCategory.SYNTAX) == "give_up"


def test_text_fallback_is_bounded_and_content_free():
    out = text_fallback("Sorting", FailureCategory.RENDER_ERROR, "manim")
    assert "Animation" in out and "Traceback" not in out


# ---- manim templates ---------------------------------------------------------

@pytest.mark.parametrize("name", list(TEMPLATES))
def test_every_example_validates_and_compiles(name):
    t = TEMPLATES[name]
    v = validate_spec({"template": name, "title": "Demo", "params": t.example})
    src = build_scene_source(v)
    compile(src, "<scene>", "exec")


@pytest.mark.parametrize("spec", [
    "nope",
    {"template": "os_system", "params": {}},
    {"template": "sorting", "params": {"algorithm": "bubble", "array": [3, 1], "extra": 1}},
    {"template": "sorting", "params": {"algorithm": "quick", "array": [3, 1]}},
    {"template": "sorting", "params": {"algorithm": "bubble", "array": list(range(1, 30))}},
    {"template": "binary_search", "params": {"array": [3, 1, 2], "target": 1}},
    {"template": "text_steps", "title": "x" * 200, "params": {"steps": ["a", "b"]}},
    {"template": "text_steps", "params": {"steps": ["__import__('os')\"; evil", "b"]}},
    {"template": "text_steps", "params": {"steps": ["a", "b"]}, "unknown": 1},
])
def test_bad_specs_rejected(spec):
    with pytest.raises(SpecError):
        validate_spec(spec)


def test_unknown_template_flagged_unsupported():
    with pytest.raises(SpecError) as e:
        validate_spec({"template": "evil", "params": {}})
    assert e.value.unsupported


def test_params_never_become_code():
    v = validate_spec({"template": "text_steps", "title": "T", "params": {"steps": ["x = 1", "y = 2"]}})
    src = build_scene_source(v)
    assert "x = 1" not in src  # only present inside base64 data


def test_oversized_spec_rejected():
    with pytest.raises(SpecError):
        validate_spec({"template": "text_steps", "params": {"steps": ["a" * 40] * 8}}, max_chars=100)


# ---- prompt guidance -----------------------------------------------------------

def test_guidance_scoped_by_mode_and_flags():
    from app.config.settings import LLMServiceConfig
    from app.visualization.guidance import visualization_guidance
    cfg = LLMServiceConfig(manim_enabled=False)
    assert "mermaid" in visualization_guidance(cfg, "tutor") and "manim" not in visualization_guidance(cfg, "tutor")
    assert visualization_guidance(cfg, "web_search") == ""                 # grounded modes off by default
    assert visualization_guidance(LLMServiceConfig(visualization_enabled=False), "tutor") == ""
    on = visualization_guidance(LLMServiceConfig(manim_enabled=True), "default")
    assert "manim" in on and "never Python" in on
    assert len(on) < 2400                                                   # stays a few hundred tokens


def test_builder_appends_guidance():
    from app.prompts.builder import PromptBuilder
    import inspect
    assert "visualization_guidance" in inspect.getsource(PromptBuilder._format_system)


def test_manim_guidance_gives_exact_params_and_architecture_hint():
    from app.config.settings import LLMServiceConfig
    from app.visualization.guidance import visualization_guidance
    g = visualization_guidance(LLMServiceConfig(manim_enabled=True), "default")
    assert '"actors"' in g and "message_flow" in g and "CLOSEST" in g
    from app.visualization.repair import build_repair_prompt
    rules = build_repair_prompt("manim", "{}", "invalid_spec", 2).messages[0]["content"]
    assert '"actors"' in rules and '"steps"' in rules


# ---- json-tagged specs (a model answered ```json with a valid Manim spec and nothing rendered) -------------------------

NMAP = {
    "template": "message_flow", "title": "Nmap Scanning Workflow",
    "params": {"actors": ["Attacker (Nmap)", "Target Host", "Network Devices"], "messages": [
        {"from": "Attacker (Nmap)", "to": "Target Host", "label": "Host discovery (ping/ARP)"},
        {"from": "Attacker (Nmap)", "to": "Network Devices", "label": "Traceroute / TTL probing"}]},
}


@pytest.mark.parametrize("tag", ["json", ""])
def test_manim_spec_in_a_json_or_untagged_fence_still_renders(tag):
    r = parse_response_blocks(f"Here you go:\n```{tag}\n{json.dumps(NMAP, indent=2)}\n```\nDone")
    assert [b.type for b in r.blocks] == ["text", "manim", "text"]
    assert r.blocks[1].spec["template"] == "message_flow"
    validate_spec(r.blocks[1].spec)                      # and the spec itself is accepted by the renderer's validator


@pytest.mark.parametrize("body", [
    '{"name": "x", "version": 1}',                       # ordinary JSON stays a code block
    '{"template": "not_a_template", "params": {}}',
    '{"template": "message_flow"}',                      # no params
    '[1, 2, 3]', 'not json at all',
])
def test_other_json_stays_ordinary_code(body):
    r = parse_response_blocks(f"```json\n{body}\n```")
    assert [b.type for b in r.blocks] == ["text"]


def test_guidance_pins_the_fence_language_and_limits():
    from app.config.settings import LLMServiceConfig
    from app.visualization.guidance import visualization_guidance
    text = visualization_guidance(LLMServiceConfig(), "smart")
    if text:                                             # empty when the feature is off for the mode in this environment
        assert "exactly manim" in text and "max 28" in text and "max 16" in text


# ---- per-message "Include diagram if necessary" -----------------------------------------------------------------------

def test_requested_diagrams_are_offered_in_every_mode_with_a_nudge():
    from app.config.settings import LLMServiceConfig
    from app.visualization.guidance import visualization_guidance
    cfg = LLMServiceConfig()
    assert visualization_guidance(cfg, "web_search") == ""                       # grounded modes: off unless requested
    asked = visualization_guidance(cfg, "web_search", requested=True)
    assert "```mermaid" in asked and "Include diagram if necessary" in asked
    base = visualization_guidance(cfg, "smart")
    assert "Include diagram if necessary" not in base                            # default behaviour is unchanged
    assert visualization_guidance(cfg, "smart", requested=True).startswith(base)


def test_requested_does_not_override_the_feature_flags():
    from app.config.settings import LLMServiceConfig
    off = LLMServiceConfig(visualization_enabled=False)
    assert visualization_guidance_off(off) == ""


def visualization_guidance_off(cfg):
    from app.visualization.guidance import visualization_guidance
    return visualization_guidance(cfg, "smart", requested=True)


def test_event_carries_the_flag_into_the_pipeline_context():
    from app.models.request import ChatMessageCreatedEvent
    fields = ChatMessageCreatedEvent.model_fields
    assert "include_diagrams" in fields and fields["include_diagrams"].default is False


# ---- real model output that was rejected: long Hinglish step lines with typographic dashes ---------------------------

def test_long_steps_with_typographic_punctuation_are_normalised_and_clipped_not_rejected():
    steps = ["1. Supervised Learning – Labeled data se mapping seekhta hai",
             "2. Unsupervised Learning – Unlabeled data se structure discover karta hai",
             "3. Semi‑Supervised Learning – Thode labeled + bahut unlabeled data use karta hai"]
    v = validate_spec({"template": "text_steps", "title": "ML types", "params": {"steps": steps}})
    out = TEMPLATES["text_steps"].build(v.params)["steps"]      # what the scene actually receives
    assert all(len(s) <= 48 and s.isascii() for s in out)
    assert out[0].startswith("1. Supervised Learning - Labeled")
    assert out[1].endswith("...")


def test_injection_shaped_text_is_still_rejected_even_with_clipping():
    with pytest.raises(SpecError):
        validate_spec({"template": "text_steps", "params": {"steps": ["ok", "x\"; __import__('os').system('id') #" + "a" * 60]}})
    with pytest.raises(SpecError):                                  # names are never clipped
        validate_spec({"template": "message_flow", "params": {"actors": ["A" * 30, "B"], "messages": []}})


def test_guidance_forbids_narrating_the_visual():
    from app.config.settings import LLMServiceConfig
    from app.visualization.guidance import visualization_guidance
    text = visualization_guidance(LLMServiceConfig(manim_enabled=True), "smart")
    assert "HARD RULES" in text and "never your JSON" in text and "NO how-to-use" in text and len(text) < 2400
