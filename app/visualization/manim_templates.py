"""
Approved Manim animation templates.

The model never writes Python. It produces a small JSON specification (`template` + `params`); this module validates it
strictly and builds the scene source from HAND-WRITTEN, reviewed template code. User/model data reaches that code only as a
base64-encoded JSON document that is validated, bounded and decoded at run time (`DATA`), so a parameter can never become
code. Anything outside the registered templates/parameters is rejected.

Specification (version 1.0):
    {"schema_version": "1.0", "template": "<name>", "title": "<=60 chars", "params": {...}}
"""

import base64
import json
import re
from dataclasses import dataclass
from typing import Any, Callable

SPEC_VERSION = "1.0"
RENDERER_VERSION = "manim-templates-1"
MAX_TITLE = 60
MAX_LABEL = 40
MAX_FRAMES = 60
_LABEL = re.compile(r"^[\w .,:+\-*/=<>()\[\]#%'?!&→←↔≤≥≠]+$", re.UNICODE)


class SpecError(ValueError):
    """The specification is not valid for the (known) template. `unsupported` marks an unknown template."""

    def __init__(self, message: str, unsupported: bool = False):
        super().__init__(message)
        self.unsupported = unsupported


_TYPOGRAPHY = str.maketrans({
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "−": "-",   # hyphens and dashes
    "‘": "'", "’": "'", "“": "'", "”": "'",                                  # curly quotes (no double quotes)
    "•": "-", " ": " ", " ": " ",
})


def _label(value: Any, name: str, limit: int = MAX_LABEL, clip: bool = False) -> str:
    """A short display string. Typographic punctuation models like to write (en/em dash, non-breaking hyphen, curly quotes)
    is converted to its ASCII form first; everything else must still pass the character whitelist (injection-shaped text is
    rejected, never repaired). `clip` shortens an over-long DISPLAY line at a word boundary instead of rejecting the whole
    animation (names and titles identify things, so they are never clipped)."""
    if isinstance(value, str):
        value = value.translate(_TYPOGRAPHY).replace("…", "...")
        text = value.strip()
        if clip and len(text) > limit and limit > 3:
            cut = text[: limit - 3]
            text = (cut.rsplit(" ", 1)[0] if " " in cut[limit // 2:] else cut).rstrip(" ,.;:-") + "..."
        value = text
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > limit or not _LABEL.match(value.strip()):
        raise SpecError(f"{name} must be 1-{limit} plain characters (letters, digits and simple punctuation)")
    return value.strip()


def _int(value: Any, name: str, lo: int, hi: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
        raise SpecError(f"{name} must be an integer between {lo} and {hi}")
    return value


def _int_list(value: Any, name: str, lo: int, hi: int, min_len: int, max_len: int) -> list[int]:
    if not isinstance(value, list) or not min_len <= len(value) <= max_len:
        raise SpecError(f"{name} must be a list of {min_len}-{max_len} integers")
    return [_int(v, f"{name}[{i}]", lo, hi) for i, v in enumerate(value)]


def _only(params: dict, allowed: set[str]) -> None:
    if not isinstance(params, dict):
        raise SpecError("params must be an object")
    extra = set(params) - allowed
    if extra:
        raise SpecError(f"unknown parameter(s): {sorted(extra)[:3]}")


# ----------------------------------------------------------------------------- data builders (pure, host side)


def _binary_search(params: dict) -> dict:
    _only(params, {"array", "target"})
    arr = _int_list(params.get("array"), "array", -999, 999, 2, 16)
    if arr != sorted(arr):
        raise SpecError("array must be sorted ascending")
    target = _int(params.get("target"), "target", -999, 999)
    lo, hi, steps = 0, len(arr) - 1, []
    found = False
    while lo <= hi:
        mid = (lo + hi) // 2
        cmp = "==" if arr[mid] == target else ("<" if arr[mid] < target else ">")
        steps.append({"low": lo, "high": hi, "mid": mid, "cmp": cmp})
        if arr[mid] == target:
            found = True
            break
        lo, hi = (mid + 1, hi) if arr[mid] < target else (lo, mid - 1)
    return {"array": arr, "target": target, "steps": steps, "found": found,
            "result": f"found at index {steps[-1]['mid']}" if found else "not found"}


def _sorting(params: dict) -> dict:
    _only(params, {"algorithm", "array"})
    algo = params.get("algorithm")
    if algo not in ("bubble", "insertion", "selection"):
        raise SpecError("algorithm must be bubble, insertion or selection")
    arr = _int_list(params.get("array"), "array", 1, 99, 2, 10)
    a, frames = list(arr), [{"arr": list(arr), "hl": [], "note": "start"}]
    n = len(a)

    def rec(hl, note):
        if len(frames) < MAX_FRAMES:
            frames.append({"arr": list(a), "hl": hl, "note": note})

    if algo == "bubble":
        for i in range(n):
            for j in range(n - 1 - i):
                rec([j, j + 1], f"compare {a[j]} and {a[j + 1]}")
                if a[j] > a[j + 1]:
                    a[j], a[j + 1] = a[j + 1], a[j]
                    rec([j, j + 1], "swap")
    elif algo == "insertion":
        for i in range(1, n):
            k = i
            while k > 0 and a[k - 1] > a[k]:
                rec([k - 1, k], f"{a[k]} < {a[k - 1]}: move left")
                a[k - 1], a[k] = a[k], a[k - 1]
                rec([k - 1, k], "swap")
                k -= 1
    else:
        for i in range(n):
            m = i
            for j in range(i + 1, n):
                rec([m, j], f"find minimum: {a[j]} vs {a[m]}")
                if a[j] < a[m]:
                    m = j
            if m != i:
                a[i], a[m] = a[m], a[i]
                rec([i, m], "swap minimum into place")
    frames.append({"arr": list(a), "hl": [], "note": "sorted"})
    return {"algorithm": algo, "frames": frames[:MAX_FRAMES + 1]}


def _recursion_stack(params: dict) -> dict:
    _only(params, {"function", "n"})
    fn = params.get("function")
    if fn not in ("factorial", "fibonacci"):
        raise SpecError("function must be factorial or fibonacci")
    n = _int(params.get("n"), "n", 1, 6 if fn == "fibonacci" else 6)
    events: list[dict] = []

    def fact(k: int) -> int:
        events.append({"op": "push", "text": f"fact({k})"})
        r = 1 if k <= 1 else k * fact(k - 1)
        events.append({"op": "pop", "text": f"fact({k}) = {r}"})
        return r

    def fib(k: int) -> int:
        events.append({"op": "push", "text": f"fib({k})"})
        r = k if k < 2 else fib(k - 1) + fib(k - 2)
        events.append({"op": "pop", "text": f"fib({k}) = {r}"})
        return r

    result = fact(n) if fn == "factorial" else fib(n)
    if len(events) > 2 * MAX_FRAMES:
        raise SpecError("this recursion is too long to animate; use a smaller n")
    return {"function": fn, "n": n, "events": events, "result": result}


def _graph_traversal(params: dict) -> dict:
    _only(params, {"algorithm", "nodes", "edges", "start"})
    algo = params.get("algorithm")
    if algo not in ("bfs", "dfs"):
        raise SpecError("algorithm must be bfs or dfs")
    nodes = params.get("nodes")
    if not isinstance(nodes, list) or not 2 <= len(nodes) <= 8:
        raise SpecError("nodes must be a list of 2-8 labels")
    nodes = [_label(v, "node", 12) for v in nodes]
    if len(set(nodes)) != len(nodes):
        raise SpecError("node labels must be unique")
    edges = params.get("edges")
    if not isinstance(edges, list) or not 1 <= len(edges) <= 14:
        raise SpecError("edges must be a list of 1-14 pairs")
    pairs = []
    for e in edges:
        if not (isinstance(e, list) and len(e) == 2 and all(isinstance(x, str) and x in nodes for x in e)) or e[0] == e[1]:
            raise SpecError("each edge must be two different node labels from nodes")
        pairs.append([nodes.index(e[0]), nodes.index(e[1])])
    start = params.get("start", nodes[0])
    if start not in nodes:
        raise SpecError("start must be one of the nodes")
    adj: dict[int, list[int]] = {i: [] for i in range(len(nodes))}
    for a, b in pairs:
        adj[a].append(b)
        adj[b].append(a)
    s, order, seen = nodes.index(start), [], set()
    if algo == "bfs":
        queue = [s]
        seen.add(s)
        while queue:
            v = queue.pop(0)
            order.append(v)
            for w in sorted(adj[v]):
                if w not in seen:
                    seen.add(w)
                    queue.append(w)
    else:
        def dfs(v: int) -> None:
            seen.add(v)
            order.append(v)
            for w in sorted(adj[v]):
                if w not in seen:
                    dfs(w)
        dfs(s)
    return {"algorithm": algo, "nodes": nodes, "edges": pairs, "order": order}


def _message_flow(params: dict) -> dict:
    _only(params, {"actors", "messages"})
    actors = params.get("actors")
    if not isinstance(actors, list) or not 2 <= len(actors) <= 5:
        raise SpecError("actors must be a list of 2-5 names")
    actors = [_label(a, "actor", 16) for a in actors]
    if len(set(actors)) != len(actors):
        raise SpecError("actor names must be unique")
    messages = params.get("messages")
    if not isinstance(messages, list) or not 1 <= len(messages) <= 10:
        raise SpecError("messages must be a list of 1-10 items")
    out = []
    for m in messages:
        if not (isinstance(m, dict) and set(m) <= {"from", "to", "label"}) or m.get("from") not in actors or m.get("to") not in actors:
            raise SpecError("each message needs from, to (actor names) and label")
        out.append({"from": actors.index(m["from"]), "to": actors.index(m["to"]), "label": _label(m.get("label"), "label", 28, clip=True)})
    return {"actors": actors, "messages": out}


def _state_machine(params: dict) -> dict:
    _only(params, {"states", "transitions", "path"})
    states = params.get("states")
    if not isinstance(states, list) or not 2 <= len(states) <= 6:
        raise SpecError("states must be a list of 2-6 names")
    states = [_label(s, "state", 14) for s in states]
    if len(set(states)) != len(states):
        raise SpecError("state names must be unique")
    transitions = params.get("transitions")
    if not isinstance(transitions, list) or not 1 <= len(transitions) <= 10:
        raise SpecError("transitions must be a list of 1-10 items")
    trans = []
    for t in transitions:
        if not (isinstance(t, dict) and set(t) <= {"from", "to", "label"}) or t.get("from") not in states or t.get("to") not in states \
                or t["from"] == t["to"]:
            raise SpecError("each transition needs from, to (two different states) and label")
        trans.append({"from": states.index(t["from"]), "to": states.index(t["to"]), "label": _label(t.get("label"), "label", 16)})
    path = params.get("path", [])
    if not isinstance(path, list) or len(path) > 10 or any(p not in states for p in path):
        raise SpecError("path must be a list of up to 10 state names")
    return {"states": states, "transitions": trans, "path": [states.index(p) for p in path]}


def _text_steps(params: dict) -> dict:
    _only(params, {"steps"})
    steps = params.get("steps")
    if not isinstance(steps, list) or not 2 <= len(steps) <= 8:
        raise SpecError("steps must be a list of 2-8 short lines")
    return {"steps": [_label(s, "step", 48, clip=True) for s in steps]}


# ----------------------------------------------------------------------------- scene code (trusted, hand-written)

_HEADER = '''import base64, json
from manim import *

DATA = json.loads(base64.b64decode("%s").decode("utf-8"))
config.background_color = "#0f0f14"
'''

_SCENES: dict[str, str] = {
"binary_search": '''
class GeneratedScene(Scene):
    def construct(self):
        arr, steps, target = DATA["array"], DATA["steps"], DATA["target"]
        n = len(arr)
        self.add(Text(DATA["title"], font_size=30).to_edge(UP))
        w = min(0.9, 11.0 / n)
        cells = VGroup(*[Square(side_length=w).set_stroke(WHITE, 2) for _ in range(n)]).arrange(RIGHT, buff=0.05)
        labels = VGroup(*[Text(str(v), font_size=max(14, int(w * 28))).move_to(cells[i]) for i, v in enumerate(arr)])
        idx = VGroup(*[Text(str(i), font_size=14, color=GRAY).next_to(cells[i], DOWN, buff=0.1) for i in range(n)])
        goal = Text("target = " + str(target), font_size=26).next_to(cells, UP, buff=0.9)
        self.play(FadeIn(cells), FadeIn(labels), FadeIn(idx), FadeIn(goal))
        for s in steps:
            lo, hi, mid = s["low"], s["high"], s["mid"]
            anims = []
            for i in range(n):
                inside = lo <= i <= hi
                color = YELLOW if i == mid else (BLUE_E if inside else BLACK)
                anims.append(cells[i].animate.set_fill(color, opacity=0.65 if inside else 0.12))
            note = Text("low=%d  high=%d  mid=%d : %d %s %d" % (lo, hi, mid, arr[mid], s["cmp"], target), font_size=24).to_edge(DOWN)
            self.play(*anims, FadeIn(note), run_time=0.8)
            self.wait(0.6)
            self.play(FadeOut(note), run_time=0.3)
        end = Text(DATA["result"], font_size=30, color=GREEN if DATA["found"] else RED).to_edge(DOWN)
        self.play(FadeIn(end))
        self.wait(1)
''',
"sorting": '''
class GeneratedScene(Scene):
    def bars(self, arr, hl):
        n = len(arr)
        w = min(0.8, 10.0 / n)
        group = VGroup()
        for i, v in enumerate(arr):
            r = Rectangle(width=w, height=0.18 * v + 0.2, fill_opacity=0.85, stroke_width=1)
            r.set_fill(YELLOW if i in hl else BLUE_D)
            group.add(r)
        group.arrange(RIGHT, buff=0.12, aligned_edge=DOWN)
        group.move_to(ORIGIN + DOWN * 0.3)
        return group

    def construct(self):
        frames = DATA["frames"]
        self.add(Text(DATA["title"], font_size=30).to_edge(UP))
        current = self.bars(frames[0]["arr"], [])
        note = Text(frames[0]["note"], font_size=24).to_edge(DOWN)
        self.play(FadeIn(current), FadeIn(note))
        for f in frames[1:]:
            nxt = self.bars(f["arr"], f["hl"])
            nn = Text(f["note"], font_size=24).to_edge(DOWN)
            self.play(Transform(current, nxt), Transform(note, nn), run_time=0.35)
        self.wait(1)
''',
"recursion_stack": '''
class GeneratedScene(Scene):
    def construct(self):
        self.add(Text(DATA["title"], font_size=30).to_edge(UP))
        base_y, h = -2.8, 0.55
        stack = []
        for ev in DATA["events"]:
            if ev["op"] == "push":
                box = Rectangle(width=3.6, height=h, fill_opacity=0.6, stroke_width=2).set_fill(BLUE_D)
                label = Text(ev["text"], font_size=22).move_to(box)
                item = VGroup(box, label)
                item.move_to([0, base_y + h * len(stack) + 0.0, 0])
                item.shift(UP * 3)
                self.play(item.animate.shift(DOWN * 3), run_time=0.4)
                stack.append(item)
            else:
                item = stack.pop()
                done = Text(ev["text"], font_size=22).move_to(item)
                self.play(Transform(item[1], done), item[0].animate.set_fill(GREEN_D), run_time=0.3)
                self.play(FadeOut(item), run_time=0.25)
        self.play(FadeIn(Text("result = " + str(DATA["result"]), font_size=30, color=GREEN).to_edge(DOWN)))
        self.wait(1)
''',
"graph_traversal": '''
class GeneratedScene(Scene):
    def construct(self):
        nodes, edges, order = DATA["nodes"], DATA["edges"], DATA["order"]
        self.add(Text(DATA["title"], font_size=30).to_edge(UP))
        g = Graph(list(range(len(nodes))), [tuple(e) for e in edges], layout="circular",
                  labels={i: Text(n, font_size=24) for i, n in enumerate(nodes)},
                  vertex_config={"radius": 0.32}, layout_scale=2.4)
        self.play(Create(g))
        seq = Text("order:", font_size=24).to_edge(DOWN)
        self.add(seq)
        visited = []
        for v in order:
            visited.append(nodes[v])
            nxt = Text("order: " + " -> ".join(visited), font_size=24).to_edge(DOWN)
            self.play(g[v].animate.set_color(YELLOW), Transform(seq, nxt), run_time=0.7)
            self.play(g[v].animate.set_color(GREEN), run_time=0.2)
        self.wait(1)
''',
"message_flow": '''
class GeneratedScene(Scene):
    def construct(self):
        actors, msgs = DATA["actors"], DATA["messages"]
        self.add(Text(DATA["title"], font_size=30).to_edge(UP))
        k = len(actors)
        xs = [(-5.0 + 10.0 * i / (k - 1)) for i in range(k)]
        top, bottom = 2.4, -3.2
        heads, lines = VGroup(), VGroup()
        for x, name in zip(xs, actors):
            box = Rectangle(width=1.9, height=0.55, stroke_width=2).move_to([x, top, 0])
            heads.add(VGroup(box, Text(name, font_size=20).move_to(box)))
            lines.add(DashedLine([x, top - 0.3, 0], [x, bottom, 0], stroke_width=2, color=GRAY))
        self.play(FadeIn(heads), Create(lines))
        step = (top - 0.5 - bottom) / (len(msgs) + 1)
        for i, m in enumerate(msgs):
            y = top - 0.4 - step * (i + 1)
            a, b = xs[m["from"]], xs[m["to"]]
            arrow = Arrow([a, y, 0], [b, y, 0], buff=0, color=YELLOW, stroke_width=4, max_tip_length_to_length_ratio=0.15)
            text = Text(m["label"], font_size=20).next_to(arrow, UP, buff=0.05)
            self.play(GrowArrow(arrow), FadeIn(text), run_time=0.6)
            self.wait(0.3)
        self.wait(1)
''',
"state_machine": '''
class GeneratedScene(Scene):
    def construct(self):
        states, trans, path = DATA["states"], DATA["transitions"], DATA["path"]
        self.add(Text(DATA["title"], font_size=30).to_edge(UP))
        n = len(states)
        pos = [2.3 * np.array([np.cos(TAU * i / n + PI / 2), np.sin(TAU * i / n + PI / 2), 0]) + DOWN * 0.3 for i in range(n)]
        nodes = []
        for p, name in zip(pos, states):
            c = Circle(radius=0.55, stroke_width=3).move_to(p)
            nodes.append(VGroup(c, Text(name, font_size=18).move_to(c)))
        arrows = VGroup()
        for t in trans:
            a, b = pos[t["from"]], pos[t["to"]]
            d = (b - a) / np.linalg.norm(b - a)
            arr = Arrow(a + d * 0.6, b - d * 0.6, buff=0, stroke_width=3, color=GRAY, max_tip_length_to_length_ratio=0.2)
            lab = Text(t["label"], font_size=16).move_to(arr.get_center() + np.array([-d[1], d[0], 0]) * 0.28)
            arrows.add(VGroup(arr, lab))
        self.play(*[FadeIn(nd) for nd in nodes], FadeIn(arrows))
        token = Dot(color=YELLOW, radius=0.12)
        if path:
            token.move_to(pos[path[0]])
            self.play(FadeIn(token), nodes[path[0]][0].animate.set_fill(YELLOW, opacity=0.35))
            for prev, cur in zip(path, path[1:]):
                self.play(nodes[prev][0].animate.set_fill(BLACK, opacity=0), token.animate.move_to(pos[cur]),
                          nodes[cur][0].animate.set_fill(YELLOW, opacity=0.35), run_time=0.8)
        self.wait(1)
''',
"text_steps": '''
class GeneratedScene(Scene):
    def construct(self):
        self.add(Text(DATA["title"], font_size=30).to_edge(UP))
        current = Text(DATA["steps"][0], font_size=36)
        self.play(Write(current))
        self.wait(0.8)
        for s in DATA["steps"][1:]:
            nxt = Text(s, font_size=36)
            self.play(Transform(current, nxt), run_time=0.9)
            self.wait(0.9)
        self.wait(0.5)
''',
}


@dataclass(frozen=True)
class Template:
    name: str
    description: str
    example: dict
    build: Callable[[dict], dict]


TEMPLATES: dict[str, Template] = {t.name: t for t in (
    Template("binary_search", "Binary search narrowing a sorted array", {"array": [2, 5, 8, 12, 16, 23, 38], "target": 23}, _binary_search),
    Template("sorting", "Bubble, insertion or selection sort on up to 10 numbers", {"algorithm": "bubble", "array": [5, 2, 8, 1, 9]}, _sorting),
    Template("recursion_stack", "Call stack of factorial or fibonacci", {"function": "factorial", "n": 4}, _recursion_stack),
    Template("graph_traversal", "BFS or DFS over a small graph", {"algorithm": "bfs", "nodes": ["A", "B", "C", "D"], "edges": [["A", "B"], ["A", "C"], ["B", "D"]], "start": "A"}, _graph_traversal),
    Template("message_flow", "Messages between up to 5 actors (sequence diagram)", {"actors": ["Client", "Server"], "messages": [{"from": "Client", "to": "Server", "label": "request"}, {"from": "Server", "to": "Client", "label": "response"}]}, _message_flow),
    Template("state_machine", "A finite-state machine with a highlighted path", {"states": ["Idle", "Running", "Done"], "transitions": [{"from": "Idle", "to": "Running", "label": "start"}, {"from": "Running", "to": "Done", "label": "finish"}], "path": ["Idle", "Running", "Done"]}, _state_machine),
    Template("text_steps", "A short derivation or explanation shown line by line", {"steps": ["2x + 3 = 11", "2x = 8", "x = 4"]}, _text_steps),
)}


@dataclass(frozen=True)
class ValidatedSpec:
    template: str
    title: str
    params: dict          # normalised, bounded copy of the model's parameters

    def canonical_json(self) -> str:
        return json.dumps({"template": self.template, "title": self.title, "params": self.params}, sort_keys=True, separators=(",", ":"))


def validate_spec(spec: Any, max_chars: int = 4000) -> ValidatedSpec:
    """Strictly validate a model-written specification. Raises SpecError (unsupported=True for an unknown template)."""
    if not isinstance(spec, dict):
        raise SpecError("the specification must be a JSON object")
    if len(json.dumps(spec, default=str)) > max_chars:
        raise SpecError("the specification is too large")
    extra = set(spec) - {"schema_version", "template", "title", "params"}
    if extra:
        raise SpecError(f"unknown field(s): {sorted(extra)[:3]}")
    if spec.get("schema_version", SPEC_VERSION) != SPEC_VERSION:
        raise SpecError(f"unsupported schema_version (expected {SPEC_VERSION})")
    name = spec.get("template")
    template = TEMPLATES.get(name) if isinstance(name, str) else None
    if template is None:
        raise SpecError(f"unknown template {str(name)[:30]!r}; supported: {', '.join(TEMPLATES)}", unsupported=True)
    params = spec.get("params", {})
    return ValidatedSpec(
        template=template.name,
        title=_label(spec.get("title", template.description[:MAX_TITLE]), "title", MAX_TITLE),
        params=_normalised(template, params),
    )


def _normalised(template: Template, params: Any) -> dict:
    if not isinstance(params, dict):
        raise SpecError("params must be an object")
    built = template.build(params)          # raises SpecError on anything invalid
    return json.loads(json.dumps(params, sort_keys=True)) if built else {}


def build_scene_source(spec: ValidatedSpec) -> str:
    """Python source of the scene for a VALIDATED spec. Data is passed as base64 JSON, never interpolated as code."""
    template = TEMPLATES[spec.template]
    data = template.build(spec.params)
    data["title"] = spec.title
    encoded = base64.b64encode(json.dumps(data, separators=(",", ":")).encode("utf-8")).decode("ascii")
    return (_HEADER % encoded) + _SCENES[spec.template]


def template_catalog() -> list[dict]:
    return [{"name": t.name, "description": t.description, "example": t.example} for t in TEMPLATES.values()]
