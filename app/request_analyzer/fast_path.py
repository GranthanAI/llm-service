"""
Conservative fast-path matcher for trivial greetings / small talk.

A message qualifies only if the *entire* normalised text is one of a small set of
social phrases. Anything else (a question, a reference to earlier context, an
attached file, a mode hint, a tool request) goes through the full pipeline.
"""

import re

_MAX_LEN = 40

# Words that can follow/precede a greeting without changing its meaning.
_FILLER = r"(?:there|everyone|all|team|friend|bot|assistant|granthan(?:ai)?)"

_PHRASES = [
    r"(?:hi+|hello+|hey+|hiya|howdy|yo|sup|greetings|hola|namaste)",
    r"good\s+(?:morning|afternoon|evening|night|day)",
    r"(?:what'?s|whats)\s+up",
    r"how\s+are\s+(?:you|u)(?:\s+doing)?(?:\s+today)?",
    r"how'?s\s+it\s+going",
    r"(?:thanks|thank\s+you|thx|ty|thanks\s+a\s+lot|thank\s+you\s+(?:so\s+much|very\s+much))",
    r"(?:ok|okay|cool|great|nice|awesome|got\s+it|alright|sure|perfect)",
    r"(?:bye|goodbye|see\s+you|see\s+ya|cya|good\s+night)",
]

# One or two phrases, each optionally followed by a filler word ("hi there", "hello, how are you").
_PHRASE = rf"(?:{'|'.join(_PHRASES)})(?:\s+{_FILLER})?"
_PATTERN = re.compile(rf"^{_PHRASE}(?:\s*(?:,|and|&)?\s*{_PHRASE})?$", re.IGNORECASE)

_STRIP = re.compile(r"[\s!.?,~:;)(\-_*\U0001F300-\U0001FAFF☀-➿]+$|^[\s!.?,~:;)(\-_*\U0001F300-\U0001FAFF☀-➿]+")


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", _STRIP.sub("", text.strip())).strip().lower()


def is_simple_smalltalk(
    text: str,
    *,
    file_ids: list[str] | None = None,
    mode_hint: object | None = None,
) -> bool:
    """True only for trivially conversational messages that need no context or tools."""
    if file_ids or mode_hint:
        return False
    if not text or len(text) > _MAX_LEN * 2:
        return False
    norm = normalize(text)
    if not norm or len(norm) > _MAX_LEN:
        return False
    return _PATTERN.match(norm) is not None
