"""Time ranges of YouTube transcript chunks.

Retrieval Service encodes a chunk's position in the video in the chunk's `section` as ``t=<start>-<end>`` (whole seconds).
These helpers decode it for prompts and citations; anything that is not exactly that form is "not a time range" (documents
keep their own section headings untouched).
"""

import re

_RANGE = re.compile(r"^t=(\d{1,7})-(\d{1,7})$")


def parse_time_section(section: str | None) -> tuple[int, int] | None:
    m = _RANGE.match(section or "")
    if not m:
        return None
    start, end = int(m.group(1)), int(m.group(2))
    return (start, max(start, end))


def format_timecode(seconds: int) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def format_range(start: int, end: int) -> str:
    return format_timecode(start) if end <= start else f"{format_timecode(start)}–{format_timecode(end)}"
