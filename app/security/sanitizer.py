"""
Input Sanitization and Prompt Injection Defense.
Implements LLD v2.0 Section 26.2.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import structlog

from app.config.logging import get_logger

if TYPE_CHECKING:
    from app.models.pipeline_context import PipelineContext

logger = get_logger("input_sanitizer")

# Known prompt structure delimiters to escape
PROMPT_DELIMITERS = [
    (r"###", r"\#\#\#"),
    (r"<\|im_start\|>", r"\<\|im_start\|\>"),
    (r"<\|im_end\|>", r"\<\|im_end\|\>"),
    (r"\[INST\]", r"\[INST\]_escaped"),
    (r"\[/INST\]", r"\[/INST\]_escaped"),
    (r"<<SYS>>", r"\<\<SYS\>\>_escaped"),
    (r"<</SYS>>", r"\<\</SYS\>\>_escaped"),
]

# Common jailbreak and instruction override patterns
JAILBREAK_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?previous\s+(instructions|prompts|rules)", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?prior\s+(guidelines|instructions)", re.IGNORECASE),
    re.compile(r"system\s+prompt\s+override", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+in\s+developer\s+mode", re.IGNORECASE),
    re.compile(r"do\s+anything\s+now\s+mode", re.IGNORECASE),
    re.compile(r"dan\s+mode\s+enabled", re.IGNORECASE),
]


class InputSanitizer:
    """
    Sanitizes user messages before Request Analysis and Prompt Building.
    Protects against delimiter injection, jailbreak overrides, and oversized payloads.
    """

    def __init__(
        self,
        max_chars: int = 4096,
        logger_instance: structlog.stdlib.BoundLogger | None = None,
    ):
        self.max_chars: int = max_chars
        self.logger = logger_instance or logger

    def sanitize(
        self,
        user_message: str,
        ctx: PipelineContext | None = None,
    ) -> str:
        """
        Executes delimiter escaping, instruction override detection, and length enforcement.
        Returns sanitized user message.
        """
        if not user_message:
            return ""

        # 1. Length enforcement (max 4096 chars)
        sanitized = user_message[: self.max_chars]

        # 2. Delimiter injection check and escaping
        for pattern, replacement in PROMPT_DELIMITERS:
            sanitized = re.sub(pattern, replacement, sanitized)

        # 3. Instruction override / jailbreak pattern detection
        jailbreak_detected = False
        for pattern in JAILBREAK_PATTERNS:
            if pattern.search(user_message):
                jailbreak_detected = True
                break

        if jailbreak_detected:
            self.logger.warning(
                "Potential prompt injection / instruction override pattern detected in input",
                conversation_id=ctx.conversation_id if ctx else None,
                message_id=ctx.message_id if ctx else None,
            )
            if ctx:
                ctx.safety_check_failed = True

        return sanitized
