"""
Personally Identifiable Information (PII) Detection and Redaction.
Implements LLD v2.0 Section 26.4.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import structlog

from app.config.logging import get_logger

if TYPE_CHECKING:
    from app.models.pipeline_context import PipelineContext

logger = get_logger("pii_detector")

# Regex patterns matching LLD Section 26.4
PII_PATTERNS = [
    (
        "EMAIL",
        re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
        "[EMAIL_REDACTED]",
    ),
    (
        "PHONE_US",
        re.compile(r"\b(\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
        "[PHONE_REDACTED]",
    ),
    (
        "SSN",
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "[SSN_REDACTED]",
    ),
    (
        "CREDIT_CARD",
        re.compile(r"\b(?:\d[ -]?){13,16}\b"),
        "[CARD_REDACTED]",
    ),
]


class PIIDetector:
    """
    Detects and redacts Personally Identifiable Information (PII) from user messages
    prior to sending prompts to external LLM providers.
    """

    def __init__(self, logger_instance: structlog.stdlib.BoundLogger | None = None):
        self.logger = logger_instance or logger

    def detect_and_redact(
        self,
        text: str,
        ctx: PipelineContext | None = None,
    ) -> tuple[str, bool]:
        """
        Scans text for PII patterns, replaces matches with redaction tokens,
        and sets ctx.pii_detected = True if any match is found.
        Returns: (redacted_text, was_pii_detected)
        """
        if not text:
            return "", False

        redacted_text = text
        pii_found = False
        detected_types: list[str] = []

        for pii_type, pattern, placeholder in PII_PATTERNS:
            if pattern.search(redacted_text):
                pii_found = True
                detected_types.append(pii_type)
                redacted_text = pattern.sub(placeholder, redacted_text)

        if pii_found:
            self.logger.info(
                "PII detected and redacted from message",
                pii_types=detected_types,
                conversation_id=ctx.conversation_id if ctx else None,
                message_id=ctx.message_id if ctx else None,
            )
            if ctx:
                ctx.pii_detected = True

        return redacted_text, pii_found
