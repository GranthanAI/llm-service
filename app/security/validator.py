"""
Output Validation and Secret Leak Prevention.
Implements LLD v2.0 Section 26.3.
"""

import re

import structlog

from app.config.logging import get_logger

logger = get_logger("output_validator")

# Known API key signature patterns to prevent leaks in generated output
SECRET_PATTERNS = [
    re.compile(r"nvapi-[a-zA-Z0-9_-]{32,}"),
    re.compile(r"AIza[0-9A-Za-z-_]{30,}"),
    re.compile(r"gsk_[a-zA-Z0-9]{32,}"),
    re.compile(r"sk-[a-zA-Z0-9]{32,}"),
    re.compile(r"tvly-[a-zA-Z0-9_-]{24,}"),
]


class OutputValidator:
    """
    Validates model completion outputs before streaming or publishing to Kafka.
    Prevents secret leaks, ensures UTF-8 encoding validity, and checks output length.
    """

    def __init__(
        self,
        min_chars: int = 1,
        max_chars: int = 200_000,
        logger_instance: structlog.stdlib.BoundLogger | None = None,
    ):
        self.min_chars: int = min_chars
        self.max_chars: int = max_chars
        self.logger = logger_instance or logger

    def validate(self, text: str) -> bool:
        """
        Validates output content. Returns True if valid, False if violations are detected.
        """
        if not text or len(text.strip()) < self.min_chars:
            self.logger.warning("Output validation failed: empty or whitespace-only response")
            return False

        if len(text) > self.max_chars:
            self.logger.warning(
                "Output validation failed: response exceeds max char limit", length=len(text)
            )
            return False

        # Ensure valid UTF-8
        try:
            text.encode("utf-8")
        except UnicodeEncodeError as exc:
            self.logger.error("Output validation failed: invalid UTF-8 encoding", error=str(exc))
            return False

        # Secret leak check
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                self.logger.critical(
                    "Output validation failed: potential API key / secret leak detected in generated response!"
                )
                return False

        return True

    def sanitize_output(self, text: str) -> str:
        """
        Removes any detected secret patterns from output if present.
        """
        sanitized = text
        for pattern in SECRET_PATTERNS:
            sanitized = pattern.sub("[REDACTED_API_KEY]", sanitized)
        return sanitized
