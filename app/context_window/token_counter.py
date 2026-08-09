"""
Token Counter with Provider-Specific Multipliers.
Implements LLD v2.0 Section 18.2.
"""

from typing import Any

import tiktoken


class TokenCounter:
    """
    High-performance token counter using tiktoken with provider calibration.
    """

    PROVIDER_MULTIPLIERS: dict[str, float] = {
        "nvidia": 1.00,
        "groq": 1.00,
        "gemini": 0.97,
        "openai": 1.00,
    }

    def __init__(self, encoding_name: str = "cl100k_base"):
        self.encoding_name = encoding_name
        try:
            self.tokenizer = tiktoken.get_encoding(encoding_name)
        except Exception:
            self.tokenizer = None

    def count_text(self, text: str, provider: str = "nvidia") -> int:
        """Counts calibrated tokens in a text string."""
        if not text:
            return 0

        multiplier = self.PROVIDER_MULTIPLIERS.get(provider.lower(), 1.00)

        if self.tokenizer is not None:
            try:
                base_count = len(self.tokenizer.encode(text))
                return max(1, int(base_count * multiplier))
            except Exception:
                pass

        # Fallback heuristic: ~4 characters per token
        char_tokens = max(1, len(text) // 4)
        return max(1, int(char_tokens * multiplier))

    def count_messages(self, messages: list[dict[str, Any]], provider: str = "nvidia") -> int:
        """
        Counts tokens across a list of OpenAI-formatted chat messages,
        including role and per-message overhead (approx 4 tokens per turn).
        """
        total = 0
        for msg in messages:
            content = str(msg.get("content", ""))
            total += self.count_text(content, provider) + 4
        total += 2  # priming tokens
        return total

    def truncate_text_to_tokens(
        self, text: str, target_tokens: int, provider: str = "nvidia"
    ) -> str:
        """Truncates text so that its token count does not exceed target_tokens."""
        if target_tokens <= 0:
            return ""

        current_tokens = self.count_text(text, provider)
        if current_tokens <= target_tokens:
            return text

        if self.tokenizer is not None:
            try:
                tokens = self.tokenizer.encode(text)
                # Adjust for provider multiplier
                multiplier = self.PROVIDER_MULTIPLIERS.get(provider.lower(), 1.00)
                adjusted_target = max(1, int(target_tokens / multiplier))
                trimmed_tokens = tokens[:adjusted_target]
                return self.tokenizer.decode(trimmed_tokens)
            except Exception:
                pass

        # Fallback character-ratio slicing
        char_target = max(10, target_tokens * 4)
        return text[:char_target]
