"""
Provider configuration schemas and defaults.
Implements LLD v2.0 Section 19.3.
"""

from pydantic import BaseModel, SecretStr


class ProviderConfig(BaseModel):
    model: str
    api_key: SecretStr
    base_url: str | None = None
    timeout_ms: int = 15000
    max_retries: int = 3
