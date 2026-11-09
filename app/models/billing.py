"""
The optional `billing` block of chat.message.created 2.4 (billing integration plan 4.4.1).

Everything is optional: an event without the block (schema 2.3, or conversation-service not yet updated) takes the
legacy routing path exactly as before.
"""

from typing import Literal

from pydantic import BaseModel


class BillingGrant(BaseModel):
    model_config = {"extra": "ignore"}  # consumers ignore unknown fields

    hold_id: str | None = None                 # null in degraded mode, or after a shadow-mode deny
    requested_model: str | None = None         # null = Auto
    granted_group: str | None = None           # null = legacy routing
    granted_model: str | None = None           # null = llm-service picks inside the group
    max_context_tokens: int | None = None
    funded_by: Literal["plan", "credits"] | None = None
    downgraded: bool = False
    enforcement: Literal["enforce", "shadow", "degraded"] | None = None
