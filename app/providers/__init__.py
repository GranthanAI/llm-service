"""
Provider Adapters and Generation Router Package.
Implements LLD v2.0 Section 19 and Section 20.
"""

from app.providers.base import BaseProviderAdapter
from app.providers.gemini import GeminiAdapter
from app.providers.groq import GroqAdapter
from app.providers.models import GenerationResponse, ProviderConfig, ProviderSelection
from app.providers.nvidia import NVIDIAAdapter
from app.providers.router import GenerationRouter

__all__ = [
    "BaseProviderAdapter",
    "GeminiAdapter",
    "GenerationResponse",
    "GenerationRouter",
    "GroqAdapter",
    "NVIDIAAdapter",
    "ProviderConfig",
    "ProviderSelection",
]
