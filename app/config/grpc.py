"""
gRPC client configuration models and defaults.
"""

from pydantic import BaseModel


class GRPCServiceConfig(BaseModel):
    host: str
    port: int
    deadline_ms: int = 2000
    max_connections: int = 20
