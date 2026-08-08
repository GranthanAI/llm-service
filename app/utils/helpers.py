"""
General helper functions and string utilities.
"""

import uuid


def generate_request_id() -> str:
    """Generate a UUID4 string for request correlation."""
    return str(uuid.uuid4())
