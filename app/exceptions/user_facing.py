"""
Errors whose message is written for the end user.

The generic classifier hides internals behind "An unexpected error occurred". These errors are
different: the situation (an unknown tool, a document that is still being indexed) is something the
user can act on, so the text is safe and meant to be shown verbatim in the chat.
"""

from app.exceptions.base import PermanentError


class UserFacingError(PermanentError):
    """Deterministic, user-actionable failure with a safe message."""

    error_type = "VALIDATION"

    def __init__(self, message: str, user_message: str, error_code: str, details: dict | None = None):
        super().__init__(message, error_code=error_code, details=details)
        self.user_message = user_message


class InvalidToolSelectionError(UserFacingError):
    """The user explicitly selected a tool that is unknown, disabled, or cannot be run as requested."""

    error_type = "TOOL"

    def __init__(self, tool_names: list[str], reason: str):
        names = ", ".join(f"'{n}'" for n in tool_names)
        super().__init__(
            f"Invalid tool selection {names}: {reason}",
            user_message=(
                f"The selected tool {names} is not available ({reason}). "
                "Remove it or choose another tool and send the message again."
            ),
            error_code="INVALID_TOOL_SELECTION",
            details={"tools": tool_names, "reason": reason},
        )


class InvalidModeError(UserFacingError):
    """The user explicitly chose a response mode that is unknown or turned off."""

    def __init__(self, mode: str, reason: str):
        super().__init__(
            f"Invalid mode '{mode}': {reason}",
            user_message=f"The selected mode '{mode}' is not available ({reason}). Choose another mode, or Auto, and send again.",
            error_code="INVALID_MODE",
            details={"mode": mode, "reason": reason},
        )


# What the user should do for each retrieval outcome that means "the attached document could not be used".
_DOCUMENT_MESSAGES = {
    "ACCESS_DENIED": "You don't have access to the attached document, or it no longer exists. Re-upload it and try again.",
    "NOT_INDEXED": "The attached document has not been indexed yet. Wait a moment and try again.",
    "INDEXING": "The attached document is still being processed. Try again in a few seconds.",
    "INDEXING_FAILED": "The attached document could not be processed, so it can't be searched. Re-upload it or retry its processing.",
    "UNAVAILABLE": "The document search service is temporarily unavailable, so the attached document could not be read. Please try again.",
    "TIMEOUT": "Searching the attached document took too long. Please try again.",
}


class DocumentContextUnavailableError(UserFacingError):
    """Documents were attached to the question but could not be searched. The model must NOT answer
    from general knowledge as if it had read them."""

    error_type = "GRPC"

    def __init__(self, status: str, detail: str = ""):
        super().__init__(
            f"Attached document context unavailable: {status} ({detail})",
            user_message=_DOCUMENT_MESSAGES.get(
                status, "The attached document could not be read right now. Please try again."
            ),
            error_code=f"DOCUMENT_{status}",
            details={"retrieval_status": status, "detail": detail},
        )
        self.status = status
