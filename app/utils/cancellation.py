"""
Cancellation Token Implementation.
Implements LLD v2.0 Section 21.4 for cooperative async task and stream aborts.
"""


class CancellationToken:
    """
    Cooperative cancellation token passed through async workflows and streams.
    Allows clients or disconnect handlers to signal immediate abort.
    """

    def __init__(self) -> None:
        self._cancelled: bool = False
        self._reason: str = ""

    def cancel(self, reason: str = "client_disconnected") -> None:
        """Signal cancellation with an optional reason."""
        self._cancelled = True
        self._reason = reason

    @property
    def is_cancelled(self) -> bool:
        """Check if cancellation has been requested."""
        return self._cancelled

    @property
    def reason(self) -> str:
        """Get the cancellation reason."""
        return self._reason

    def __repr__(self) -> str:
        return f"<CancellationToken cancelled={self._cancelled} reason='{self._reason}'>"
