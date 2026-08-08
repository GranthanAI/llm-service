"""
Request Analysis Exceptions.
"""


class AnalysisError(Exception):
    """Base exception for Request Analyzer failures."""

    pass


class PlanParseError(AnalysisError):
    """Raised when JSON parsing of Groq analysis output fails."""

    pass


class CriticalAnalysisError(AnalysisError):
    """Raised when analysis fails critically and cannot recover."""

    pass


class UnknownModeError(AnalysisError):
    """Raised when execution plan specifies an unregistered or unsupported mode."""

    def __init__(self, mode: str):
        super().__init__(f"Unknown or unsupported mode: '{mode}'")
        self.mode = mode


class ContextOverflowError(Exception):
    """Raised when prompt cannot fit into model context window even after maximal trimming."""

    pass
