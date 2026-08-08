"""
Unit tests for exceptions and mode handler state.
"""


from app.context.schemas import ContextBundle
from app.exceptions.analysis import (
    CriticalAnalysisError,
    PlanParseError,
    UnknownModeError,
)
from app.exceptions.grpc import (
    GRPCTimeoutError,
    GRPCUnavailableError,
)
from app.exceptions.provider import (
    AllProvidersFailedError,
    CircuitOpenError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from app.exceptions.tool import (
    RequiredToolFailedError,
    ToolError,
    ToolTimeoutError,
    ToolValidationError,
    UnknownToolError,
)
from app.models.execution_plan import ExecutionPlan
from app.workflow_engine.mode_handlers.state import ModeHandlerState


def test_exception_hierarchies():
    # Provider
    pe = ProviderError("fail", provider="nvidia", status_code=500)
    assert pe.provider == "nvidia"
    assert issubclass(AllProvidersFailedError, ProviderError)
    assert issubclass(ProviderTimeoutError, ProviderError)
    assert issubclass(ProviderRateLimitError, ProviderError)
    assert issubclass(CircuitOpenError, ProviderError)

    # Tool
    te = ToolError("tool failed", tool_name="web_search")
    assert te.tool_name == "web_search"
    assert issubclass(RequiredToolFailedError, ToolError)
    assert issubclass(ToolTimeoutError, ToolError)
    assert issubclass(UnknownToolError, ToolError)
    assert issubclass(ToolValidationError, ToolError)

    # gRPC
    ge = GRPCUnavailableError("unavailable", service="memory_service", code=14)
    assert ge.service == "memory_service"
    assert issubclass(GRPCTimeoutError, Exception)

    # Analysis
    ume = UnknownModeError("invalid_mode")
    assert ume.mode == "invalid_mode"
    assert issubclass(PlanParseError, Exception)
    assert issubclass(CriticalAnalysisError, Exception)


def test_mode_handler_state():
    state = ModeHandlerState(
        conversation_id="conv_1",
        user_id="user_1",
        request_id="req_1",
        user_message="test message",
        plan=ExecutionPlan(),
        context_bundle=ContextBundle(),
    )
    assert state.conversation_id == "conv_1"
    assert len(state.tool_results) == 0
