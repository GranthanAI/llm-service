from app.exceptions.analysis import ContextOverflowError
from app.exceptions.base import PermanentError
from app.utils.error_handler import ErrorHandler


def test_context_overflow_tells_the_user_what_to_do_instead_of_the_generic_text():
    exc = ContextOverflowError("The request needs 5753 tokens but at most 5752 fit the context budget for this model. "
                               "Shorten the message or start a new conversation.")
    c = ErrorHandler().classify(exc)
    assert c.error_code == "CONTEXT_OVERFLOW" and not c.is_retriable
    assert "start a new conversation" in c.user_message and "5753" in c.user_message
    assert "Unable to complete request" not in c.user_message


def test_other_permanent_errors_keep_the_generic_message():
    c = ErrorHandler().classify(PermanentError("boom", error_code="X"))
    assert c.user_message == "Unable to complete request with provided input."
