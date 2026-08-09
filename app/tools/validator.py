"""
Tool Parameter Validation Engine.
Implements LLD v2.0 Section 15.1.
"""

from typing import Any

from app.models.tool import ToolParams, ValidationResult


class ToolValidator:
    """Validator utility for tool input parameters."""

    @staticmethod
    def validate_required_fields(
        params: ToolParams, required_fields: list[str]
    ) -> ValidationResult:
        """Validate that all required parameter keys exist and are non-empty."""
        if not isinstance(params.params, dict):
            return ValidationResult(valid=False, error="Tool parameters must be a dictionary")

        missing = [f for f in required_fields if f not in params.params or params.params[f] is None]
        if missing:
            return ValidationResult(
                valid=False,
                error=f"Missing required parameter(s): {', '.join(missing)}",
            )
        return ValidationResult(valid=True)

    @staticmethod
    def validate_field_types(
        params: ToolParams, field_types: dict[str, type | tuple[type, ...]]
    ) -> ValidationResult:
        """Validate parameter types against expected Python types."""
        if not isinstance(params.params, dict):
            return ValidationResult(valid=False, error="Tool parameters must be a dictionary")

        for field_name, expected_type in field_types.items():
            if field_name in params.params:
                val: Any = params.params[field_name]
                if val is not None and not isinstance(val, expected_type):
                    return ValidationResult(
                        valid=False,
                        error=f"Invalid type for '{field_name}'. Expected {expected_type}, got {type(val)}",
                    )
        return ValidationResult(valid=True)
