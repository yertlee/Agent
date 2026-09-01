"""M2 canonical ErrorEnvelope and central error catalog (03 §4-5)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


@dataclass(frozen=True)
class ErrorDefinition:
    layer: str
    action: str
    retryable: bool


_DEFINITIONS = {
    "CONFIG_INVALID": ErrorDefinition("contract", "fail_run", False),
    "CONFIG_MISSING": ErrorDefinition("contract", "fail_run", False),
    "CONTRACT_SCHEMA_INVALID": ErrorDefinition("contract", "reject", False),
    "CONTRACT_INVALID_STATE": ErrorDefinition("contract", "reject", False),
    "CONTRACT_CONFIRM_REQUIRED": ErrorDefinition("contract", "ask_user", False),
    "CONTRACT_CONFIRM_EXPIRED": ErrorDefinition("contract", "ask_user", False),
    "AUTH_RESOURCE_FORBIDDEN": ErrorDefinition("auth", "reject", False),
    "AUTH_IDENTITY_MISMATCH": ErrorDefinition("auth", "reject", False),
    "AUTH_CAPABILITY_DENIED": ErrorDefinition("auth", "reject", False),
    "AUTH_SCOPE_MISMATCH": ErrorDefinition("auth", "reject", False),
    "ORDER_NOT_FOUND": ErrorDefinition("business", "ask_user", False),
    "POLICY_CONFLICT": ErrorDefinition("data_quality", "wait_human", False),
    "ELIGIBILITY_DENIED": ErrorDefinition("business", "stop", False),
    "ELIGIBILITY_MANUAL": ErrorDefinition("business", "wait_human", False),
    "ACTIVE_CASE_EXISTS": ErrorDefinition("business", "return_existing", False),
    "IDEMPOTENCY_CONFLICT": ErrorDefinition("business", "reject", False),
    "STATE_TRANSITION_REJECTED": ErrorDefinition("business", "reject", False),
    "DATA_MISSING": ErrorDefinition("data_quality", "ask_user", False),
    "DATA_STALE": ErrorDefinition("data_quality", "ask_user", False),
    "DATA_CONFLICT": ErrorDefinition("data_quality", "wait_human", False),
    "INFRA_TIMEOUT": ErrorDefinition("infra", "retry", True),
    "INFRA_UNAVAILABLE": ErrorDefinition("infra", "retry", True),
    "INFRA_RATE_LIMITED": ErrorDefinition("infra", "retry", True),
    "TOOL_CONTRACT_VIOLATION": ErrorDefinition("contract", "reject", False),
    "TOOL_EXECUTION_FAILED": ErrorDefinition("infra", "retry", True),
    "DEADLINE_EXCEEDED": ErrorDefinition("infra", "stop", False),
    "CANCELLED": ErrorDefinition("infra", "stop", False),
}

LEGACY_ERROR_MAP = {
    "PHONE_MISMATCH": "AUTH_IDENTITY_MISMATCH",
    "DB_NOT_FOUND": "CONFIG_MISSING",
    "DB_ERROR": "INFRA_UNAVAILABLE",
    "INVALID_PARAMS": "CONTRACT_SCHEMA_INVALID",
    "AFTERSALES_ALREADY_EXISTS": "ACTIVE_CASE_EXISTS",
    "AFTERSALES_NOT_ALLOWED": "ELIGIBILITY_DENIED",
    "AFTERSALES_NOT_FOUND": "DATA_MISSING",
    "NO_HITS": "DATA_MISSING",
}


class ErrorEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    layer: str
    action: str
    message_key: str = Field(min_length=1)
    retryable: bool
    retry_after_ms: Optional[int] = Field(default=None, ge=0)
    details: dict[str, Any] = Field(default_factory=dict)
    trace_id: str = Field(default_factory=lambda: f"trace_{uuid4().hex}", min_length=1)

    @field_validator("code")
    @classmethod
    def canonical_code(cls, value: str) -> str:
        if value not in _DEFINITIONS:
            raise ValueError("error code is not in canonical catalog")
        return value


class ErrorCatalog:
    """Fail-closed catalog used by Registry and executor boundaries."""

    @staticmethod
    def definition(code: str) -> ErrorDefinition:
        if code not in _DEFINITIONS:
            raise KeyError(f"unknown canonical error code: {code}")
        return _DEFINITIONS[code]

    @staticmethod
    def canonicalize(code: str) -> str:
        return LEGACY_ERROR_MAP.get(code, code)

    @classmethod
    def envelope(
        cls,
        code: str,
        *,
        details: Optional[Mapping[str, Any]] = None,
        message_key: Optional[str] = None,
        retry_after_ms: Optional[int] = None,
        trace_id: Optional[str] = None,
    ) -> ErrorEnvelope:
        canonical = cls.canonicalize(code)
        definition = cls.definition(canonical)
        return ErrorEnvelope(
            code=canonical,
            layer=definition.layer,
            action=definition.action,
            message_key=message_key or canonical.lower(),
            retryable=definition.retryable,
            retry_after_ms=retry_after_ms,
            details=dict(details or {}),
            trace_id=trace_id or f"trace_{uuid4().hex}",
        )

    @classmethod
    def validate_tool_errors(cls, codes: list[str]) -> None:
        for code in codes:
            try:
                cls.definition(cls.canonicalize(code))
            except KeyError as exc:
                raise ValueError(str(exc)) from exc


__all__ = ["ErrorCatalog", "ErrorDefinition", "ErrorEnvelope", "LEGACY_ERROR_MAP"]
