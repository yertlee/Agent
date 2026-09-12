"""Frozen R4 A2A message contracts.

R4 deliberately keeps the transport contract separate from the canonical M1
objects.  A message is an immutable request/result/error envelope; the
runtime owns trusted context and uses this module only for validation and
stable hashing.  No model-generated value is trusted merely because it is
present in a valid JSON object.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .domain.objects import sha256_json


A2A_SCHEMA_VERSION = "r4.a2a.message.v1"
A2A_RESULT_SCHEMA_VERSION = "r4.a2a.result.v1"
A2A_ERROR_SCHEMA_VERSION = "r4.a2a.error.v1"
HASH_PATTERN = r"^[0-9a-f]{64}$"


class A2AContractError(ValueError):
    """Deterministic, machine-readable contract failure."""

    def __init__(self, code: str, message: str = "") -> None:
        self.code = str(code)
        super().__init__(message or self.code)


class A2AErrorEnvelopeV1(BaseModel):
    """Safe error payload carried by an A2A error message."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_assignment=True)

    schema_version: str = A2A_ERROR_SCHEMA_VERSION
    code: str = Field(min_length=1, pattern=r"^[A-Z][A-Z0-9_.-]+$")
    message_key: str = Field(min_length=1)
    retryable: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


class A2AResultReferenceV1(BaseModel):
    """Reference used when a result is already persisted in canonical state."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_assignment=True)

    result_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    contract: str = Field(min_length=1)
    payload_hash: str = Field(pattern=HASH_PATTERN)
    source_version: str = Field(min_length=1)


class A2AMessageEnvelopeV1(BaseModel):
    """Immutable versioned A2A request/result/error envelope.

    Exactly one of ``payload``, ``result_ref`` and ``error`` is present.  The
    envelope hash covers that selected value.  Trusted runtime fields are
    still verified by :class:`agent.r4_a2a_runtime.A2AMessageVerifier`; this
    contract only establishes shape and deterministic hashing.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, validate_assignment=True)

    message_id: str = Field(min_length=1)
    correlation_id: str = Field(min_length=1)
    parent_message_id: Optional[str] = None
    dependency_message_ids: tuple[str, ...] = ()
    run_id: str = Field(min_length=1)
    plan_revision_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    attempt_id: str = Field(min_length=1)
    trace_id: str = Field(min_length=1)
    sender_ref: str = Field(min_length=1)
    receiver_ref: str = Field(min_length=1)
    capability_ref: str = Field(min_length=1)
    schema_version: str = A2A_SCHEMA_VERSION
    message_kind: str
    created_at: datetime
    deadline: datetime
    idempotency_key: str = Field(min_length=1)
    payload_hash: str = Field(pattern=HASH_PATTERN)
    payload: Optional[dict[str, Any]] = None
    result_ref: Optional[A2AResultReferenceV1] = None
    error: Optional[A2AErrorEnvelopeV1] = None

    @field_validator("message_kind")
    @classmethod
    def normalize_message_kind(cls, value: str) -> str:
        value = str(value).upper()
        if value not in {"REQUEST", "RESULT", "ERROR"}:
            raise ValueError("message_kind must be REQUEST, RESULT or ERROR")
        return value

    @field_validator("created_at", "deadline")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("A2A timestamps must be timezone-aware")
        return value.astimezone(timezone.utc)

    @field_validator("dependency_message_ids")
    @classmethod
    def unique_dependencies(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not str(item) for item in value) or len(value) != len(set(value)):
            raise ValueError("dependency_message_ids must be unique and non-empty")
        # Dependencies are a set for execution purposes.  Store their
        # canonical order so a transport replay cannot acquire a second
        # identity merely by permuting the same dependency list.
        return tuple(sorted(str(item) for item in value))

    @model_validator(mode="after")
    def validate_payload_choice_and_hash(self) -> "A2AMessageEnvelopeV1":
        choices = [self.payload is not None, self.result_ref is not None, self.error is not None]
        if sum(choices) != 1:
            raise ValueError("exactly one of payload, result_ref or error is required")
        if self.message_kind == "REQUEST" and self.payload is None:
            raise ValueError("REQUEST must contain typed payload")
        if self.message_kind == "RESULT" and self.payload is None and self.result_ref is None:
            raise ValueError("RESULT must contain payload or result_ref")
        if self.message_kind == "ERROR" and self.error is None:
            raise ValueError("ERROR must contain ErrorEnvelope")
        selected: Any
        if self.payload is not None:
            selected = self.payload
        elif self.result_ref is not None:
            selected = self.result_ref
        else:
            selected = self.error
        expected = sha256_json(selected)
        if self.payload_hash != expected:
            raise ValueError("payload_hash does not match selected message payload")
        if self.message_kind == "ERROR" and self.payload is not None:
            raise ValueError("ERROR cannot carry typed payload")
        if self.message_kind != "ERROR" and self.error is not None:
            raise ValueError("only ERROR can carry ErrorEnvelope")
        return self

    @property
    def selected_payload(self) -> Any:
        if self.payload is not None:
            return self.payload
        if self.result_ref is not None:
            return self.result_ref
        return self.error

    @classmethod
    def request(
        cls,
        *,
        message_id: str,
        correlation_id: str,
        run_id: str,
        plan_revision_id: str,
        task_id: str,
        attempt_id: str,
        trace_id: str,
        sender_ref: str,
        receiver_ref: str,
        capability_ref: str,
        created_at: datetime,
        deadline: datetime,
        idempotency_key: str,
        payload: Mapping[str, Any],
        parent_message_id: str | None = None,
        dependency_message_ids: tuple[str, ...] = (),
    ) -> "A2AMessageEnvelopeV1":
        return cls(
            message_id=message_id,
            correlation_id=correlation_id,
            parent_message_id=parent_message_id,
            dependency_message_ids=dependency_message_ids,
            run_id=run_id,
            plan_revision_id=plan_revision_id,
            task_id=task_id,
            attempt_id=attempt_id,
            trace_id=trace_id,
            sender_ref=sender_ref,
            receiver_ref=receiver_ref,
            capability_ref=capability_ref,
            message_kind="REQUEST",
            created_at=created_at,
            deadline=deadline,
            idempotency_key=idempotency_key,
            payload_hash=sha256_json(dict(payload)),
            payload=dict(payload),
        )

    @classmethod
    def result(
        cls,
        *,
        request: "A2AMessageEnvelopeV1",
        sender_ref: str,
        receiver_ref: str,
        attempt_id: str,
        payload: Mapping[str, Any] | None = None,
        result_ref: A2AResultReferenceV1 | None = None,
    ) -> "A2AMessageEnvelopeV1":
        selected = dict(payload) if payload is not None else result_ref
        if selected is None:
            raise ValueError("result payload or result_ref is required")
        return cls(
            message_id=f"{request.message_id}:result:{attempt_id}",
            correlation_id=request.correlation_id,
            parent_message_id=request.message_id,
            dependency_message_ids=request.dependency_message_ids,
            run_id=request.run_id,
            plan_revision_id=request.plan_revision_id,
            task_id=request.task_id,
            attempt_id=attempt_id,
            trace_id=request.trace_id,
            sender_ref=sender_ref,
            receiver_ref=receiver_ref,
            capability_ref=request.capability_ref,
            message_kind="RESULT",
            created_at=request.created_at,
            deadline=request.deadline,
            idempotency_key=f"{request.idempotency_key}:result",
            payload_hash=sha256_json(selected),
            payload=dict(payload) if payload is not None else None,
            result_ref=result_ref,
        )

    @classmethod
    def error_message(
        cls,
        *,
        request: "A2AMessageEnvelopeV1",
        sender_ref: str,
        receiver_ref: str,
        attempt_id: str,
        error: A2AErrorEnvelopeV1,
    ) -> "A2AMessageEnvelopeV1":
        return cls(
            message_id=f"{request.message_id}:error:{attempt_id}",
            correlation_id=request.correlation_id,
            parent_message_id=request.message_id,
            dependency_message_ids=request.dependency_message_ids,
            run_id=request.run_id,
            plan_revision_id=request.plan_revision_id,
            task_id=request.task_id,
            attempt_id=attempt_id,
            trace_id=request.trace_id,
            sender_ref=sender_ref,
            receiver_ref=receiver_ref,
            capability_ref=request.capability_ref,
            message_kind="ERROR",
            created_at=request.created_at,
            deadline=request.deadline,
            idempotency_key=f"{request.idempotency_key}:error",
            payload_hash=sha256_json(error),
            error=error,
        )


def request_fingerprint(envelope: A2AMessageEnvelopeV1) -> str:
    """Stable idempotency fingerprint excluding per-attempt identifiers."""

    return sha256_json(
        {
            "message_kind": envelope.message_kind,
            "schema_version": envelope.schema_version,
            "run_id": envelope.run_id,
            "plan_revision_id": envelope.plan_revision_id,
            "task_id": envelope.task_id,
            "correlation_id": envelope.correlation_id,
            "parent_message_id": envelope.parent_message_id,
            "dependency_message_ids": sorted(str(item) for item in envelope.dependency_message_ids),
            "sender_ref": envelope.sender_ref,
            "receiver_ref": envelope.receiver_ref,
            "capability_ref": envelope.capability_ref,
            "trace_id": envelope.trace_id,
            "deadline": envelope.deadline,
            "payload_hash": envelope.payload_hash,
        }
    )


__all__ = [
    "A2AContractError",
    "A2AErrorEnvelopeV1",
    "A2AResultReferenceV1",
    "A2AMessageEnvelopeV1",
    "A2A_SCHEMA_VERSION",
    "A2A_RESULT_SCHEMA_VERSION",
    "A2A_ERROR_SCHEMA_VERSION",
    "request_fingerprint",
]
