"""R5 action-scoped confirm contract (ADR-0002 §5-6).

The M2 ``ConfirmToken`` only expresses creation semantics.  R5 introduces an
independent, versioned action contract so a creation confirmation can never
authorize cancel or modify: every token is bound to exactly one
``action`` (create|cancel|modify), the target case, the expected case state
version and the exact payload being approved.  Tokens and previews are issued
by the trusted runtime only; the model never authors them.
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .domain.objects import HASH_PATTERN, sha256_json


R5_ACTION_TOKEN_SCHEMA_VERSION = "r5.action_token.v1"
R5_ACTIONS = ("create", "cancel", "modify")
R5_ALLOWED_MODIFY_STATUSES = frozenset({"REQUESTED", "UNDER_REVIEW", "APPROVED", "RETURN_PENDING"})
R5_CANCELLABLE_STATUSES = frozenset(
    {"REQUESTED", "UNDER_REVIEW", "APPROVED", "RETURN_PENDING", "RETURNED", "REFUND_PENDING", "EXCHANGE_PENDING", "HUMAN_REVIEW"}
)

ActionLiteral = Literal["create", "cancel", "modify"]


class R5ActionPreview(BaseModel):
    """Runtime-authored action preview shown to the user before confirmation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = R5_ACTION_TOKEN_SCHEMA_VERSION
    preview_id: str = Field(min_length=1)
    action: ActionLiteral
    order_id: str = Field(min_length=1)
    case_id: str | None = None
    service: str = Field(min_length=1)
    amount: Decimal
    fields_to_change: dict[str, str] = Field(default_factory=dict)
    rule_explanation: str = Field(min_length=1)
    expected_state_version: int | None = None
    topic_version: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    payload_hash: str = Field(default="", pattern=r"^(|[0-9a-f]{64})$")

    @field_validator("action")
    @classmethod
    def valid_action(cls, value: str) -> str:
        if value not in R5_ACTIONS:
            raise ValueError("unsupported action")
        return value

    def computed_payload_hash(self) -> str:
        return preview_payload_hash(
            action=self.action,
            order_id=self.order_id,
            case_id=self.case_id,
            service=self.service,
            amount=self.amount,
            fields_to_change=self.fields_to_change,
            expected_state_version=self.expected_state_version,
            topic_version=self.topic_version,
        )


def preview_payload_hash(
    *,
    action: str,
    order_id: str,
    case_id: str | None,
    service: str,
    amount: Decimal,
    fields_to_change: Mapping[str, str],
    expected_state_version: int | None,
    topic_version: str,
) -> str:
    return sha256_json(
        {
            "action": str(action),
            "order_id": str(order_id),
            "case_id": None if case_id is None else str(case_id),
            "service": str(service),
            "amount": str(amount),
            "fields_to_change": {str(k): str(v) for k, v in sorted(fields_to_change.items())},
            "expected_state_version": expected_state_version,
            "topic_version": str(topic_version),
        }
    )


class R5ActionToken(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    token_id: str = Field(min_length=1)
    token_hash: str = Field(pattern=HASH_PATTERN)
    schema_version: str = R5_ACTION_TOKEN_SCHEMA_VERSION
    action: ActionLiteral
    session_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    order_id: str = Field(min_length=1)
    case_id: str | None = None
    service: str = Field(min_length=1)
    amount: Decimal
    payload_hash: str = Field(pattern=HASH_PATTERN)
    expected_state_version: int | None = None
    topic_version: str = Field(min_length=1)
    issued_at: datetime
    expires_at: datetime
    status: str = "ISSUED"
    consumed_at: datetime | None = None
    revoked_at: datetime | None = None


def action_binding_hash(
    *,
    action: str,
    session_id: str,
    user_id: str,
    run_id: str,
    task_id: str,
    order_id: str,
    case_id: str | None,
    service: str,
    amount: Decimal,
    payload_hash: str,
    expected_state_version: int | None,
    topic_version: str,
) -> str:
    return sha256_json(
        {
            "schema_version": R5_ACTION_TOKEN_SCHEMA_VERSION,
            "action": str(action),
            "session_id": session_id,
            "user_id": user_id,
            "run_id": run_id,
            "task_id": task_id,
            "order_id": order_id,
            "case_id": None if case_id is None else str(case_id),
            "service": service,
            "amount": str(amount),
            "payload_hash": payload_hash,
            "expected_state_version": expected_state_version,
            "topic_version": topic_version,
        }
    )


__all__ = [
    "R5_ACTION_TOKEN_SCHEMA_VERSION",
    "R5_ACTIONS",
    "R5_ALLOWED_MODIFY_STATUSES",
    "R5_CANCELLABLE_STATUSES",
    "R5ActionPreview",
    "R5ActionToken",
    "action_binding_hash",
    "preview_payload_hash",
]
