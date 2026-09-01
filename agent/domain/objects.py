"""Versioned, immutable M1 execution objects.

The legacy LangGraph models remain adapters only.  These Pydantic contracts are
the persistence boundary for ``Run -> PlanRevision -> Task -> TaskAttempt ->
Result`` and intentionally reject unknown fields.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "m1.v1"
HASH_PATTERN = r"^[0-9a-fA-F]{64}$"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def canonical_json(value: Any) -> str:
    """Canonical UTF-8 JSON used by all M1 hashes."""
    def normalize(item: Any) -> Any:
        if isinstance(item, BaseModel):
            return normalize(item.model_dump(mode="python", exclude_none=False))
        if isinstance(item, dict):
            return {str(k): normalize(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [normalize(v) for v in item]
        if isinstance(item, Enum):
            return item.value
        if isinstance(item, datetime):
            return item.isoformat().replace("+00:00", "Z")
        return item
    value = normalize(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


# Stable names for callers that distinguish JSON canonicalization from hashing.
canonical_hash = sha256_json
hash_payload = sha256_json


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, frozen=True)


class PlanStatus(str, Enum):
    DRAFT = "DRAFT"
    VALIDATED = "VALIDATED"
    ACTIVE = "ACTIVE"
    SUPERSEDED = "SUPERSEDED"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"


class TaskStatus(str, Enum):
    CREATED = "CREATED"
    READY = "READY"
    RUNNING = "RUNNING"
    RETRY_WAIT = "RETRY_WAIT"
    WAITING_USER = "WAITING_USER"
    WAITING_HUMAN = "WAITING_HUMAN"
    BLOCKED = "BLOCKED"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class AttemptStatus(str, Enum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    RETRY_WAIT = "RETRY_WAIT"
    WAITING_USER = "WAITING_USER"
    WAITING_HUMAN = "WAITING_HUMAN"
    BLOCKED = "BLOCKED"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class ResultStatus(str, Enum):
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    CANCELLED = "CANCELLED"


class RunStatus(str, Enum):
    CREATED = "CREATED"
    READY = "READY"
    RUNNING = "RUNNING"
    RETRY_WAIT = "RETRY_WAIT"
    WAITING_USER = "WAITING_USER"
    WAITING_HUMAN = "WAITING_HUMAN"
    BLOCKED = "BLOCKED"
    PARTIAL = "PARTIAL"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class InputBinding(StrictModel):
    name: str = Field(min_length=1)
    kind: str
    source_task_id: Optional[str] = None
    path: str = ""
    token_purpose: Optional[str] = None
    required: bool = True

    @model_validator(mode="after")
    def validate_binding(self) -> "InputBinding":
        if self.kind not in {"result", "invocation_context", "confirm_token"}:
            raise ValueError("binding kind must be result, invocation_context or confirm_token")
        if self.kind == "result" and (not self.source_task_id or not self.path):
            raise ValueError("result binding requires source_task_id and path")
        if self.kind != "result" and self.source_task_id is not None:
            raise ValueError("source_task_id is only valid for result bindings")
        if self.kind == "confirm_token" and not self.token_purpose:
            raise ValueError("confirm_token binding requires token_purpose")
        if self.kind == "invocation_context" and not self.path:
            raise ValueError("invocation_context binding requires path")
        return self


class Task(StrictModel):
    task_id: str = Field(min_length=1)
    plan_revision_id: str = Field(min_length=1)
    agent_ref: str = Field(min_length=1)
    capability_refs: List[str] = Field(min_length=1)
    depends_on: List[str] = Field(default_factory=list)
    input_bindings: List[InputBinding] = Field(default_factory=list)
    output_contract: str = Field(min_length=1)
    failure_strategy: str = Field(min_length=1)
    side_effect: str
    timeout_ms: int = Field(gt=0)
    deadline: Optional[datetime] = None

    @field_validator("side_effect")
    @classmethod
    def valid_side_effect(cls, value: str) -> str:
        if value not in {"READ_ONLY", "WRITE"}:
            raise ValueError("side_effect must be READ_ONLY or WRITE")
        return value

    @field_validator("capability_refs")
    @classmethod
    def unique_capabilities(cls, value: List[str]) -> List[str]:
        if any(not ref for ref in value) or len(value) != len(set(value)):
            raise ValueError("capability_refs must be non-empty and unique")
        return value

    @field_validator("depends_on")
    @classmethod
    def unique_dependencies(cls, value: List[str]) -> List[str]:
        if len(value) != len(set(value)):
            raise ValueError("depends_on must be unique")
        return value

    @field_validator("failure_strategy")
    @classmethod
    def valid_failure_strategy(cls, value: str) -> str:
        allowed = {"RETRY", "REPLAN_LOCAL", "WAIT_USER", "WAIT_HUMAN", "BLOCK", "CONTINUE_PARTIAL", "FAIL_RUN"}
        if value not in allowed:
            raise ValueError("invalid failure_strategy")
        return value


class PlanRevision(StrictModel):
    plan_revision_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    schema_version: str = SCHEMA_VERSION
    created_by: str = Field(min_length=1)
    revision_reason: str = Field(min_length=1)
    supersedes_plan_revision_id: Optional[str] = None
    version: int = Field(gt=0)
    status: PlanStatus = PlanStatus.DRAFT
    tasks: List[Task] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def task_parent_ids_match(self) -> "PlanRevision":
        if any(t.plan_revision_id != self.plan_revision_id for t in self.tasks):
            raise ValueError("task plan_revision_id must match parent revision")
        return self

    @field_validator("created_by")
    @classmethod
    def valid_creator(cls, value: str) -> str:
        if value not in {"supervisor", "runtime"}:
            raise ValueError("created_by must be supervisor or runtime")
        return value

    @field_validator("revision_reason")
    @classmethod
    def valid_revision_reason(cls, value: str) -> str:
        if value not in {"initial", "local_replan", "topic_switch", "recovery"}:
            raise ValueError("invalid revision_reason")
        return value


class Usage(StrictModel):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    logical_calls: int = Field(default=0, ge=0)
    physical_attempts: int = Field(default=0, ge=0)
    internal_retries: int = Field(default=0, ge=0)
    agent_retries: int = Field(default=0, ge=0)
    latency_ms: int = Field(default=0, ge=0)


class Run(StrictModel):
    run_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    schema_version: str = SCHEMA_VERSION
    plan_revision_id: Optional[str] = None
    initial_world_hash: str = Field(pattern=HASH_PATTERN)
    status: RunStatus = RunStatus.CREATED
    state_version: int = Field(default=0, ge=0)
    next_seq_no: int = Field(default=1, ge=1)
    freeze_status: str = "OPEN"
    shared_state: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class TaskAttempt(StrictModel):
    attempt_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    plan_revision_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    agent_ref: str = Field(min_length=1)
    attempt_no: int = Field(gt=0)
    status: AttemptStatus = AttemptStatus.CREATED
    input_hash: str = Field(pattern=HASH_PATTERN)
    deadline: Optional[datetime] = None
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None


class Result(StrictModel):
    schema_version: str = SCHEMA_VERSION
    result_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    plan_revision_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    attempt_id: str = Field(min_length=1)
    status: ResultStatus
    output_contract: str = Field(min_length=1)
    payload: Optional[Dict[str, Any]] = None
    payload_hash: str = Field(default="", pattern=HASH_PATTERN)
    business_code: Optional[str] = None
    error_ref: Optional[str] = None
    evidence_refs: List[str] = Field(default_factory=list)
    claim_refs: List[str] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def compute_or_check_payload_hash(self) -> "Result":
        expected = sha256_json(self.payload)
        if self.payload_hash and self.payload_hash != expected:
            raise ValueError("payload_hash does not match payload")
        object.__setattr__(self, "payload_hash", expected)
        return self


def model_hash(value: StrictModel) -> str:
    return sha256_json(value)
