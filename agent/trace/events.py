from __future__ import annotations

import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import Field, model_validator

from agent.domain.objects import HASH_PATTERN, StrictModel, canonical_json, sha256_json

EVENT_TYPES = {
    "RUN_CREATED", "PLAN_CREATED", "PLAN_VALIDATED", "TASK_STATE_CHANGED", "ATTEMPT_STARTED",
    "MODEL_CALLED", "MODEL_RETURNED", "TOOL_CALLED", "TOOL_RETURNED", "RESULT_WRITTEN", "ERROR",
    "TOKEN_EVENT", "REVIEW_EVENT", "CHECKPOINT", "STATE_DELTA", "RUN_FROZEN",
}


class TraceActor(str, Enum):
    RUNTIME = "runtime"
    AGENT = "agent"
    TOOL = "tool"
    REVIEWER = "reviewer"
    SIMULATOR = "simulator"
    RECORDER = "recorder"


class TraceEvent(StrictModel):
    trace_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    plan_revision_id: Optional[str] = None
    task_id: Optional[str] = None
    attempt_id: Optional[str] = None
    parent_event_id: Optional[str] = None
    seq_no: int = Field(gt=0)
    event_type: str
    occurred_at: datetime
    scene_clock: Optional[datetime] = None
    actor: TraceActor
    schema_version: str = "m1.trace.v1"
    payload: dict[str, Any] = Field(default_factory=dict)
    payload_hash: str = Field(default="", pattern=HASH_PATTERN)

    @model_validator(mode="after")
    def validate_payload_hash(self) -> "TraceEvent":
        if self.event_type not in EVENT_TYPES:
            raise ValueError(f"unsupported event_type: {self.event_type}")
        expected = sha256_json(self.payload)
        if self.payload_hash and self.payload_hash != expected:
            raise ValueError("payload_hash does not match payload")
        object.__setattr__(self, "payload_hash", expected)
        return self

    def envelope_hash(self) -> str:
        return sha256_json(self)


def build_event(*, run_id: str, session_id: str, event_type: str, seq_no: int, payload: Optional[dict[str, Any]] = None, plan_revision_id: Optional[str] = None, task_id: Optional[str] = None, attempt_id: Optional[str] = None, parent_event_id: Optional[str] = None, actor: str = "runtime", scene_clock: Optional[datetime] = None, trace_id: Optional[str] = None) -> TraceEvent:
    return TraceEvent(trace_id=trace_id or str(uuid4()), run_id=run_id, session_id=session_id, plan_revision_id=plan_revision_id, task_id=task_id, attempt_id=attempt_id, parent_event_id=parent_event_id, seq_no=seq_no, event_type=event_type, occurred_at=datetime.now(timezone.utc), scene_clock=scene_clock, actor=actor, payload=payload or {})


_SENSITIVE_KEYS = re.compile(r"(?:token|secret|password|phone|address|payment|raw_payload)", re.I)


def sensitive_surface_scan(value: Any) -> list[str]:
    """Return key paths containing prohibited raw sensitive fields."""
    findings: list[str] = []
    def walk(item: Any, path: str = "") -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                current = f"{path}.{key}" if path else str(key)
                key_text = str(key).lower()
                # Stable hashes/references are explicitly allowed by 05; raw
                # token, phone, address and payment values are not.
                safe_hash_or_ref = key_text.endswith(("_hash", "_ref")) or key_text in {"token_id_hash", "payload_hash", "snapshot_hash"}
                if _SENSITIVE_KEYS.search(str(key)) and not safe_hash_or_ref:
                    findings.append(current)
                walk(child, current)
        elif isinstance(item, list):
            for idx, child in enumerate(item): walk(child, f"{path}[{idx}]")
    walk(value)
    return findings
