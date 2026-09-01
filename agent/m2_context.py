"""Trusted, immutable InvocationContext for M2 executor calls (01 §5, 03 §3)."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class InvocationContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    session_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    plan_revision_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    attempt_id: str = Field(min_length=1)
    agent_ref: str = Field(min_length=1)
    auth_scope: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)
    deadline: datetime
    cancellation: bool = False
    config_version: str = Field(min_length=1)
    registry_version: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    trace_id: str = Field(min_length=1)

    def is_expired(self, now: Optional[datetime] = None) -> bool:
        now = now or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        deadline = self.deadline if self.deadline.tzinfo else self.deadline.replace(tzinfo=timezone.utc)
        return now >= deadline


__all__ = ["InvocationContext"]
