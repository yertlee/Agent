"""Versioned deterministic Policy Catalog for M2 (04 §1, 02 §2)."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field
from agent.m2_errors import ErrorCatalog


class PolicyConflictError(RuntimeError):
    def __init__(self, rule_ids: tuple[str, ...]):
        self.rule_ids = rule_ids
        self.envelope = ErrorCatalog.envelope("POLICY_CONFLICT", details={"rule_ids": list(rule_ids)})
        super().__init__(self.envelope.code)


class PolicyRule(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rule_id: str = Field(min_length=1)
    decision_logic: str = Field(min_length=1)
    source: str = Field(min_length=1)
    effective_from: datetime
    effective_to: Optional[datetime] = None
    scope: str = Field(min_length=1)
    priority: int = 0
    supersedes: Optional[str] = None
    version: str = Field(min_length=1)

    def active_at(self, when: datetime) -> bool:
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        start = self.effective_from if self.effective_from.tzinfo else self.effective_from.replace(tzinfo=timezone.utc)
        end = self.effective_to if self.effective_to is None or self.effective_to.tzinfo else self.effective_to.replace(tzinfo=timezone.utc)
        return when >= start and (end is None or when < end)


class PolicyCatalog:
    def __init__(self, rules: list[PolicyRule], *, version: str):
        if not version:
            raise ValueError("policy catalog version is required")
        ids = [r.rule_id for r in rules]
        if len(ids) != len(set(ids)):
            raise ValueError("policy rule IDs must be unique")
        self.version = version
        self.rules = tuple(rules)

    def resolve(self, *, service: str, when: Optional[datetime] = None) -> PolicyRule:
        when = when or datetime.now(timezone.utc)
        candidates = [r for r in self.rules if r.scope in {service, "*"} and r.active_at(when)]
        if not candidates:
            raise LookupError(f"no active policy rule for {service}")
        top_priority = max(r.priority for r in candidates)
        top = [r for r in candidates if r.priority == top_priority]
        if len(top) > 1 and len({r.decision_logic for r in top}) > 1:
            raise PolicyConflictError(tuple(sorted(r.rule_id for r in top)))
        candidates.sort(key=lambda r: (-r.priority, r.rule_id))
        return candidates[0]


__all__ = ["PolicyCatalog", "PolicyRule", "PolicyConflictError"]
