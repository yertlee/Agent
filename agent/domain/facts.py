"""Canonical DomainFact models for M2 (02 §1)."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .objects import HASH_PATTERN, sha256_json


class EntityType(str, Enum):
    ORDER = "order"
    LOGISTICS = "logistics"
    PRODUCT = "product"
    ELIGIBILITY = "eligibility"
    AFTERSALES = "aftersales"


class FactSource(str, Enum):
    SYSTEM = "system"
    SIMULATOR = "simulator"
    POLICY_CATALOG = "policy_catalog"
    HUMAN = "human"


class DataQuality(str, Enum):
    FRESH = "FRESH"
    STALE = "STALE"
    MISSING = "MISSING"
    CONFLICT = "CONFLICT"


class DomainFactBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    fact_id: str = Field(min_length=1)
    entity_id: str = Field(min_length=1)
    entity_type: EntityType
    source: FactSource
    observed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    snapshot_hash: str = Field(default="", pattern=HASH_PATTERN)
    version: str = Field(min_length=1)
    data_quality: DataQuality = DataQuality.FRESH

    @model_validator(mode="after")
    def compute_snapshot_hash(self) -> "DomainFactBase":
        payload = self.model_dump(mode="json", exclude={"snapshot_hash"})
        expected = sha256_json(payload)
        if self.snapshot_hash and self.snapshot_hash != expected:
            raise ValueError("snapshot_hash does not match fact payload")
        object.__setattr__(self, "snapshot_hash", expected)
        return self


class OrderFact(DomainFactBase):
    entity_type: EntityType = EntityType.ORDER
    user_id: str = Field(min_length=1)
    status: str = Field(min_length=1)
    amount: float = Field(ge=0)
    items: list[dict[str, Any]] = Field(default_factory=list)
    created_at: datetime


class LogisticsFact(DomainFactBase):
    entity_type: EntityType = EntityType.LOGISTICS
    order_id: str = Field(min_length=1)
    status: str = Field(min_length=1)
    events: list[dict[str, Any]] = Field(default_factory=list)
    delivered_at: Optional[datetime] = None


class EligibilityFact(DomainFactBase):
    entity_type: EntityType = EntityType.ELIGIBILITY
    service: str = Field(min_length=1)
    decision: str
    rule_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    policy_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def valid_decision(self) -> "EligibilityFact":
        if self.decision not in {"ALLOW", "DENY", "MANUAL"}:
            raise ValueError("decision must be ALLOW, DENY or MANUAL")
        return self


__all__ = ["DataQuality", "DomainFactBase", "EligibilityFact", "EntityType", "FactSource", "LogisticsFact", "OrderFact"]
