"""Canonical M2 Registry and ToolSpec (03 §1-2)."""
from __future__ import annotations

from typing import Any, Callable, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .m2_errors import ErrorCatalog


class ToolSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tool_ref: str = Field(pattern=r"^[a-z][a-z0-9_-]*/[a-z][a-z0-9_-]*@[A-Za-z0-9._-]+$")
    capability_ref: str = Field(pattern=r"^[a-z][a-z0-9_-]*/[a-z][a-z0-9_-]*@[A-Za-z0-9._-]+$")
    owner: str = Field(min_length=1)
    args_schema: str = Field(min_length=1)
    result_schema: str = Field(min_length=1)
    risk: str
    implementation_mode: str
    side_effect: str
    timeout_ms: int = Field(gt=0)
    allowed_error_codes: tuple[str, ...] = ()
    callable: Callable[..., Any]

    _SCHEMAS = {
        "order.read.v1": ({"order_id", "phone_last4"}, {"order_id", "phone_last4"}),
        "aftersales.query.v1": ({"order_id", "phone_last4"}, {"order_id", "phone_last4"}),
        "aftersales.create.v1": ({"order_id", "phone_last4", "service_type", "reason"}, {"order_id", "phone_last4", "service_type", "reason"}),
        "logistics.query.v1": ({"carrier_code", "tracking_no"}, {"carrier_code", "tracking_no", "phone_last4"}),
        "handoff.v1": ({"summary", "reason"}, {"summary", "reason"}),
        "policy.query.v1": ({"query"}, {"query", "top_k"}),
        "product.read.v1": ({"sku"}, {"sku"}),
    }

    @field_validator("risk")
    @classmethod
    def valid_risk(cls, value: str) -> str:
        if value not in {"READ", "WRITE", "HIGH_RISK"}:
            raise ValueError("risk must be READ, WRITE or HIGH_RISK")
        return value

    @field_validator("implementation_mode")
    @classmethod
    def valid_mode(cls, value: str) -> str:
        if value not in {"LIVE", "SIMULATED", "FAULT"}:
            raise ValueError("mode must be LIVE, SIMULATED or FAULT")
        return value

    @field_validator("side_effect")
    @classmethod
    def valid_side_effect(cls, value: str) -> str:
        if value not in {"READ_ONLY", "WRITE"}:
            raise ValueError("side_effect must be READ_ONLY or WRITE")
        return value

    @field_validator("allowed_error_codes")
    @classmethod
    def valid_errors(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("allowed_error_codes must be unique")
        ErrorCatalog.validate_tool_errors(list(values))
        return values

    def validate_candidate_args(self, args: Mapping[str, Any]) -> None:
        if not isinstance(args, Mapping):
            raise ValueError("tool args must be an object")
        reserved = {"session_id", "user_id", "run_id", "task_id", "attempt_id", "auth_scope", "idempotency_key", "deadline", "cancellation", "confirm_token"}
        if reserved.intersection(args):
            raise ValueError("candidate args cannot override trusted invocation context")
        schema = self._SCHEMAS.get(self.args_schema)
        if schema is None:
            raise ValueError(f"unregistered args schema: {self.args_schema}")
        required, allowed = schema
        keys = set(args)
        if not required.issubset(keys) or not keys.issubset(allowed):
            raise ValueError(f"arguments do not match {self.args_schema}")


class Registry:
    def __init__(self, specs: Optional[list[ToolSpec]] = None):
        self._specs: dict[str, ToolSpec] = {}
        for spec in specs or []:
            self.register(spec)

    def register(self, spec: ToolSpec) -> None:
        if spec.tool_ref in self._specs:
            raise ValueError(f"duplicate tool_ref: {spec.tool_ref}")
        if any(existing.capability_ref == spec.capability_ref and existing.tool_ref == spec.tool_ref for existing in self._specs.values()):
            raise ValueError(f"duplicate registry reference: {spec.tool_ref}")
        self._specs[spec.tool_ref] = spec

    def get(self, tool_ref: str) -> ToolSpec:
        try:
            return self._specs[tool_ref]
        except KeyError as exc:
            raise KeyError(f"unregistered tool_ref: {tool_ref}") from exc

    def require_capability(self, tool_ref: str, capability_ref: str) -> ToolSpec:
        if not capability_ref:
            raise PermissionError("explicit capability authorization is required")
        spec = self.get(tool_ref)
        if spec.capability_ref != capability_ref:
            raise PermissionError("tool capability does not match registry")
        return spec

    def refs(self) -> tuple[str, ...]:
        return tuple(sorted(self._specs))

    def as_manifest(self) -> list[dict[str, Any]]:
        return [spec.model_dump(exclude={"callable"}) for spec in self._specs.values()]


def build_m2_registry() -> Registry:
    from .agent_tools import (
        create_aftersales_tool,
        get_order_info_tool,
        handoff_to_human_tool,
        query_aftersales_tool,
        query_logistics_snapshot_tool,
    )
    from .tool_registry import policy_rag_search_tool

    return Registry([
        ToolSpec(tool_ref="order/get_info@v1", capability_ref="order/read@v1", owner="order-agent", args_schema="order.read.v1", result_schema="order.result.v1", risk="READ", implementation_mode="SIMULATED", side_effect="READ_ONLY", timeout_ms=5000, allowed_error_codes=("AUTH_IDENTITY_MISMATCH", "ORDER_NOT_FOUND", "DATA_MISSING", "CONFIG_MISSING", "INFRA_UNAVAILABLE"), callable=get_order_info_tool),
        ToolSpec(tool_ref="aftersales/query@v1", capability_ref="aftersales/read@v1", owner="aftersales-agent", args_schema="aftersales.query.v1", result_schema="aftersales.result.v1", risk="READ", implementation_mode="SIMULATED", side_effect="READ_ONLY", timeout_ms=5000, allowed_error_codes=("AUTH_IDENTITY_MISMATCH", "ORDER_NOT_FOUND", "DATA_MISSING"), callable=query_aftersales_tool),
        ToolSpec(tool_ref="aftersales/create@v1", capability_ref="aftersales/write@v1", owner="aftersales-agent", args_schema="aftersales.create.v1", result_schema="aftersales.result.v1", risk="HIGH_RISK", implementation_mode="SIMULATED", side_effect="WRITE", timeout_ms=5000, allowed_error_codes=("AUTH_IDENTITY_MISMATCH", "ORDER_NOT_FOUND", "ACTIVE_CASE_EXISTS", "ELIGIBILITY_DENIED", "ELIGIBILITY_MANUAL", "CONTRACT_CONFIRM_REQUIRED", "IDEMPOTENCY_CONFLICT"), callable=create_aftersales_tool),
        ToolSpec(tool_ref="logistics/query@v1", capability_ref="logistics/read@v1", owner="logistics-agent", args_schema="logistics.query.v1", result_schema="logistics.result.v1", risk="READ", implementation_mode="SIMULATED", side_effect="READ_ONLY", timeout_ms=5000, allowed_error_codes=("DATA_MISSING", "DATA_STALE", "DATA_CONFLICT", "ELIGIBILITY_DENIED", "INFRA_UNAVAILABLE"), callable=query_logistics_snapshot_tool),
        ToolSpec(tool_ref="human/handoff@v1", capability_ref="human/handoff@v1", owner="supervisor", args_schema="handoff.v1", result_schema="handoff.result.v1", risk="HIGH_RISK", implementation_mode="SIMULATED", side_effect="WRITE", timeout_ms=5000, allowed_error_codes=("CONTRACT_SCHEMA_INVALID",), callable=handoff_to_human_tool),
        ToolSpec(tool_ref="policy/search@v1", capability_ref="policy/read@v1", owner="policy-agent", args_schema="policy.query.v1", result_schema="policy.result.v1", risk="READ", implementation_mode="SIMULATED", side_effect="READ_ONLY", timeout_ms=5000, allowed_error_codes=("DATA_MISSING", "POLICY_CONFLICT", "INFRA_UNAVAILABLE"), callable=policy_rag_search_tool),
    ])


__all__ = ["Registry", "ToolSpec", "build_m2_registry"]
