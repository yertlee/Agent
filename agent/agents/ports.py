"""Typed M3 Agent ports backed by the admitted registry."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from ..domain.objects import sha256_json
from ..m2_context import InvocationContext
from ..m2_errors import ErrorCatalog
from ..m2_executor import M2Executor
from ..m3_registry import M3Registry, build_m3_registry


class TypedAgentResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    contract: str = Field(min_length=1)
    ok: bool
    payload: Any = None
    payload_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    error_code: Optional[str] = None
    agent_ref: str = Field(min_length=1)
    tool_ref: str = Field(min_length=1)

    @classmethod
    def from_value(cls, *, contract: str, agent_ref: str, tool_ref: str, value: Any) -> "TypedAgentResult":
        if isinstance(value, Mapping) and value.get("success") is False:
            code = ErrorCatalog.canonicalize(str(value.get("code") or "TOOL_EXECUTION_FAILED"))
            ErrorCatalog.definition(code)
            return cls(contract=contract, ok=False, payload=None, payload_hash=sha256_json({"error_code": code}), error_code=code, agent_ref=agent_ref, tool_ref=tool_ref)
        payload = value.get("data") if isinstance(value, Mapping) and "data" in value else value
        return cls(contract=contract, ok=True, payload=payload, payload_hash=sha256_json(payload), agent_ref=agent_ref, tool_ref=tool_ref)


_SCHEMAS = {
    "order.read.v1": ({"order_id", "phone_last4"}, {"order_id", "phone_last4"}),
    "aftersales.query.v1": ({"order_id", "phone_last4"}, {"order_id", "phone_last4"}),
    "aftersales.create.v1": ({"order_id", "phone_last4", "service_type", "reason"}, {"order_id", "phone_last4", "service_type", "reason"}),
    "logistics.query.v1": ({"carrier_code", "tracking_no"}, {"carrier_code", "tracking_no", "phone_last4"}),
    "handoff.v1": ({"summary", "reason"}, {"summary", "reason"}),
    "policy.query.v1": ({"query"}, {"query", "top_k"}),
    "product.read.v1": ({"sku"}, {"sku"}),
}


class AgentPort:
    tool_ref: str
    agent_ref: str
    output_contract: str

    def __init__(self, *, registry: M3Registry | None = None):
        self.registry = registry or build_m3_registry()

    def _context(self) -> InvocationContext:
        owner = self.agent_ref.split("@", 1)[0]
        capability = self.registry.get(self.tool_ref).capability_ref
        return InvocationContext(session_id=f"m3_session_{uuid4().hex}", user_id="m3_fixture_user", run_id=f"m3_run_{uuid4().hex}", plan_revision_id=f"m3_plan_{uuid4().hex}", task_id=f"m3_task_{uuid4().hex}", attempt_id=f"m3_attempt_{uuid4().hex}", agent_ref=self.agent_ref, auth_scope=capability, idempotency_key=f"m3_idemp_{uuid4().hex}", deadline=datetime.now(timezone.utc) + timedelta(seconds=5), config_version="m3.v1", registry_version="m3.registry.v1", dataset_version="m3.dev44.v1", trace_id=f"m3_trace_{uuid4().hex}")

    def invoke(self, args: Mapping[str, Any], *, context: InvocationContext | None = None) -> TypedAgentResult:
        context = context or self._context()
        spec = self.registry.get(self.tool_ref)
        if context.agent_ref.split("@", 1)[0] != spec.owner.split("@", 1)[0] or context.auth_scope != spec.capability_ref:
            raise PermissionError("agent port context is not owned by registered manifest")
        # Ports are typed adapters only.  All validation, timeout, cancellation,
        # authorization and error normalization belongs to the canonical M2
        # executor; a port never calls a ToolSpec callable directly.
        result = M2Executor(self.registry.canonical).invoke(
            self.tool_ref, context, args, capability_ref=spec.capability_ref
        )
        if not result.ok:
            return TypedAgentResult(
                contract=self.output_contract,
                ok=False,
                payload=None,
                payload_hash=sha256_json({"error_code": result.error.code if result.error else "TOOL_EXECUTION_FAILED"}),
                error_code=result.error.code if result.error else "TOOL_EXECUTION_FAILED",
                agent_ref=self.agent_ref,
                tool_ref=self.tool_ref,
            )
        return TypedAgentResult.from_value(contract=self.output_contract, agent_ref=self.agent_ref, tool_ref=self.tool_ref, value=result.data)


class ProductAgent(AgentPort):
    tool_ref = "product/get@v1"
    agent_ref = "product-agent@v1"
    output_contract = "product.result.v1"


class OrderAgent(AgentPort):
    tool_ref = "order/get_info@v1"
    agent_ref = "order-agent@v1"
    output_contract = "order.result.v1"


class LogisticsAgent(AgentPort):
    tool_ref = "logistics/query@v1"
    agent_ref = "logistics-agent@v1"
    output_contract = "logistics.result.v1"


class PolicyAgent(AgentPort):
    tool_ref = "policy/search@v1"
    agent_ref = "policy-agent@v1"
    output_contract = "policy.result.v1"


class AfterSalesAgent(AgentPort):
    tool_ref = "aftersales/query@v1"
    agent_ref = "aftersales-agent@v1"
    output_contract = "aftersales.result.v1"

    def invoke_write(self, args: Mapping[str, Any], *, context: InvocationContext | None = None) -> TypedAgentResult:
        # Explicit tool selection keeps the port immutable and thread-safe.
        context = context or self._context().model_copy(update={"auth_scope": "aftersales/write@v1"})
        spec = self.registry.get("aftersales/create@v1")
        if context.agent_ref.split("@", 1)[0] != spec.owner.split("@", 1)[0] or context.auth_scope != spec.capability_ref:
            raise PermissionError("agent port context is not owned by registered manifest")
        result = M2Executor(self.registry.canonical).invoke("aftersales/create@v1", context, args, capability_ref=spec.capability_ref)
        if not result.ok:
            return TypedAgentResult(contract=self.output_contract, ok=False, payload=None, payload_hash=sha256_json({"error_code": result.error.code if result.error else "TOOL_EXECUTION_FAILED"}), error_code=result.error.code if result.error else "TOOL_EXECUTION_FAILED", agent_ref=self.agent_ref, tool_ref="aftersales/create@v1")
        return TypedAgentResult.from_value(contract=self.output_contract, agent_ref=self.agent_ref, tool_ref="aftersales/create@v1", value=result.data)


__all__ = ["AfterSalesAgent", "AgentPort", "LogisticsAgent", "OrderAgent", "PolicyAgent", "ProductAgent", "TypedAgentResult"]
