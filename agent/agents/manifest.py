"""Five M3 Agent manifests; each is a strict subset of Registry authority."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..m3_registry import M3Registry, build_m3_registry


class AgentManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    agent_ref: str = Field(pattern=r"^[a-z][a-z0-9-]*@v[0-9]+$")
    owner: str = Field(min_length=1)
    role: str = Field(min_length=1)
    allowed_capabilities: tuple[str, ...] = Field(min_length=1)
    input_types: tuple[str, ...] = Field(min_length=1)
    output_types: tuple[str, ...] = Field(min_length=1)
    shared_state_write: bool = False


def build_m3_manifests() -> tuple[AgentManifest, ...]:
    return (
        AgentManifest(agent_ref="product-agent@v1", owner="product-agent", role="product fact and catalog read", allowed_capabilities=("product/read@v1",), input_types=("ProductQuery",), output_types=("ProductFact", "Result")),
        AgentManifest(agent_ref="order-agent@v1", owner="order-agent", role="order ownership and order fact read", allowed_capabilities=("order/read@v1",), input_types=("OrderQuery",), output_types=("OrderFact", "Result")),
        AgentManifest(agent_ref="logistics-agent@v1", owner="logistics-agent", role="deterministic logistics snapshot read", allowed_capabilities=("logistics/read@v1",), input_types=("LogisticsQuery",), output_types=("LogisticsFact", "Result")),
        AgentManifest(agent_ref="policy-agent@v1", owner="policy-agent", role="policy evidence and catalog read", allowed_capabilities=("policy/read@v1",), input_types=("PolicyQuery",), output_types=("PolicyFact", "Result")),
        AgentManifest(agent_ref="aftersales-agent@v1", owner="aftersales-agent", role="aftersales query and guarded request orchestration", allowed_capabilities=("aftersales/read@v1", "aftersales/write@v1"), input_types=("AfterSalesQuery", "AfterSalesRequest", "EligibilityFact", "ConfirmTokenBinding"), output_types=("AfterSalesCase", "EligibilityFact", "Result", "ErrorEnvelope")),
    )


def validate_manifests(manifests: tuple[AgentManifest, ...] | list[AgentManifest], registry: M3Registry | None = None) -> None:
    registry = registry or build_m3_registry()
    refs = {spec.capability_ref for spec in (registry.get(ref) for ref in registry.refs())}
    owners = {spec.owner for spec in (registry.get(ref) for ref in registry.refs())}
    if len({m.agent_ref for m in manifests}) != len(manifests):
        raise ValueError("agent_ref must be unique")
    for manifest in manifests:
        if manifest.shared_state_write:
            raise ValueError("agents cannot write shared state")
        if not set(manifest.allowed_capabilities).issubset(refs):
            raise ValueError(f"manifest capability is not admitted: {manifest.agent_ref}")
        for capability in manifest.allowed_capabilities:
            spec = next(registry.get(ref) for ref in registry.refs() if registry.get(ref).capability_ref == capability)
            if spec.owner != manifest.owner:
                raise ValueError(f"manifest owner mismatch: {manifest.agent_ref}")
    if not set(m.owner for m in manifests).issubset(owners | {"product-agent"}):
        raise ValueError("manifest owner is not represented by Registry")


__all__ = ["AgentManifest", "build_m3_manifests", "validate_manifests"]
