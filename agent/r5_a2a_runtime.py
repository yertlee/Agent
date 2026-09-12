"""R5 five-domain A2A runtime: thin extension of the frozen R4 runtime.

R5-A adds Product, AfterSales read and AfterSales eligibility capabilities to
the R4 message contract.  The R4 ledger, verifier core, dependency handling,
bounded retry, late barrier and terminal lifecycle are reused unchanged; this
module only registers three new read routes, three adapters and the semantic
checks specific to the new domains.  No write capability is dispatched over
A2A: guarded writes go through the M2 transaction path (R5-B).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .agents.ports import AgentPort
from .domain.objects import sha256_json
from .m3_registry import M3Registry
from .r4_a2a_runtime import (
    A2AContractError,
    A2AMessageVerifier,
    R4A2ARuntime,
    ROUTES,
    TypedAgentResult,
    _Route,
    build_r4_registry,
)
from .r5_aftersales_repository import (
    R5_AFTERSALES_SOURCE_VERSION,
    build_aftersales_read_callable,
    build_eligibility_callable,
)
from .r5_contracts import R5ToolSpec
from .r5_product_repository import R5_PRODUCT_SOURCE_VERSION, build_product_read_callable


R5_ROUTES: dict[str, _Route] = {
    "product/read@v1": _Route(
        "product/read@v1", "product-agent@v1", "product/get@v1", "product.result.v1", "product.read.v1"
    ),
    "aftersales/read@v1": _Route(
        "aftersales/read@v1", "aftersales-agent@v1", "aftersales/query@v1", "aftersales.result.v1", "aftersales.query.v1"
    ),
    "aftersales/eligibility@v1": _Route(
        "aftersales/eligibility@v1", "aftersales-agent@v1", "aftersales/eligibility@v1", "eligibility.result.v1", "aftersales.eligibility.v1"
    ),
}


def register_r5_routes() -> None:
    """Idempotently admit the R5 read routes into the shared ROUTES table."""
    for capability, route in R5_ROUTES.items():
        existing = ROUTES.get(capability)
        if existing is not None and existing != route:
            raise A2AContractError("A2A_R5_ROUTE_CONFLICT")
        ROUTES[capability] = route


class R5EligibilityPort(AgentPort):
    tool_ref = "aftersales/eligibility@v1"
    agent_ref = "aftersales-agent@v1"
    output_contract = "eligibility.result.v1"


class _ProductPort(AgentPort):
    tool_ref = "product/get@v1"
    agent_ref = "product-agent@v1"
    output_contract = "product.result.v1"


class _AfterSalesQueryPort(AgentPort):
    tool_ref = "aftersales/query@v1"
    agent_ref = "aftersales-agent@v1"
    output_contract = "aftersales.result.v1"


def _with_source_version(result: TypedAgentResult, source_version: str) -> TypedAgentResult:
    """Inject the protocol-level source_version into an ok specialist payload."""
    if not result.ok or not isinstance(result.payload, Mapping):
        return result
    payload = dict(result.payload)
    payload.setdefault("source_version", source_version)
    return result.model_copy(update={"payload": payload, "payload_hash": sha256_json(payload)})


class _ProductReadAdapter:
    agent_ref = "product-agent@v1"
    capability_ref = "product/read@v1"

    def __init__(self, registry: M3Registry):
        self.port = _ProductPort(registry=registry)

    def invoke(self, payload: Mapping[str, Any], *, context) -> TypedAgentResult:
        return _with_source_version(self.port.invoke(payload, context=context), R5_PRODUCT_SOURCE_VERSION)


class _AfterSalesReadAdapter:
    agent_ref = "aftersales-agent@v1"
    capability_ref = "aftersales/read@v1"

    def __init__(self, registry: M3Registry):
        self.port = _AfterSalesQueryPort(registry=registry)

    def invoke(self, payload: Mapping[str, Any], *, context) -> TypedAgentResult:
        return _with_source_version(self.port.invoke(payload, context=context), R5_AFTERSALES_SOURCE_VERSION)


class _EligibilityReadAdapter:
    agent_ref = "aftersales-agent@v1"
    capability_ref = "aftersales/eligibility@v1"

    def __init__(self, registry: M3Registry):
        self.port = R5EligibilityPort(registry=registry)

    def invoke(self, payload: Mapping[str, Any], *, context) -> TypedAgentResult:
        result = self.port.invoke(payload, context=context)
        if result.ok and isinstance(result.payload, Mapping):
            return _with_source_version(result, str(result.payload.get("policy_version") or R5_AFTERSALES_SOURCE_VERSION))
        return result


class R5MessageVerifier(A2AMessageVerifier):
    """R4 verifier plus product/after-sales semantic binding checks."""

    def verify_specialist_result(self, request, result: TypedAgentResult) -> _Route:
        route = super().verify_specialist_result(request, result)
        if not result.ok or not isinstance(result.payload, Mapping):
            return route
        payload = dict(result.payload)
        request_payload = request.payload or {}
        if route.capability_ref == "product/read@v1":
            if str(payload.get("sku") or "") != str(request_payload.get("sku") or ""):
                raise A2AContractError("A2A_SEMANTIC_WRONG")
            if str(payload.get("source") or "") != "simulator":
                raise A2AContractError("A2A_PRODUCT_SOURCE_UNAUTHORIZED")
        if route.capability_ref == "aftersales/read@v1":
            if str(payload.get("order_id") or "") != str(request_payload.get("order_id") or ""):
                raise A2AContractError("A2A_SEMANTIC_WRONG")
        if route.capability_ref == "aftersales/eligibility@v1":
            expected = f"{request_payload.get('order_id')}:{request_payload.get('service')}"
            if str(payload.get("service") or "") != str(request_payload.get("service") or ""):
                raise A2AContractError("A2A_SEMANTIC_WRONG")
            if str(payload.get("entity_id") or "") != expected:
                raise A2AContractError("A2A_SEMANTIC_WRONG")
            if str(payload.get("decision") or "") not in {"ALLOW", "DENY", "MANUAL"}:
                raise A2AContractError("A2A_ELIGIBILITY_DECISION_INVALID")
        return route


def build_r5_registry(
    *,
    db_path: str | Path,
    product_db_path: str | Path,
    aftersales_db_path: str | Path,
    logistics_db_path: str | Path | None = None,
    policy_reader=None,
    policy_retriever=None,
    policy_authority=None,
    policy_as_of=None,
    policy_mode: str = "bm25",
    implementation_mode: str = "SIMULATED",
) -> M3Registry:
    """R4 read registry plus R5 product/after-sales read capabilities.

    ``implementation_mode`` follows the ToolSpec contract; R5 demo data is
    local self-built data and is declared as such in each payload.
    """
    registry = build_r4_registry(
        db_path=db_path,
        logistics_db_path=logistics_db_path,
        policy_reader=policy_reader,
        policy_retriever=policy_retriever,
        policy_authority=policy_authority,
        policy_as_of=policy_as_of,
        policy_mode=policy_mode,
        implementation_mode=implementation_mode,
    )
    canonical = registry.canonical
    canonical.register(
        R5ToolSpec(
            tool_ref="product/get@v1",
            capability_ref="product/read@v1",
            owner="product-agent",
            args_schema="product.read.v1",
            result_schema="product.result.v1",
            risk="READ",
            implementation_mode=implementation_mode,
            side_effect="READ_ONLY",
            timeout_ms=5000,
            allowed_error_codes=("DATA_MISSING", "CONFIG_MISSING", "TOOL_CONTRACT_VIOLATION"),
            callable=build_product_read_callable(product_db_path),
        )
    )
    canonical.register(
        R5ToolSpec(
            tool_ref="aftersales/query@v1",
            capability_ref="aftersales/read@v1",
            owner="aftersales-agent",
            args_schema="aftersales.query.v1",
            result_schema="aftersales.result.v1",
            risk="READ",
            implementation_mode=implementation_mode,
            side_effect="READ_ONLY",
            timeout_ms=5000,
            allowed_error_codes=("AUTH_IDENTITY_MISMATCH", "ORDER_NOT_FOUND", "DATA_MISSING", "CONFIG_MISSING"),
            callable=build_aftersales_read_callable(aftersales_db_path),
        )
    )
    canonical.register(
        R5ToolSpec(
            tool_ref="aftersales/eligibility@v1",
            capability_ref="aftersales/eligibility@v1",
            owner="aftersales-agent",
            args_schema="aftersales.eligibility.v1",
            result_schema="eligibility.result.v1",
            risk="READ",
            implementation_mode=implementation_mode,
            side_effect="READ_ONLY",
            timeout_ms=5000,
            allowed_error_codes=("AUTH_IDENTITY_MISMATCH", "ORDER_NOT_FOUND", "DATA_MISSING", "CONFIG_MISSING"),
            callable=build_eligibility_callable(aftersales_db_path),
        )
    )
    return registry


class R5A2ARuntime(R4A2ARuntime):
    """R4 A2A runtime with the five read domains admitted."""

    def __init__(
        self,
        *,
        db_path: str | Path,
        product_db_path: str | Path,
        aftersales_db_path: str | Path,
        logistics_db_path: str | Path | None = None,
        ledger_path: str | Path = ":memory:",
        **kwargs: Any,
    ):
        register_r5_routes()
        registry = build_r5_registry(
            db_path=db_path,
            product_db_path=product_db_path,
            aftersales_db_path=aftersales_db_path,
            logistics_db_path=logistics_db_path,
            policy_reader=kwargs.get("policy_reader"),
            policy_retriever=kwargs.get("policy_retriever"),
            policy_authority=kwargs.get("policy_authority"),
            policy_as_of=kwargs.get("policy_as_of"),
            policy_mode=kwargs.get("policy_mode", "bm25"),
            implementation_mode=kwargs.get("implementation_mode", "SIMULATED"),
        )
        super().__init__(
            db_path=db_path,
            logistics_db_path=logistics_db_path,
            ledger_path=ledger_path,
            registry=registry,
            policy_reader=kwargs.get("policy_reader"),
            policy_retriever=kwargs.get("policy_retriever"),
            policy_authority=kwargs.get("policy_authority"),
            policy_as_of=kwargs.get("policy_as_of"),
            policy_mode=kwargs.get("policy_mode", "bm25"),
            scene_clock=kwargs.get("scene_clock"),
            timeout_seconds=kwargs.get("timeout_seconds", 5.0),
            failure_script=kwargs.get("failure_script"),
            retry_budget=kwargs.get("retry_budget", 2),
            disabled_capabilities=kwargs.get("disabled_capabilities", ()),
        )
        self.adapters.update(
            {
                "product/read@v1": _ProductReadAdapter(self.registry),
                "aftersales/read@v1": _AfterSalesReadAdapter(self.registry),
                "aftersales/eligibility@v1": _EligibilityReadAdapter(self.registry),
            }
        )
        self.verifier = R5MessageVerifier(self.registry, authority_snapshot=self.authority_snapshot)


__all__ = [
    "R5A2ARuntime",
    "R5EligibilityPort",
    "R5MessageVerifier",
    "R5_ROUTES",
    "build_r5_registry",
    "register_r5_routes",
]
