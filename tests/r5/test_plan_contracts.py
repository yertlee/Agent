"""R5-C plan contract, validator and Router/Planner boundary tests."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent.r5_plan_contracts import (
    R5BindingV1,
    R5PlanEdgeV1,
    R5PlanNodeV1,
    R5PlanV1,
    R5_ALL_CAPABILITIES,
)
from agent.r5_router_planner import (
    DeterministicRouterPlannerAdapter,
    R5RouterDecisionV1,
    R5RouterPlannerBoundary,
)


def _node(node_id: str, capability: str, *, bindings=None, args=None) -> R5PlanNodeV1:
    return R5PlanNodeV1(node_id=node_id, capability_ref=capability, args=args or {}, bindings=bindings or {})


def test_two_different_capability_combinations_are_both_legal() -> None:
    order_only = R5PlanV1(business_goal="查订单", nodes=(_node("order", "order/read@v1"),))
    order_policy = R5PlanV1(
        business_goal="查订单并解释政策",
        nodes=(_node("order", "order/read@v1"), _node("policy", "policy/read@v1")),
    )
    logistics_only = R5PlanV1(business_goal="查物流", nodes=(_node("logistics", "logistics/read@v1"),))
    assert order_only.derived_capabilities() == {"order/read@v1"}
    assert order_policy.derived_capabilities() == {"order/read@v1", "policy/read@v1"}
    assert logistics_only.derived_capabilities() == {"logistics/read@v1"}


def test_schema_has_no_duplicate_topology_or_capability_fields() -> None:
    # R4's triple-duplication fields must not exist; extra fields are rejected.
    with pytest.raises(ValidationError):
        R5PlanV1.model_validate({"business_goal": "g", "topology": "order_only", "requested_capabilities": ["order/read@v1"]})


def test_derived_views_match_structure() -> None:
    plan = R5PlanV1(
        business_goal="先订单再物流",
        nodes=(_node("order", "order/read@v1"), _node("logistics", "logistics/read@v1", bindings={"order_id": R5BindingV1(kind="result", source_node_id="order", path="order_id")})),
        edges=(R5PlanEdgeV1(upstream_node_id="order", downstream_node_id="logistics"),),
    )
    assert plan.derived_capabilities() == {"order/read@v1", "logistics/read@v1"}
    assert plan.derived_dependency_order() == ("order", "logistics")
    assert plan.ancestors()["logistics"] == {"order"}


def test_result_binding_must_reference_upstream_node() -> None:
    with pytest.raises(ValidationError):
        R5PlanV1(
            business_goal="bad",
            nodes=(
                _node("logistics", "logistics/read@v1", bindings={"order_id": R5BindingV1(kind="result", source_node_id="order", path="order_id")}),
                _node("order", "order/read@v1"),
            ),
            edges=(),
        )


def test_context_binding_requires_registered_key() -> None:
    with pytest.raises(ValidationError):
        R5BindingV1(kind="context", context_key="secret_token")
    ok = R5BindingV1(kind="context", context_key="phone_last4")
    assert ok.required is True


def test_write_requires_eligibility_ancestor() -> None:
    with pytest.raises(ValidationError):
        R5PlanV1(business_goal="直接写", nodes=(_node("write", "aftersales/write@v1"),))
    ok = R5PlanV1(
        business_goal="先资格后写",
        nodes=(_node("eligibility", "aftersales/eligibility@v1"), _node("write", "aftersales/write@v1")),
        edges=(R5PlanEdgeV1(upstream_node_id="eligibility", downstream_node_id="write"),),
    )
    assert "aftersales/write@v1" in ok.derived_capabilities()


def test_cyclic_plan_rejected() -> None:
    with pytest.raises(ValidationError):
        R5PlanV1(
            business_goal="cycle",
            nodes=(_node("a", "order/read@v1"), _node("b", "logistics/read@v1")),
            edges=(R5PlanEdgeV1(upstream_node_id="a", downstream_node_id="b"), R5PlanEdgeV1(upstream_node_id="b", downstream_node_id="a")),
        )


def test_unknown_capability_rejected() -> None:
    with pytest.raises(ValidationError):
        _node("x", "sql/execute@v1")


def test_duplicate_node_and_edge_rejected() -> None:
    with pytest.raises(ValidationError):
        R5PlanV1(business_goal="dup", nodes=(_node("order", "order/read@v1"), _node("order", "order/read@v1")))
    with pytest.raises(ValidationError):
        R5PlanV1(
            business_goal="dup-edge",
            nodes=(_node("a", "order/read@v1"), _node("b", "logistics/read@v1")),
            edges=(R5PlanEdgeV1(upstream_node_id="a", downstream_node_id="b"), R5PlanEdgeV1(upstream_node_id="a", downstream_node_id="b")),
        )


def test_clarification_plan_cannot_contain_writes() -> None:
    with pytest.raises(ValidationError):
        R5PlanV1(
            business_goal="clarify",
            nodes=(_node("write", "aftersales/write@v1"),),
            needs_clarification=True,
            clarification_reason="missing order id",
        )


def test_router_decision_validation() -> None:
    ok = R5RouterDecisionV1(intents=("ORDER_QUERY",), entities={"order_id": "20260101001"}, business_goal="查订单")
    assert ok.needs_clarification is False
    with pytest.raises(ValidationError):
        R5RouterDecisionV1(intents=("NOT_AN_INTENT",), business_goal="x")
    with pytest.raises(ValidationError):
        R5RouterDecisionV1(intents=("ORDER_QUERY",), entities={"raw_sql": "select 1"}, business_goal="x")
    with pytest.raises(ValidationError):
        R5RouterDecisionV1(intents=("ORDER_QUERY",), needs_clarification=True, business_goal="x")


class _SequenceProvider:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = 0

    def __call__(self, prompt, router_schema, plan_schema):
        self.calls += 1
        out = self.outputs[min(self.calls - 1, len(self.outputs) - 1)]
        if isinstance(out, Exception):
            raise out
        return out


def _valid_output() -> dict:
    return {
        "router": {"intents": ("ORDER_QUERY",), "entities": {"order_id": "20260101001"}, "business_goal": "查订单"},
        "plan": {"business_goal": "查订单", "nodes": ({"node_id": "order", "capability_ref": "order/read@v1"},), "edges": ()},
    }


def test_boundary_accepts_valid_output() -> None:
    provider = _SequenceProvider([_valid_output()])
    result = R5RouterPlannerBoundary().run(user_message="查订单 20260101001", provider=provider)
    assert result.eventual_schema_valid is True
    assert result.first_attempt_schema_valid is True
    assert result.first_root_cause is None
    assert result.provider_called and result.provider_returned
    assert result.plan.derived_capabilities() == {"order/read@v1"}


def test_boundary_repairs_within_budget_and_keeps_first_root_cause() -> None:
    provider = _SequenceProvider([{"router": {"intents": ("BOGUS",), "business_goal": "x"}, "plan": {}}, _valid_output()])
    result = R5RouterPlannerBoundary().run(user_message="查订单", provider=provider)
    assert result.eventual_schema_valid is True
    assert result.first_attempt_schema_valid is False
    assert result.first_root_cause == "ROUTER_SCHEMA_INVALID"
    assert len(result.attempts) == 2


def test_boundary_exhaustion_returns_safe_clarification() -> None:
    provider = _SequenceProvider([{"bad": 1}, {"bad": 2}])
    result = R5RouterPlannerBoundary().run(user_message="???", provider=provider)
    assert result.eventual_schema_valid is False
    assert result.plan.needs_clarification is True
    assert result.plan.nodes == ()


def test_boundary_provider_error_is_recorded() -> None:
    provider = _SequenceProvider([RuntimeError("provider down")])
    result = R5RouterPlannerBoundary().run(user_message="查订单", provider=provider)
    assert result.provider_called is True
    assert result.provider_returned is False
    assert result.first_root_cause == "PROVIDER_ERROR"


def test_boundary_rejects_unadmitted_capabilities() -> None:
    out = _valid_output()
    boundary = R5RouterPlannerBoundary(allowed_capabilities=frozenset({"product/read@v1"}))
    result = boundary.run(user_message="查订单", provider=_SequenceProvider([out]))
    assert result.eventual_schema_valid is False
    assert result.first_root_cause == "CAPABILITY_NOT_ADMITTED"


def test_deterministic_adapter_outputs_are_schema_valid() -> None:
    adapter = DeterministicRouterPlannerAdapter()
    boundary = R5RouterPlannerBoundary()
    for message in ("我的订单 20260101001 到哪了？", "我想申请退货", "退货政策是什么", "有哪些商品库存"):
        result = boundary.run(user_message=message, provider=adapter)
        assert result.eventual_schema_valid is True, message


def test_budget_is_pre_registered_constant() -> None:
    with pytest.raises(ValueError):
        R5RouterPlannerBoundary(model_attempt_budget=5)
    assert R5RouterPlannerBoundary().model_attempt_budget == 2


def test_capability_vocabulary_matches_runtime_admission() -> None:
    # Every capability a plan may reference is either dispatchable as an A2A
    # read route or handled by the runtime as a guarded write / escalation.
    from agent.r4_a2a_runtime import ROUTES
    from agent.r5_a2a_runtime import register_r5_routes
    from agent.r5_plan_contracts import R5_ESCALATION_CAPABILITIES, R5_WRITE_CAPABILITIES

    register_r5_routes()
    read_capabilities = R5_ALL_CAPABILITIES - R5_WRITE_CAPABILITIES - R5_ESCALATION_CAPABILITIES
    assert read_capabilities.issubset(set(ROUTES))
    assert not (R5_WRITE_CAPABILITIES & set(ROUTES))
    assert not (R5_ESCALATION_CAPABILITIES & set(ROUTES))
