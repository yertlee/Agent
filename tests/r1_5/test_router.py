from __future__ import annotations

import pytest

from agent.r1_5_router import (
    BusinessIntent,
    DeterministicRouterAdapter,
    RouterContractError,
    RoutingDecisionV1,
    StructuredRouter,
    to_customer_goal,
)


def _decision(intent: BusinessIntent = BusinessIntent.ORDER_AND_LOGISTICS) -> RoutingDecisionV1:
    return RoutingDecisionV1(
        primary_intent=intent,
        entities={
            "order_id": "90000001",
            "phone_last4": "2468",
            "carrier_code": "carrier_test",
            "tracking_no": "track_test_01",
        },
        confidence=0.9,
    )


def test_router_returns_structured_decision_and_policy_capabilities():
    router = StructuredRouter(DeterministicRouterAdapter(_decision()))
    decision = router.route("arbitrary text with no routing keywords")
    assert isinstance(decision, RoutingDecisionV1)
    assert decision.primary_intent is BusinessIntent.ORDER_AND_LOGISTICS
    assert decision.required_capabilities == ("order/read@v1", "logistics/read@v1")
    goal = to_customer_goal(decision)
    assert goal.goal_type is BusinessIntent.ORDER_AND_LOGISTICS
    assert not goal.needs_clarification


def test_model_cannot_escalate_capability_policy():
    bad = _decision().model_copy(update={"required_capabilities": ("aftersales/write@v1",)})
    with pytest.raises(RouterContractError):
        StructuredRouter(DeterministicRouterAdapter(bad)).route("x")


def test_negative_synonym_is_not_keyword_accepted_by_deterministic_adapter():
    adapter = DeterministicRouterAdapter(_decision(BusinessIntent.ORDER_QUERY))
    decision = StructuredRouter(adapter).route("不要查物流，只查订单")
    assert decision.primary_intent is BusinessIntent.ORDER_QUERY
    assert to_customer_goal(decision).goal_type is BusinessIntent.ORDER_QUERY


def test_r2_goal_boundary_rejects_non_order_logistics_intents():
    decision = _decision(BusinessIntent.POLICY_QA)
    with pytest.raises(RouterContractError):
        to_customer_goal(decision)


def test_missing_entities_require_clarification_by_policy():
    decision = RoutingDecisionV1(primary_intent=BusinessIntent.LOGISTICS_QUERY, entities={}, confidence=0.8)
    goal = to_customer_goal(decision)
    assert goal.needs_clarification
