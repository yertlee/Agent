"""R5 entity-candidate policy and bounded replan tests."""
from __future__ import annotations

import pytest

from agent.r5_entity_candidates import ambiguous_kinds, extract_candidates, single_values
from agent.r5_replan import BoundedReplanPolicy
from agent.r5_router_planner import R5RouterPlannerBoundary, build_router_planner_prompt


class _FixedProvider:
    def __init__(self, router: dict, plan: dict):
        self.router, self.plan = router, plan

    def __call__(self, prompt, router_schema, plan_schema):
        return {"router": self.router, "plan": self.plan}


def _order_plan(order_args=None, entities=None):
    router = {"intents": ("ORDER_QUERY",), "entities": entities or {}, "business_goal": "查订单"}
    plan = {"schema_version": "r5.plan.v1", "business_goal": "查订单", "nodes": ({"node_id": "order_read", "capability_ref": "order/read@v1", "args": order_args or {}, "bindings": {}, "failure_strategy": "FAIL_RUN"},), "edges": ()}
    return router, plan


# ---------------------------------------------------------------- candidates

def test_extract_order_id_and_sku_and_phone() -> None:
    candidates = extract_candidates("订单 20260320001，尾号 1234，商品 SKU-1001")
    kinds = {c.kind: c.value for c in candidates}
    assert kinds["order_id"] == "20260320001"
    assert kinds["sku"] == "SKU-1001"
    assert kinds["phone_last4"] == "1234"


def test_tracking_requires_logistics_context() -> None:
    assert not [c for c in extract_candidates("数字 7609205232746 出现但无上下文") if c.kind == "tracking_no"]
    assert [c for c in extract_candidates("运单号 7609205232746 到哪了") if c.kind == "tracking_no"]


def test_two_order_ids_are_ambiguous() -> None:
    candidates = extract_candidates("订单 20260320001 和 20260320002 哪个先发")
    assert ambiguous_kinds(candidates) == {"order_id": ["20260320001", "20260320002"]}
    assert single_values(candidates) == {}


def test_single_value_fill() -> None:
    candidates = extract_candidates("订单 20260320001 现在什么状态")
    assert single_values(candidates)["order_id"] == "20260320001"


# ------------------------------------------------------------- entity policy

def test_ambiguous_without_model_choice_forces_clarification() -> None:
    router, plan = _order_plan()
    result = R5RouterPlannerBoundary().run(user_message="订单 20260320001 和 20260320002 哪个先发", provider=_FixedProvider(router, plan))
    assert result.plan.needs_clarification is True
    assert result.plan.clarification_reason == "ambiguous_entity_candidates"


def test_ambiguous_with_model_choice_is_respected() -> None:
    router, plan = _order_plan(entities={"order_id": "20260320001"})
    plan["nodes"][0]["args"] = {"order_id": "20260320001"}
    result = R5RouterPlannerBoundary().run(user_message="订单 20260320001 和 20260320002 哪个先发", provider=_FixedProvider(router, plan))
    assert result.plan.needs_clarification is False
    assert len(result.plan.nodes) == 1


def test_single_candidate_is_filled_deterministically() -> None:
    router, plan = _order_plan(order_args={"order_id": "20260320001"})
    result = R5RouterPlannerBoundary().run(user_message="订单 20260320001 现在什么状态", provider=_FixedProvider(router, plan))
    assert result.router.entities.get("order_id") == "20260320001"
    assert result.plan.needs_clarification is False


def test_missing_required_entity_forces_clarification() -> None:
    router, plan = _order_plan()
    result = R5RouterPlannerBoundary().run(user_message="我的订单现在什么状态", provider=_FixedProvider(router, plan))
    assert result.plan.needs_clarification is True
    assert str(result.plan.clarification_reason).startswith("missing_required_entity")


def test_prompt_exposes_shared_capability_argument_contract() -> None:
    prompt = build_router_planner_prompt("查物流", allowed_capabilities=frozenset({"order/read@v1", "logistics/read@v1"}))
    assert "logistics/read@v1" in prompt
    assert "carrier_code" in prompt and "tracking_no" in prompt
    assert "order/read@v1" in prompt
    assert "plan.edges" in prompt
    # Runtime-supplied arguments must not become clarification reasons.
    assert "不要因此请求澄清" in prompt
    assert "phone_last4" in prompt


# ------------------------------------------------------------------- replan

def test_replan_requires_retryable_error() -> None:
    policy = BoundedReplanPolicy({"order/read@v1": ("order/read@v1",)}, budget=1)
    assert policy.decide(capability_ref="order/read@v1", error_code="DATA_MISSING").replan is False
    assert policy.decide(capability_ref="order/read@v1", error_code="INFRA_TIMEOUT").replan is True


def test_replan_respects_budget() -> None:
    policy = BoundedReplanPolicy({"order/read@v1": ("order/read@v1",)}, budget=1)
    assert policy.decide(capability_ref="order/read@v1", error_code="INFRA_TIMEOUT").replan is True
    assert policy.decide(capability_ref="order/read@v1", error_code="INFRA_TIMEOUT").replan is False


def test_replan_rejects_write_and_incompatible_alternative() -> None:
    policy = BoundedReplanPolicy({"aftersales/write@v1": ("aftersales/write@v1",), "order/read@v1": ("logistics/read@v1",)}, budget=2)
    assert policy.decide(capability_ref="aftersales/write@v1", error_code="INFRA_TIMEOUT", side_effect="WRITE").replan is False
    # logistics requires different args than order, so it is not arg-compatible
    decision = policy.decide(capability_ref="order/read@v1", error_code="INFRA_TIMEOUT")
    assert decision.replan is False and decision.reason == "no_legitimate_alternative"


def test_replan_without_declared_alternative_stops() -> None:
    policy = BoundedReplanPolicy({}, budget=1)
    decision = policy.decide(capability_ref="order/read@v1", error_code="INFRA_UNAVAILABLE")
    assert decision.replan is False and decision.reason == "no_legitimate_alternative"
