"""R5 deterministic plan normalization tests (capability argument contract)."""
from __future__ import annotations

from agent.r5_capability_args import normalize_plan
from agent.r5_plan_contracts import R5BindingV1, R5PlanEdgeV1, R5PlanNodeV1, R5PlanV1


CTX = {"phone_last4": "1234", "user_id": "demo_user_1234"}


def _plan(*nodes, edges=(), goal="g"):
    return R5PlanV1(nodes=tuple(nodes), edges=tuple(edges), business_goal=goal)


def test_missing_context_argument_is_filled_deterministically() -> None:
    plan = _plan(R5PlanNodeV1(node_id="order_read", capability_ref="order/read@v1", args={"order_id": "20260320001"}))
    result = normalize_plan(plan, entities={"order_id": "20260320001"}, trusted_context=CTX, user_text="订单 20260320001 状态")
    node = result.plan.nodes[0]
    assert node.args == {"order_id": "20260320001"}
    assert node.bindings["phone_last4"].kind == "context"
    assert node.bindings["phone_last4"].context_key == "phone_last4"
    assert result.unplannable_reason is None


def test_model_node_args_are_preserved_when_router_entity_is_missing() -> None:
    plan = _plan(R5PlanNodeV1(node_id="product_read", capability_ref="product/read@v1", args={"sku": "SKU-1001"}))
    result = normalize_plan(plan, entities={}, trusted_context=CTX, user_text="查商品")
    assert result.unplannable_reason is None
    assert result.plan.nodes[0].args == {"sku": "SKU-1001"}


def test_conflicting_router_entity_and_node_arg_fails_closed() -> None:
    plan = _plan(R5PlanNodeV1(node_id="order_read", capability_ref="order/read@v1", args={"order_id": "20260320001"}))
    result = normalize_plan(plan, entities={"order_id": "20260320002"}, trusted_context=CTX, user_text="查订单")
    assert result.plan.needs_clarification is True
    assert result.unplannable_reason == "conflicting_entity:order_id"


def test_same_kind_nodes_keep_distinct_explicit_args() -> None:
    plan = _plan(
        R5PlanNodeV1(node_id="product_read_a", capability_ref="product/read@v1", args={"sku": "SKU-1001"}),
        R5PlanNodeV1(node_id="product_read_b", capability_ref="product/read@v1", args={"sku": "SKU-1002"}),
    )
    result = normalize_plan(plan, entities={}, trusted_context=CTX, user_text="查两个商品")
    assert result.unplannable_reason is None
    assert result.plan.nodes[0].args["sku"] == "SKU-1001"
    assert result.plan.nodes[1].args["sku"] == "SKU-1002"


def test_logistics_does_not_guess_missing_carrier() -> None:
    plan = _plan(R5PlanNodeV1(node_id="logistics_read", capability_ref="logistics/read@v1", args={"tracking_no": "T1"}))
    result = normalize_plan(plan, entities={}, trusted_context=CTX, user_text="查物流 T1")
    assert result.plan.needs_clarification is True
    assert result.unplannable_reason == "missing_entity:carrier_code"


def test_invalid_derived_binding_fails_closed() -> None:
    policy = R5PlanNodeV1(node_id="policy_read", capability_ref="policy/read@v1", args={})
    logistics = R5PlanNodeV1(
        node_id="logistics_read",
        capability_ref="logistics/read@v1",
        args={},
        bindings={"carrier_code": R5BindingV1(kind="result", source_node_id="policy_read", path="carrier_code")},
    )
    plan = _plan(logistics, policy, edges=(R5PlanEdgeV1(upstream_node_id="policy_read", downstream_node_id="logistics_read"),))
    result = normalize_plan(plan, entities={}, trusted_context=CTX, user_text="查物流")
    assert result.plan.needs_clarification is True
    assert result.unplannable_reason == "invalid_binding:logistics_read:carrier_code"


def test_multiple_derived_upstreams_fail_closed_without_typed_binding() -> None:
    order_a = R5PlanNodeV1(node_id="order_read_a", capability_ref="order/read@v1", args={"order_id": "o1"})
    order_b = R5PlanNodeV1(node_id="order_read_b", capability_ref="order/read@v1", args={"order_id": "o2"})
    logistics = R5PlanNodeV1(node_id="logistics_read", capability_ref="logistics/read@v1", args={})
    plan = _plan(
        order_a,
        order_b,
        logistics,
        edges=(
            R5PlanEdgeV1(upstream_node_id="order_read_a", downstream_node_id="logistics_read"),
            R5PlanEdgeV1(upstream_node_id="order_read_b", downstream_node_id="logistics_read"),
        ),
    )
    result = normalize_plan(plan, entities={}, trusted_context=CTX, user_text="查物流")
    assert result.plan.needs_clarification is True
    assert result.unplannable_reason == "ambiguous_derived_source:logistics_read:carrier_code"


def test_missing_entity_fails_closed_to_clarification() -> None:
    plan = _plan(R5PlanNodeV1(node_id="order_read", capability_ref="order/read@v1", args={}))
    result = normalize_plan(plan, entities={}, trusted_context=CTX, user_text="我的订单")
    assert result.plan.needs_clarification is True
    assert result.unplannable_reason == "missing_entity:order_id"


def test_missing_trusted_context_fails_closed() -> None:
    plan = _plan(R5PlanNodeV1(node_id="order_read", capability_ref="order/read@v1", args={"order_id": "o1"}))
    result = normalize_plan(plan, entities={"order_id": "o1"}, trusted_context={}, user_text="订单 o1")
    assert result.plan.needs_clarification is True
    # phone_last4 is user-stated-else-context; with neither available it fails closed.
    assert result.unplannable_reason == "missing_entity:phone_last4"


def test_stated_phone_wins_over_session_context() -> None:
    plan = _plan(R5PlanNodeV1(node_id="order_read", capability_ref="order/read@v1", args={"order_id": "o1"}))
    result = normalize_plan(plan, entities={"order_id": "o1", "phone_last4": "0000"}, trusted_context=CTX, user_text="订单 o1 尾号 0000")
    node = result.plan.nodes[0]
    assert node.args["phone_last4"] == "0000"
    assert "phone_last4" not in node.bindings


def test_policy_extra_argument_is_dropped_to_match_contract() -> None:
    plan = _plan(R5PlanNodeV1(node_id="policy_read", capability_ref="policy/read@v1", args={"query": "换货多久", "service": "换货"}))
    result = normalize_plan(plan, entities={"service": "换货"}, trusted_context=CTX, user_text="换货的有效期是多久")
    assert result.plan.nodes[0].args == {"query": "换货的有效期是多久"}


def test_logistics_derives_carrier_and_tracking_from_order_node() -> None:
    order = R5PlanNodeV1(node_id="order_read", capability_ref="order/read@v1", args={"order_id": "o1"})
    logistics = R5PlanNodeV1(node_id="logistics_read", capability_ref="logistics/read@v1", args={})
    plan = _plan(order, logistics, edges=(R5PlanEdgeV1(upstream_node_id="order_read", downstream_node_id="logistics_read"),))
    result = normalize_plan(plan, entities={"order_id": "o1"}, trusted_context=CTX, user_text="订单 o1 的物流")
    node = [n for n in result.plan.nodes if n.node_id == "logistics_read"][0]
    assert node.bindings["carrier_code"].kind == "result"
    assert node.bindings["carrier_code"].source_node_id == "order_read"
    assert node.bindings["tracking_no"].path == "tracking_no"


def test_logistics_uses_entity_when_no_order_node() -> None:
    logistics = R5PlanNodeV1(node_id="logistics_read", capability_ref="logistics/read@v1", args={"tracking_no": "T1"})
    plan = _plan(logistics)
    result = normalize_plan(plan, entities={"tracking_no": "T1", "carrier_code": "yuantong"}, trusted_context=CTX, user_text="运单 T1")
    assert result.plan.nodes[0].args == {"carrier_code": "yuantong", "tracking_no": "T1"}


def test_model_arg_conformance_counts_raw_plan_only() -> None:
    conformant = R5PlanNodeV1(node_id="product_read", capability_ref="product/read@v1", args={"sku": "S1"}, bindings={})
    incomplete = R5PlanNodeV1(node_id="order_read", capability_ref="order/read@v1", args={"order_id": "o1"})
    result = normalize_plan(_plan(conformant, incomplete), entities={"sku": "S1", "order_id": "o1"}, trusted_context=CTX, user_text="x")
    assert result.model_arg_conformant_nodes == 1
    assert result.total_nodes == 2


def test_normalization_never_changes_capabilities() -> None:
    plan = _plan(
        R5PlanNodeV1(node_id="order_read", capability_ref="order/read@v1", args={"order_id": "o1"}),
        R5PlanNodeV1(node_id="logistics_read", capability_ref="logistics/read@v1", args={"tracking_no": "T1"}),
    )
    before = plan.derived_capabilities()
    result = normalize_plan(plan, entities={"order_id": "o1", "tracking_no": "T1", "carrier_code": "c"}, trusted_context=CTX, user_text="x")
    assert result.plan.derived_capabilities() == before


def test_clarification_plan_passes_through() -> None:
    plan = R5PlanV1(nodes=(), edges=(), needs_clarification=True, clarification_reason="missing", business_goal="g")
    result = normalize_plan(plan, entities={}, trusted_context=CTX, user_text="?")
    assert result.plan.needs_clarification is True
    assert result.actions == []
