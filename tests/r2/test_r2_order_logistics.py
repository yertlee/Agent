from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from agent.domain.plan_validator import PlanLimits, PlanValidationError, PlanValidator
from agent.domain.objects import InputBinding, PlanRevision, Task
from agent.m3_bindings import BindingError, BindingResolver, ResultBinding
from agent.r1_5_router import BusinessIntent, CustomerGoalV1
from agent.r2_logistics_repository import R2_LOGISTICS_SOURCE_VERSION, R2LogisticsRepository, order_ref_hash, seed_rows
from agent.r2_order_logistics import DagCandidateV1, DagNodeV1, DeterministicR2Provider, R2OrderLogisticsRuntime, R2RuntimeError
from agent.interactive_runtime import ModelResult
from eval.harness.contracts import ExecutionMode


def _order_db(path: Path) -> tuple[str, str, str, str]:
    values = ("90000001", "2468", "carrier_test", "track_test_01")
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE orders (order_id TEXT PRIMARY KEY, phone_last4 TEXT NOT NULL, order_status TEXT, pay_status TEXT, shipment_status TEXT, created_at TEXT, shipped_at TEXT, delivered_at TEXT, carrier_code TEXT, tracking_no TEXT)")
    conn.execute("INSERT INTO orders VALUES (?,?,?,?,?,?,?,?,?,?)", (values[0], values[1], "PAID", "PAID", "IN_TRANSIT", "2026-01-01", "2026-01-02", "", values[2], values[3]))
    conn.commit(); conn.close()
    return values


def _goal(order_id: str, phone: str, carrier: str, tracking: str, kind: BusinessIntent = BusinessIntent.ORDER_AND_LOGISTICS) -> CustomerGoalV1:
    entities = {"order_id": order_id, "phone_last4": phone, "carrier_code": carrier, "tracking_no": tracking}
    if kind is BusinessIntent.ORDER_QUERY:
        entities = {"order_id": order_id, "phone_last4": phone}
        caps = ("order/read@v1",)
    elif kind is BusinessIntent.LOGISTICS_QUERY:
        entities = {"carrier_code": carrier, "tracking_no": tracking, "phone_last4": phone}
        caps = ("logistics/read@v1",)
    else:
        caps = ("order/read@v1", "logistics/read@v1")
    return CustomerGoalV1(goal_type=kind, entities=entities, required_capabilities=caps)


class ExplicitPlannerProvider:
    test_only = True

    def __call__(self, _prompt: str, schema: type):
        if schema is DagCandidateV1:
            return ModelResult(DagCandidateV1(nodes=(
                DagNodeV1(node_key="order", capability_ref="order/read@v1"),
                DagNodeV1(node_key="logistics", capability_ref="logistics/read@v1", depends_on=("order",), input_sources={"carrier_code": "order.payload.carrier_code", "tracking_no": "order.payload.tracking_no"}),
            )))
        raise AssertionError(f"unexpected schema: {schema.__name__}")


class GoalAwarePlannerProvider(ExplicitPlannerProvider):
    def __call__(self, prompt: str, schema: type):
        if schema is not DagCandidateV1:
            return super().__call__(prompt, schema)
        if "goal_type=ORDER_QUERY" in prompt:
            return ModelResult(DagCandidateV1(nodes=(DagNodeV1(node_key="order", capability_ref="order/read@v1"),)))
        if "goal_type=LOGISTICS_QUERY" in prompt:
            return ModelResult(DagCandidateV1(nodes=(DagNodeV1(node_key="logistics", capability_ref="logistics/read@v1"),)))
        return super().__call__(prompt, schema)


def _seed(path: Path, order_id: str, carrier: str, tracking: str, quality: str = "FRESH") -> None:
    seed_rows(path, [{"order_ref_hash": order_ref_hash(order_id), "carrier_code": carrier, "tracking_no": tracking, "delivery_state": "IN_TRANSIT", "shipment_status": "IN_TRANSIT", "observed_at": "2026-01-03T00:00:00Z", "data_quality": quality, "events": [{"event_code": "PICKED_UP", "event_time": "2026-01-02T00:00:00Z"}]}])


def _runtime(order_db: Path, logistics_db: Path | None, artifact: Path, *, mode: ExecutionMode = ExecutionMode.SIMULATED, failure_script=None, provider=None, allow_replan: bool = True):
    return R2OrderLogisticsRuntime(db_path=order_db, logistics_db_path=logistics_db, provider=provider or GoalAwarePlannerProvider(), artifact_root=artifact, mode=mode, failure_script=failure_script, allow_replan=allow_replan)


def test_versioned_order_derived_snapshot_is_read_only(tmp_path: Path):
    order_db = tmp_path / "orders.db"; order_id, phone, carrier, tracking = _order_db(order_db)
    logistics_db = tmp_path / "logistics.db"; _seed(logistics_db, order_id, carrier, tracking)
    metadata = R2LogisticsRepository(logistics_db).metadata()
    assert metadata["source_version"] == R2_LOGISTICS_SOURCE_VERSION
    conn = sqlite3.connect(logistics_db)
    assert conn.execute("select source_name from logistics_snapshots").fetchone()[0] == "versioned_order_derived_logistics_snapshot"
    conn.close()
    ro = sqlite3.connect(f"file:{logistics_db.resolve()}?mode=ro", uri=True)
    with pytest.raises(sqlite3.OperationalError):
        ro.execute("delete from logistics_snapshots")
    ro.close()


def test_same_customer_goal_changes_only_with_repository_world(tmp_path: Path):
    order_db = tmp_path / "orders.db"; order_id, phone, carrier, tracking = _order_db(order_db)
    goal = _goal(order_id, phone, carrier, tracking); outcomes = {}
    for quality in ("FRESH", "STALE", "CONFLICT"):
        logistics_db = tmp_path / f"{quality.lower()}.db"; _seed(logistics_db, order_id, carrier, tracking, quality)
        result = _runtime(order_db, logistics_db, tmp_path / quality).run(user_id="synthetic-user", goal=goal)
        outcomes[quality] = result
    assert all(result.route == "order_to_logistics" for result in outcomes.values())
    assert outcomes["FRESH"].code == "OK"
    assert outcomes["STALE"].code == "DATA_STALE"
    assert outcomes["CONFLICT"].code == "DATA_CONFLICT"
    assert outcomes["CONFLICT"].status == "BLOCKED"


def test_goal_types_produce_real_candidate_topologies(tmp_path: Path):
    order_db = tmp_path / "orders.db"; order_id, phone, carrier, tracking = _order_db(order_db)
    logistics_db = tmp_path / "logistics.db"; _seed(logistics_db, order_id, carrier, tracking)
    for goal_type, count in ((BusinessIntent.ORDER_QUERY, 1), (BusinessIntent.LOGISTICS_QUERY, 1), (BusinessIntent.ORDER_AND_LOGISTICS, 2)):
        result = _runtime(order_db, logistics_db, tmp_path / goal_type.value).run(user_id="synthetic-user", goal=_goal(order_id, phone, carrier, tracking, goal_type))
        assert len(result.plan_revisions[0].tasks) == count and result.status == "SUCCEEDED"


def test_timeout_policy_is_the_only_replan_trigger(tmp_path: Path):
    order_db = tmp_path / "orders.db"; order_id, phone, carrier, tracking = _order_db(order_db)
    logistics_db = tmp_path / "logistics.db"; _seed(logistics_db, order_id, carrier, tracking)
    result = _runtime(order_db, logistics_db, tmp_path / "timeout", mode=ExecutionMode.FAULT, failure_script={"logistics/query@v1": {"code": "INFRA_TIMEOUT"}}).run(user_id="synthetic-user", goal=_goal(order_id, phone, carrier, tracking))
    assert result.replan_count == 1 and len(result.plan_revisions) == 2
    assert result.freeze_bundle.failure_script is not None and result.response.startswith("订单与物流状态已核验")


def test_replan_policy_can_be_disabled_without_mutating_the_goal(tmp_path: Path):
    order_db = tmp_path / "orders.db"; order_id, phone, carrier, tracking = _order_db(order_db)
    logistics_db = tmp_path / "logistics.db"; _seed(logistics_db, order_id, carrier, tracking)
    result = _runtime(order_db, logistics_db, tmp_path / "no-replan", mode=ExecutionMode.FAULT, failure_script={"logistics/query@v1": {"code": "INFRA_TIMEOUT"}}, allow_replan=False).run(user_id="synthetic-user", goal=_goal(order_id, phone, carrier, tracking))
    assert result.replan_count == 0 and len(result.plan_revisions) == 1
    assert result.code == "INFRA_TIMEOUT" and result.status == "FAILED"


def test_missing_ownership_and_source_fail_closed(tmp_path: Path):
    order_db = tmp_path / "orders.db"; order_id, phone, carrier, tracking = _order_db(order_db)
    logistics_db = tmp_path / "logistics.db"; _seed(logistics_db, order_id, carrier, tracking)
    unauthorized = _runtime(order_db, logistics_db, tmp_path / "wrong").run(user_id="synthetic-user", goal=_goal(order_id, "9999", carrier, tracking))
    assert unauthorized.code == "AUTH_IDENTITY_MISMATCH" and all(item.payload is None for item in unauthorized.results)
    unavailable = _runtime(order_db, None, tmp_path / "missing").run(user_id="synthetic-user", goal=_goal(order_id, phone, carrier, tracking))
    assert unavailable.code == "INFRA_UNAVAILABLE"


def test_binding_resolver_is_required_for_dependent_logistics():
    tasks = {"order": Task(task_id="order", plan_revision_id="p", agent_ref="order-agent@v1", capability_refs=["order/read@v1"], output_contract="order.result.v1", failure_strategy="FAIL_RUN", side_effect="READ_ONLY", timeout_ms=1000), "logistics": Task(task_id="logistics", plan_revision_id="p", agent_ref="logistics-agent@v1", capability_refs=["logistics/read@v1"], depends_on=["order"], input_bindings=[InputBinding(name="tracking_no", kind="result", source_task_id="order", path="payload.tracking_no", required=True)], output_contract="logistics.result.v1", failure_strategy="FAIL_RUN", side_effect="READ_ONLY", timeout_ms=1000)}
    resolver = BindingResolver(tasks)
    with pytest.raises(BindingError):
        resolver.resolve_result(ResultBinding(name="tracking_no", source_task_id="order", path="payload.tracking_no", expected_contract="order.result.v1", expected_type="str"), target_task_id="logistics", run_id="r", plan_revision_id="p", results={})


def test_live_rejects_deterministic_adapter_and_fault_requires_script(tmp_path: Path):
    with pytest.raises(R2RuntimeError) as live_error:
        R2OrderLogisticsRuntime(db_path=tmp_path / "orders.db", provider=DeterministicR2Provider(), mode=ExecutionMode.LIVE)
    assert live_error.value.code == "LIVE_REQUIRES_EXTERNAL_PROVIDER"
    with pytest.raises(R2RuntimeError) as fault_error:
        R2OrderLogisticsRuntime(db_path=tmp_path / "orders.db", provider=ExplicitPlannerProvider(), mode=ExecutionMode.FAULT)
    assert fault_error.value.code == "FAULT_REQUIRES_EXPLICIT_SCRIPT"


def test_plan_validator_rejects_cycle_and_external_binding():
    p = "plan_r2"
    a = Task(task_id="a", plan_revision_id=p, agent_ref="order-agent@v1", capability_refs=["order/read@v1"], depends_on=["b"], output_contract="order.result.v1", failure_strategy="FAIL_RUN", side_effect="READ_ONLY", timeout_ms=5000)
    b = Task(task_id="b", plan_revision_id=p, agent_ref="logistics-agent@v1", capability_refs=["logistics/read@v1"], depends_on=["a"], input_bindings=[InputBinding(name="tracking_no", kind="result", source_task_id="a", path="payload.tracking_no")], output_contract="logistics.result.v1", failure_strategy="FAIL_RUN", side_effect="READ_ONLY", timeout_ms=5000)
    rev = PlanRevision(plan_revision_id=p, run_id="run", created_by="runtime", revision_reason="local_replan", version=1, status="ACTIVE", tasks=[a, b])
    validator = PlanValidator(agents=["order-agent@v1", "logistics-agent@v1"], capabilities=["order/read@v1", "logistics/read@v1"], limits=PlanLimits())
    with pytest.raises(PlanValidationError):
        validator.validate(rev)


def test_runtime_rejects_combined_candidate_without_dependency_binding(tmp_path: Path):
    order_db = tmp_path / "orders.db"; order_id, phone, carrier, tracking = _order_db(order_db)
    logistics_db = tmp_path / "logistics.db"; _seed(logistics_db, order_id, carrier, tracking)
    invalid = DagCandidateV1(nodes=(
        DagNodeV1(node_key="first", capability_ref="order/read@v1"),
        DagNodeV1(node_key="second", capability_ref="logistics/read@v1"),
    ))
    result = _runtime(order_db, logistics_db, tmp_path / "invalid", provider=DeterministicR2Provider(candidate=invalid)).run(
        user_id="synthetic-user", goal=_goal(order_id, phone, carrier, tracking)
    )
    assert result.status == "FAILED"
    assert result.code == "DAG_TOPOLOGY_INVALID"
    assert result.tool_calls == 0
