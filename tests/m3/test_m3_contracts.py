from datetime import datetime, timezone

import pytest

from agent.agents import build_m3_manifests, validate_manifests
from agent.m3_bindings import BindingError, BindingResolver, ContextBinding, ResultBinding
from agent.m3_registry import build_m3_registry
from agent.m3_supervisor import CandidateIntent, IntentRouter, M3Supervisor
from agent.domain.objects import InputBinding, Result, ResultStatus, Task
from agent.domain.objects import sha256_json


def task(tid, rev="p", deps=None, contract="order.result.v1"):
    return Task(task_id=tid, plan_revision_id=rev, agent_ref="order-agent@v1", capability_refs=["order/read@v1"], depends_on=deps or [], output_contract=contract, failure_strategy="FAIL_RUN", side_effect="READ_ONLY", timeout_ms=1000)


def test_m3_registry_admits_product_and_manifests_are_subsets():
    registry = build_m3_registry()
    manifests = build_m3_manifests()
    validate_manifests(manifests, registry)
    assert "product/get@v1" in registry.refs()
    assert "product/read@v1" in {item["capability_ref"] for item in registry.as_manifest()}
    assert len(manifests) == 5


def test_product_port_is_typed_and_unknown_sku_is_canonical_error():
    from agent.agents import ProductAgent
    port = ProductAgent()
    success = port.invoke({"sku": "SKU-FIXTURE-001"})
    missing = port.invoke({"sku": "SKU-MISSING"})
    assert success.ok and success.contract == "product.result.v1" and len(success.payload_hash) == 64
    assert not missing.ok and missing.error_code == "ORDER_NOT_FOUND"


def test_router_is_candidate_only_and_supervisor_activates_validated_plan():
    candidate = IntentRouter().classify("帮我查订单")
    assert isinstance(candidate, CandidateIntent) and candidate.intent == "ORDER"
    supervisor = M3Supervisor()
    draft = supervisor.draft_plan(run_id="run_m3", candidate_intent=candidate)
    assert draft.tasks and draft.candidate_intent == candidate
    revision = supervisor.activate_plan(draft)
    assert revision.status.value == "ACTIVE"


def test_binding_checks_dependency_contract_hash_and_context_path():
    tasks = {"a": task("a"), "b": task("b", deps=["a"])}
    resolver = BindingResolver(tasks)
    binding = ResultBinding(name="order", source_task_id="a", path="payload.order_id", expected_contract="order.result.v1", expected_type="str")
    result = Result(result_id="res", run_id="r", plan_revision_id="p", task_id="a", attempt_id="att", status=ResultStatus.SUCCEEDED, output_contract="order.result.v1", payload={"order_id": "o"}, business_code="OK")
    assert resolver.resolve_result(binding, target_task_id="b", run_id="r", plan_revision_id="p", results={"a": result}) == "o"
    with pytest.raises(BindingError):
        resolver.resolve_result(binding.model_copy(update={"source_task_id": "b"}), target_task_id="a", run_id="r", plan_revision_id="p", results={"b": result})
    with pytest.raises(BindingError):
        resolver.resolve_result(binding, target_task_id="b", run_id="other", plan_revision_id="p", results={"a": result})
    with pytest.raises(BindingError):
        resolver.resolve_context(ContextBinding(name="x", path="user_input"), type("C", (), {})())


def test_binding_input_binding_is_closed_over_dependencies():
    resolver = BindingResolver({"a": task("a"), "b": task("b")})
    with pytest.raises(BindingError):
        resolver.validate(InputBinding(name="x", kind="result", source_task_id="a", path="payload.id"), target_task_id="b")


def test_supervisor_persists_active_revision_and_replan_supersedes_without_reuse(tmp_path):
    from agent.domain.objects import Run
    from agent.storage.repositories import M1Repository
    repo = M1Repository(str(tmp_path / "m3.db"))
    try:
        repo.create_session("s", "u")
        repo.create_run(Run(run_id="r", session_id="s", initial_world_hash=sha256_json({"fixture": "m3"})))
        supervisor = M3Supervisor(repository=repo)
        first = task("a", rev="p0")
        second = task("b", rev="p0", deps=["a"])
        second = second.model_copy(update={"input_bindings": [InputBinding(name="order", kind="result", source_task_id="a", path="payload.order_id", required=True)]})
        from agent.m3_supervisor import PlanDraft
        draft = PlanDraft(draft_id="d0", run_id="r", candidate_intent=IntentRouter().classify("查订单"), tasks=(first, second))
        active = supervisor.activate_plan(draft)
        old_attempt = __import__("agent.domain.objects", fromlist=["TaskAttempt"]).TaskAttempt(attempt_id="old-a", run_id="r", plan_revision_id=active.plan_revision_id, task_id="a", agent_ref="order-agent@v1", attempt_no=1, input_hash=sha256_json({"old": 1}))
        repo.create_attempt(old_attempt)
        from agent.m1_runtime import Supervisor as M1Supervisor
        M1Supervisor(repo).start_attempt("old-a")
        repo.append_result(Result(result_id="old-result", run_id="r", plan_revision_id=active.plan_revision_id, task_id="a", attempt_id="old-a", status=ResultStatus.SUCCEEDED, output_contract="order.result.v1", payload={"order_id": "o1"}, business_code="OK"))
        replanned = supervisor.replan(active, tasks=list(active.tasks))
        assert repo.conn.execute("select count(*) from plan_revisions").fetchone()[0] == 2
        assert replanned.supersedes_plan_revision_id == active.plan_revision_id
        assert all(a.task_id != b.task_id for a in active.tasks for b in replanned.tasks)
        new_a, new_b = replanned.tasks
        assert new_b.depends_on == [new_a.task_id]
        assert new_b.input_bindings[0].source_task_id == new_a.task_id
        assert repo.conn.execute("select count(*) from results where result_id='old-result'").fetchone()[0] == 1
        new_attempt = __import__("agent.domain.objects", fromlist=["TaskAttempt"]).TaskAttempt(attempt_id="new-a", run_id="r", plan_revision_id=replanned.plan_revision_id, task_id=new_a.task_id, agent_ref="order-agent@v1", attempt_no=1, input_hash=sha256_json({"new": 1}))
        repo.create_attempt(new_attempt)
        assert repo.conn.execute("select plan_revision_id from task_attempts where attempt_id='new-a'").fetchone()[0] == replanned.plan_revision_id
        assert repo.conn.execute("select count(*) from task_attempts where attempt_id='old-a'").fetchone()[0] == 1
    finally:
        repo.close()
