"""Evaluator construct tests: prove wrong results are rejected before model runs.

Covers the review-mandated cases: invalid schema + clarification, missing goal,
wrong entity, unrequested write, tool failure, and "business not submitted but
claimed as success".  These are execution-level guarantees provided by
``R5PlanExecutor`` and the metric definitions in the evaluator.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from eval.r5_plan_executor import ExecutionOutcome, NodeOutcome, R5PlanExecutor
from eval.r5_router_planner_eval import (
    R5_INTENT_LABELS,
    _append_checkpoint_case,
    _build_result_assertions,
    _data_run_context,
    _demo_paths,
    _format_intent_metrics,
    _handoff_legal,
    _intent_metric_counts,
    _safe_entity_summary,
    _safe_execution_outcome,
    _run_context,
    _task_outcome_ok,
    _write_checkpoint_header,
)


ROOT = Path(__file__).parents[2]
EXECUTOR = R5PlanExecutor(
    orders_db=ROOT / "ecommerce.db",
    product_db=ROOT / "data" / "r5_demo_v1.db",
    aftersales_db=ROOT / "data" / "r5_aftersales_demo_v1.db",
    logistics_db=ROOT / "data" / "r5_logistics_demo_v1.db",
)
CTX = {"order_id": "20260320001", "phone_last4": "1234", "user_id": "demo_user_1234"}


def _read_plan(capability, args, *, bindings=None, edges=(), nodes_extra=()):
    nodes = [{"node_id": capability.split("/")[0] + "_" + capability.split("/")[1].split("@")[0], "capability_ref": capability, "args": args, "bindings": bindings or {}, "failure_strategy": "FAIL_RUN"}]
    nodes.extend(nodes_extra)
    return {"schema_version": "r5.plan.v1", "business_goal": "g", "nodes": tuple(nodes), "edges": tuple(edges)}


def test_invalid_schema_plus_clarification_is_not_executed() -> None:
    # A schema-invalid plan must be refused, not executed, and cannot earn
    # credit for happening to carry a clarification flag.
    out = EXECUTOR.execute({"schema_version": "r5.plan.v1", "business_goal": "g", "nodes": [{"node_id": "x", "capability_ref": "admin/refund@v1"}], "needs_clarification": True}, trusted_context=CTX, allows_write=False)
    assert out.status == "INVALID_PLAN"
    assert out.executed is False
    assert out.terminal == "REJECT"


def test_valid_clarification_plan_stops_without_tools() -> None:
    out = EXECUTOR.execute({"schema_version": "r5.plan.v1", "business_goal": "g", "nodes": (), "edges": (), "needs_clarification": True, "clarification_reason": "missing order"}, trusted_context=CTX, allows_write=False)
    assert out.status == "CLARIFY" and out.executed is False


def test_missing_goal_plan_still_executes_but_is_not_conformant() -> None:
    # An order request answered by a product-only plan: the product read runs,
    # but the plan does not cover the required goal (evaluator marks it
    # non-conformant and not task-complete).
    plan = _read_plan("product/read@v1", {"sku": "SKU-1001"})
    out = EXECUTOR.execute(plan, trusted_context=CTX, allows_write=False)
    assert out.status == "COMPLETED"
    assert "order/read@v1" not in out.successful_reads


def test_wrong_entity_fails_business_check() -> None:
    plan = _read_plan("order/read@v1", {"order_id": "20260320001", "phone_last4": "0000"})
    out = EXECUTOR.execute(plan, trusted_context=CTX, allows_write=False)
    assert out.status == "FAILED"
    assert out.error_code in {"AUTH_IDENTITY_MISMATCH", "AUTH_RESOURCE_FORBIDDEN"}


def test_tool_failure_is_not_counted_as_success() -> None:
    plan = _read_plan("product/read@v1", {"sku": "SKU-DOES-NOT-EXIST"})
    out = EXECUTOR.execute(plan, trusted_context=CTX, allows_write=False)
    assert out.status == "FAILED"
    assert out.error_code == "DATA_MISSING"
    assert out.successful_reads == []


def test_unrequested_write_is_rejected_without_execution() -> None:
    plan = {
        "schema_version": "r5.plan.v1", "business_goal": "g",
        "nodes": (
            {"node_id": "aftersales_eligibility", "capability_ref": "aftersales/eligibility@v1", "args": {"order_id": "20260320001", "service": "refund"}, "bindings": {"phone_last4": {"kind": "context", "context_key": "phone_last4"}}, "failure_strategy": "FAIL_RUN"},
            {"node_id": "aftersales_write", "capability_ref": "aftersales/write@v1", "args": {}, "bindings": {}, "failure_strategy": "FAIL_RUN"},
        ),
        "edges": ({"upstream_node_id": "aftersales_eligibility", "downstream_node_id": "aftersales_write"},),
    }
    out = EXECUTOR.execute(plan, trusted_context=CTX, allows_write=False)
    assert out.status == "REJECT" and out.unrequested_write is True
    assert out.executed is False
    assert out.successful_reads == []  # nothing ran, including the read


def test_write_stops_at_pending_confirmation_not_submitted() -> None:
    plan = {
        "schema_version": "r5.plan.v1", "business_goal": "g",
        "nodes": (
            {"node_id": "aftersales_eligibility", "capability_ref": "aftersales/eligibility@v1", "args": {"order_id": "20260320001", "service": "refund"}, "bindings": {"phone_last4": {"kind": "context", "context_key": "phone_last4"}}, "failure_strategy": "FAIL_RUN"},
            {"node_id": "aftersales_write", "capability_ref": "aftersales/write@v1", "args": {}, "bindings": {}, "failure_strategy": "FAIL_RUN"},
        ),
        "edges": ({"upstream_node_id": "aftersales_eligibility", "downstream_node_id": "aftersales_write"},),
    }
    out = EXECUTOR.execute(plan, trusted_context=CTX, allows_write=True)
    assert out.terminal == "PENDING_CONFIRMATION"
    assert out.status == "WAITING_CONFIRMATION"
    # No submission happened: the write node was never executed.
    assert all(n.status != "SUCCEEDED" or n.capability_ref != "aftersales/write@v1" for n in out.node_results)


def test_confirmation_script_is_not_auto_supported() -> None:
    plan = {
        "schema_version": "r5.plan.v1", "business_goal": "g",
        "nodes": ({"node_id": "aftersales_write", "capability_ref": "aftersales/write@v1", "args": {}, "bindings": {}, "failure_strategy": "FAIL_RUN"},),
        "edges": (),
    }
    # Even with a caller-provided script, this iteration has no auto-approval:
    # it must fail closed rather than silently submit.
    out = EXECUTOR.execute(plan, trusted_context=CTX, allows_write=True, confirmation_script={"decision": "approve"})
    assert out.status in {"FAILED", "INVALID_PLAN"}
    assert out.terminal != "PENDING_CONFIRMATION" or out.status == "FAILED"


def test_dataset_execution_metadata_is_present() -> None:
    import json

    rows = [json.loads(line) for line in (ROOT / "eval" / "datasets" / "r5" / "dev.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 123
    for row in rows:
        assert row["execution"]["expected_terminal"] in {"ANSWER", "CLARIFY", "REJECT", "ASK_USER", "HUMAN", "PENDING_CONFIRMATION"}
        if row["execution"]["allows_write"]:
            assert row["execution"]["write_requires_confirmation"] is True
            assert "aftersales/write@v1" in row["expected"]["optional_capabilities"]


def test_handoff_escalation_reaches_human_terminal() -> None:
    plan = {
        "schema_version": "r5.plan.v1", "business_goal": "投诉",
        "nodes": ({"node_id": "human_handoff", "capability_ref": "human/handoff@v1", "args": {}, "bindings": {}, "failure_strategy": "FAIL_RUN"},),
        "edges": (),
    }
    out = EXECUTOR.execute(plan, trusted_context=CTX, allows_write=False, allows_escalation=True)
    assert out.status == "ESCALATED" and out.terminal == "HUMAN"
    assert out.executed is True


def test_unrequested_escalation_is_rejected() -> None:
    plan = {
        "schema_version": "r5.plan.v1", "business_goal": "投诉",
        "nodes": ({"node_id": "human_handoff", "capability_ref": "human/handoff@v1", "args": {}, "bindings": {}, "failure_strategy": "FAIL_RUN"},),
        "edges": (),
    }
    out = EXECUTOR.execute(plan, trusted_context=CTX, allows_write=False, allows_escalation=False)
    assert out.status == "REJECT" and out.executed is False
    assert out.error_code == "UNREQUESTED_ESCALATION"


def test_pending_confirmation_requires_waiting_status() -> None:
    common = {
        "expected_terminal": "PENDING_CONFIRMATION",
        "schema_valid": True,
        "unrequested_write": False,
        "reads_ok": True,
        "outcome_terminal": "ANSWER",
        "outcome_error": None,
    }
    assert _task_outcome_ok(**common, outcome_status="WAITING_CONFIRMATION") is True
    assert _task_outcome_ok(**common, outcome_status="COMPLETED") is False


def test_intent_metrics_use_fixed_r5_label_set() -> None:
    counts = _intent_metric_counts({"PRODUCT_QUERY"}, {"PRODUCT_QUERY", "UNREGISTERED"}, R5_INTENT_LABELS)
    metrics = _format_intent_metrics(counts)
    assert tuple(metrics) == R5_INTENT_LABELS
    assert metrics["PRODUCT_QUERY"]["f1"] == 1.0
    # A candidate-only label cannot expand the macro denominator.
    assert "UNREGISTERED" not in metrics


def test_handoff_legal_requires_required_subset_and_rejects_forbidden() -> None:
    required = {"order/read@v1"}
    optional = {"logistics/read@v1"}
    forbidden = {"aftersales/write@v1"}
    assert _handoff_legal(schema_valid=True, predicted=required, required=required, optional=optional, forbidden=forbidden)
    assert _handoff_legal(schema_valid=True, predicted=required | optional, required=required, optional=optional, forbidden=forbidden)
    assert not _handoff_legal(schema_valid=True, predicted=set(), required=required, optional=optional, forbidden=forbidden)
    assert not _handoff_legal(schema_valid=True, predicted=required | forbidden, required=required, optional=optional, forbidden=forbidden)


def test_wrong_target_success_fails_result_assertion_and_task_completion() -> None:
    outcome = ExecutionOutcome(
        status="COMPLETED",
        terminal="ANSWER",
        executed=True,
        successful_reads=["order/read@v1"],
        node_results=[NodeOutcome(
            node_id="order_read",
            capability_ref="order/read@v1",
            ok=True,
            payload={"order_id": "20260320002", "order_status": "PAID"},
            status="SUCCEEDED",
        )],
    )
    assertions, assertions_ok = _build_result_assertions(
        {"order_id": "20260320001"}, outcome, {"order/read@v1"}
    )
    assert assertions_ok is False
    assert assertions[0]["checks"][0]["matches"] is False
    assert "20260320001" not in str(assertions)
    assert _task_outcome_ok(
        expected_terminal="ANSWER",
        schema_valid=True,
        unrequested_write=False,
        reads_ok=True,
        outcome_status="COMPLETED",
        outcome_terminal="ANSWER",
        outcome_error=None,
        result_assertions_ok=assertions_ok,
    ) is False


def test_safe_entity_summary_keeps_presence_and_digest_only() -> None:
    summary = _safe_entity_summary({"order_id": "20260320001", "phone_last4": "1234"})
    assert set(summary) == {"order_id", "phone_last4"}
    assert all(set(value) == {"present", "value_digest"} for value in summary.values())
    assert "20260320001" not in str(summary)


def test_case_checkpoint_is_incremental_ndjson(tmp_path: Path) -> None:
    checkpoint = tmp_path / "r5.checkpoint.jsonl"
    _write_checkpoint_header(checkpoint, split="dev", candidate="oracle", total=1)
    _append_checkpoint_case(
        checkpoint,
        index=1,
        total=1,
        case_result={"case_id": "R5D-test", "result_assertions": [], "task_ok": False},
        elapsed_ms=12.5,
    )
    lines = checkpoint.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert '"record_type": "case"' in lines[1]


def test_optional_data_dir_uses_isolated_four_db_fixture_names(tmp_path: Path) -> None:
    paths = _demo_paths(tmp_path)
    assert [path.name for path in paths.values()] == [
        "ecommerce.db",
        "r5_demo_v1.db",
        "r5_aftersales_demo_v1.db",
        "r5_logistics_demo_v1.db",
    ]
    context = _data_run_context(tmp_path)
    assert context["source_kind"] == "custom_data_dir"
    assert set(context["database_hashes"]) == set(paths)


def test_deterministic_run_context_does_not_import_llm_or_load_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """No-key candidates must not touch the credential-loading module."""
    import builtins
    import sys

    sys.modules.pop("agent.llm", None)
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "agent.llm":
            raise AssertionError("deterministic run context imported agent.llm")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    context = _run_context("keyword_router")
    assert context["candidate"] == "keyword_router"
    assert context["model"] is None
    assert context["provider_host"] is None
    assert context["credential_present"] is False


def test_saved_execution_evidence_omits_payload_values() -> None:
    outcome = ExecutionOutcome(
        status="COMPLETED",
        terminal="ANSWER",
        executed=True,
        node_results=[NodeOutcome(node_id="product_read", capability_ref="product/read@v1", ok=True, payload={"sku": "SKU-1001", "price": 10}, status="SUCCEEDED")],
    )
    evidence = _safe_execution_outcome(outcome)
    node = evidence["node_results"][0]
    assert "payload" not in node
    assert node["payload_present"] is True
    assert node["payload_keys"] == ["price", "sku"]
    assert "SKU-1001" not in str(evidence)
