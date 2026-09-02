import inspect
import re

import pytest

from agent.agents import OrderAgent
from agent.m2_executor import M2Executor
from agent.m3_runtime import M3ScenarioRunner


def _case(**overrides):
    value = {
        "scenario_id": "real-integration",
        "turns": ["查询订单 ORD-SYN-01，后四位 1001"],
        "world_fixture_ref": "world-synth-01",
        "expected_intent": "order_query",
        "expected_tool_path": ["get_order_info_tool"],
        "expected_terminal_class": "PASS",
    }
    value.update(overrides)
    return value


def test_port_calls_canonical_m2_executor(monkeypatch):
    calls = []
    original = M2Executor.invoke
    def spy(self, *args, **kwargs):
        calls.append(args[0])
        return original(self, *args, **kwargs)
    monkeypatch.setattr(M2Executor, "invoke", spy)
    result = OrderAgent().invoke({"order_id": "ORD-SYN-01", "phone_last4": "1001"})
    assert calls == ["order/get_info@v1"]
    assert result.tool_ref == "order/get_info@v1"


def test_gold_path_is_not_an_input_to_planner(monkeypatch, tmp_path):
    captured = []
    from agent.m3_supervisor import M3Supervisor
    original = M3Supervisor.draft_operations
    def spy(self, **kwargs):
        captured.append(kwargs["operations"])
        return original(self, **kwargs)
    monkeypatch.setattr(M3Supervisor, "draft_operations", spy)
    M3ScenarioRunner(db_dir=tmp_path).run(_case(expected_tool_path=["policy_rag_search_tool"]))
    assert captured and captured[0] == ["order/get_info@v1"]


def test_failure_script_changes_observed_terminal(tmp_path):
    ok = M3ScenarioRunner(db_dir=tmp_path / "ok").run(_case())
    failed = M3ScenarioRunner(db_dir=tmp_path / "failed").run(_case(failure_script={"version": "fault.v1", "action": "RETURN_ERROR", "target_tool": "order/get_info@v1", "error_code": "ORDER_NOT_FOUND"}))
    assert ok.observed_terminal_class == "PASS"
    assert failed.observed_terminal_class == "PASS"
    assert failed.business_code == "ORDER_NOT_FOUND"


def test_runtime_fault_and_block_remain_non_terminal_success(tmp_path):
    fault = M3ScenarioRunner(db_dir=tmp_path / "fault").run(_case(world_fixture_ref="world-synth-15", turns=["查询物流运单 LOG-SYN-15"], expected_intent="logistics", expected_tool_path=["query_logistics_snapshot_tool"], expected_terminal_class="FAILED"))
    blocked = M3ScenarioRunner(db_dir=tmp_path / "blocked").run(_case(world_fixture_ref="world-synth-23", turns=["高风险售后请求需要审核"], expected_terminal_class="BLOCKED"))
    assert fault.observed_terminal_class == "FAILED"
    assert blocked.observed_terminal_class == "BLOCKED"


def test_runtime_persistence_reopens_and_has_core_evidence(tmp_path):
    result = M3ScenarioRunner(db_dir=tmp_path).run(_case())
    from agent.storage.m2 import M2Repository
    repo = M2Repository(result.db_path, initialize=False)
    try:
        assert repo.conn.execute("select status from runs where run_id=?", (result.run_id,)).fetchone()[0] == "SUCCEEDED"
        assert repo.conn.execute("select count(*) from plan_revisions where run_id=?", (result.run_id,)).fetchone()[0] == 1
        assert repo.conn.execute("select count(*) from task_attempts where run_id=? and status='SUCCEEDED'", (result.run_id,)).fetchone()[0] == 1
        assert repo.conn.execute("select count(*) from results where run_id=?", (result.run_id,)).fetchone()[0] == 1
        assert repo.conn.execute("select count(*) from trace_outbox where run_id=?", (result.run_id,)).fetchone()[0] > 0
    finally:
        repo.close()


def test_world_fingerprint_mismatch_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="fingerprint|hash"):
        M3ScenarioRunner(db_dir=tmp_path).run(_case(initial_world_hash="0" * 64))


def test_runtime_has_no_reports_dependency():
    from agent import m3_runtime
    assert "reports" not in inspect.getsource(m3_runtime)
