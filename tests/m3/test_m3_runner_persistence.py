from __future__ import annotations

import threading
import time
import json

from agent.m2_executor import M2Executor
from agent.m3_runtime import M3ScenarioRunner
from agent.storage.m2 import M2Repository


def _mixed_case():
    return {
        "scenario_id": "persisted-mixed-join",
        "turns": ["查订单 ORD-SYN-20 并查询退货政策"],
        "world_fixture_ref": "world-synth-20",
        "expected_intent": "mixed",
        "expected_tool_path": ["get_order_info_tool", "policy_rag_search_tool"],
        "expected_terminal_class": "PASS",
        "expected_business_codes": ["OK"],
    }


def test_runner_executes_independent_reads_in_parallel_and_persists_same_attempt(tmp_path, monkeypatch):
    active = 0
    maximum = 0
    lock = threading.Lock()
    original = M2Executor.invoke

    def invoke(self, *args, **kwargs):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        try:
            time.sleep(0.03)
            return original(self, *args, **kwargs)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(M2Executor, "invoke", invoke)
    result = M3ScenarioRunner(db_dir=tmp_path).run(_mixed_case())
    assert result.observed_terminal_class == "PASS"
    assert maximum >= 2

    repo = M2Repository(result.db_path, initialize=False)
    try:
        attempts = {row[1]: row[0] for row in repo.conn.execute("SELECT attempt_id,task_id FROM task_attempts WHERE run_id=?", (result.run_id,))}
        trace_attempts = {json.loads(row[0]).get("attempt_id") for row in repo.conn.execute("SELECT envelope_json FROM trace_outbox WHERE run_id=?", (result.run_id,)) if json.loads(row[0]).get("attempt_id")}
        result_attempts = {row[0] for row in repo.conn.execute("SELECT attempt_id FROM results WHERE run_id=?", (result.run_id,))}
        assert set(attempts.values()) == trace_attempts & set(attempts.values())
        assert result_attempts == set(attempts.values())
        assert repo.conn.execute("SELECT COUNT(*) FROM trace_outbox WHERE run_id=? AND envelope_json LIKE '%parallel_join%'", (result.run_id,)).fetchone()[0] >= 1
    finally:
        repo.close()


def test_runner_persists_partial_mixed_join_without_dropping_sibling(tmp_path):
    case = _mixed_case()
    case["failure_script"] = {
        "version": "fault.v1",
        "action": "RETURN_ERROR",
        "target_tool": "policy/search@v1",
        "error_code": "INFRA_UNAVAILABLE",
    }
    result = M3ScenarioRunner(db_dir=tmp_path).run(case)
    assert result.observed_terminal_class == "FAILED"
    repo = M2Repository(result.db_path, initialize=False)
    try:
        rows = repo.conn.execute("SELECT task_id,status,business_code FROM results WHERE run_id=? ORDER BY created_at", (result.run_id,)).fetchall()
        assert len(rows) == 2
        assert {row[2] for row in rows} == {"OK", "INFRA_UNAVAILABLE"}
        assert repo.conn.execute("SELECT COUNT(*) FROM trace_outbox WHERE run_id=? AND envelope_json LIKE '%parallel_join%'", (result.run_id,)).fetchone()[0] >= 1
        assert repo.conn.execute("SELECT COUNT(*) FROM trace_outbox WHERE run_id=? AND envelope_json LIKE '%\"partial\":true%'", (result.run_id,)).fetchone()[0] >= 1
    finally:
        repo.close()


def test_mixed_join_summary_is_order_independent_when_first_branch_fails(tmp_path):
    case = _mixed_case()
    case["scenario_id"] = "persisted-mixed-order-failure"
    case["failure_script"] = {
        "version": "fault.v1",
        "action": "RETURN_ERROR",
        "target_tool": "order/get_info@v1",
        "error_code": "INFRA_UNAVAILABLE",
    }
    result = M3ScenarioRunner(db_dir=tmp_path).run(case)
    assert result.observed_terminal_class == "FAILED"
    assert result.business_code == "INFRA_UNAVAILABLE"
    repo = M2Repository(result.db_path, initialize=False)
    try:
        assert repo.conn.execute("SELECT COUNT(*) FROM results WHERE run_id=?", (result.run_id,)).fetchone()[0] == 2
    finally:
        repo.close()
