import sqlite3
import threading

import pytest

from agent.storage.m2 import M2Repository, apply_m2_schema
from agent.m2_registry import build_m2_registry

from .helpers import setup_repo


def test_m2_schema_tables_and_migration_failure_rollback():
    conn = sqlite3.connect(":memory:")
    from agent.storage.migrations.m1 import apply_m1_schema
    apply_m1_schema(conn)
    with pytest.raises(RuntimeError):
        apply_m2_schema(conn, fail_after=2)
    assert not {r[0] for r in conn.execute("select name from sqlite_master where type='table'")} & {"registry_entries", "confirm_tokens", "aftersales_cases"}
    apply_m2_schema(conn)
    assert {"registry_entries", "confirm_tokens", "aftersales_cases", "tool_submit_log", "m2_audit_log"} <= {r[0] for r in conn.execute("select name from sqlite_master where type='table'")}


def test_invalid_attempt_rolls_back_token_case_and_events(tmp_path):
    repo, task = setup_repo(tmp_path)
    try:
        from agent.domain.confirm_tokens import ConfirmTokenManager
        from agent.domain.eligibility import EligibilityEngine
        from agent.domain.policy import PolicyCatalog, PolicyRule
        from agent.domain.objects import sha256_json
        from agent.domain.aftersales import AfterSalesService
        from .helpers import order_fact
        order = order_fact()
        eligibility = EligibilityEngine(PolicyCatalog([PolicyRule(rule_id="r", decision_logic="allow", source="p", effective_from=__import__("datetime").datetime(2025, 1, 1, tzinfo=__import__("datetime").timezone.utc), scope="refund", version="v1")], version="v1")).check(order, service="refund")
        _, raw = ConfirmTokenManager(repo).issue(session_id="s1", user_id="u1", run_id="r1", task_id=task.task_id, order_id="o1", service="refund", amount=__import__("decimal").Decimal("10"), payload_hash=order.snapshot_hash, topic_version="v1")
        with pytest.raises(ValueError):
            AfterSalesService(repo).create_case(session_id="s1", user_id="u1", run_id="r1", task_id=task.task_id, attempt_id="missing", order=order, eligibility=eligibility, service="refund", reason="x", amount=__import__("decimal").Decimal("10"), raw_token=raw, idempotency_key="id")
        assert repo.conn.execute("select count(*) from aftersales_cases").fetchone()[0] == 0
        assert repo.conn.execute("select status from confirm_tokens").fetchone()[0] == "ISSUED"
    finally:
        repo.close()


def test_registry_manifest_persists_without_callable_or_duplicate(tmp_path):
    repo, _ = setup_repo(tmp_path)
    try:
        for spec in build_m2_registry()._specs.values():
            repo.register_tool(spec)
        assert repo.conn.execute("select count(*) from registry_entries").fetchone()[0] == 6
        with pytest.raises(sqlite3.IntegrityError):
            repo.register_tool(next(iter(build_m2_registry()._specs.values())))
    finally:
        repo.close()


def test_active_case_unique_index_rejects_concurrent_same_owner_order(tmp_path):
    repo, task = setup_repo(tmp_path)
    path = str(tmp_path / "m2.db")
    repo.conn.execute("INSERT INTO aftersales_cases VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", ("case_a", "u1", "o1", "refund", "REQUESTED", 0, "r1", "s1", task.task_id, "id-a", "fp-a", "reason", "10", "now", "now"))
    repo.conn.commit()
    repo.close()
    outcomes = []
    lock = threading.Lock()

    def worker(case_id):
        other = M2Repository(path, initialize=False)
        try:
            other.conn.execute("BEGIN IMMEDIATE")
            other.conn.execute("INSERT INTO aftersales_cases VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (case_id, "u1", "o1", "refund", "REQUESTED", 0, "r1", "s1", task.task_id, case_id, case_id, "reason", "10", "now", "now"))
            other.conn.commit()
            outcome = "won"
        except sqlite3.IntegrityError:
            other.conn.rollback()
            outcome = "unique_conflict"
        finally:
            other.close()
        with lock:
            outcomes.append(outcome)

    # A pre-existing active row makes both concurrent candidates lose safely.
    a = threading.Thread(target=worker, args=("case_b",))
    b = threading.Thread(target=worker, args=("case_c",))
    a.start(); b.start(); a.join(); b.join()
    assert outcomes.count("unique_conflict") == 2
