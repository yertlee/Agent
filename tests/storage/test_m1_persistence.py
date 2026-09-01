import sqlite3
import tempfile
import unittest
import threading
from pathlib import Path

from agent.domain.objects import PlanRevision, PlanStatus, Result, ResultStatus, Run, Task, TaskAttempt, AttemptStatus, sha256_json
from agent.storage.repositories import M1Repository
from agent.m1_runtime import CASConflict, Supervisor


class M1StorageTests(unittest.TestCase):
    def setUp(self):
        self.repo = M1Repository()
        self.repo.create_session("s1", "u1")
        self.run = Run(run_id="r1", session_id="s1", initial_world_hash=sha256_json({"world": 1}))
        self.repo.create_run(self.run)
        self.task = Task(task_id="t1", plan_revision_id="p1", agent_ref="order@v1", capability_refs=["order/read@v1"], output_contract="order.read.v1", failure_strategy="FAIL_RUN", side_effect="READ_ONLY", timeout_ms=1000)
        self.plan = PlanRevision(plan_revision_id="p1", run_id="r1", created_by="supervisor", revision_reason="initial", version=1, status=PlanStatus.ACTIVE, tasks=[self.task])
        self.repo.create_plan_revision(self.plan)

    def tearDown(self): self.repo.close()

    def test_parent_chain_and_result_immutable(self):
        attempt = TaskAttempt(attempt_id="a1", run_id="r1", plan_revision_id="p1", task_id="t1", agent_ref="order@v1", attempt_no=1, status=AttemptStatus.SUCCEEDED, input_hash=sha256_json({"q": 1}))
        self.repo.create_attempt(attempt)
        result = Result(result_id="res1", run_id="r1", plan_revision_id="p1", task_id="t1", attempt_id="a1", status=ResultStatus.SUCCEEDED, output_contract="order.read.v1", payload={"order_id": "safe"})
        self.repo.append_result(result)
        self.assertEqual(self.repo.get_result_for_attempt("a1")["payload_hash"], result.payload_hash)
        self.assertFalse(hasattr(self.repo, "cas_shared_state"))
        self.assertFalse(hasattr(self.repo, "reserve_seq"))
        self.assertEqual(self.repo.conn.execute("select count(*) from trace_outbox where run_id='r1'").fetchone()[0], 1)
        with self.assertRaises(sqlite3.IntegrityError): self.repo.conn.execute("UPDATE results SET payload_json=?", ("{}",))
        with self.assertRaises(sqlite3.IntegrityError): self.repo.conn.execute("DELETE FROM results")
        with self.assertRaises(sqlite3.IntegrityError): self.repo.conn.execute("UPDATE plan_revisions SET status='COMPLETED' WHERE plan_revision_id='p1'")

    def test_cas_conflict_and_checkpoint_atomicity(self):
        sup = Supervisor(self.repo)
        self.assertEqual(sup.reduce("r1", expected_version=0, patch={"shared_state": {"order": "safe"}}, checkpoint_id="cp1"), 1)
        with self.assertRaises(CASConflict): sup.reduce("r1", expected_version=0, patch={"shared_state": {"x": 1}}, checkpoint_id="cp2")
        row = self.repo.conn.execute("SELECT state_version, snapshot_json FROM checkpoints WHERE run_id='r1'").fetchone()
        self.assertEqual(row[0], 1); self.assertIn("order", row[1])

    def test_schema_foreign_keys_and_unique_attempts(self):
        with self.assertRaises(sqlite3.IntegrityError): self.repo.conn.execute("INSERT INTO task_attempts(attempt_id,run_id,plan_revision_id,task_id,agent_ref,attempt_no,status,input_hash,created_at,updated_at) VALUES ('bad','r1','p1','missing','a',1,'CREATED',?, ?, ?)", (sha256_json({'x': 1}), 'now', 'now'))
        with self.assertRaises(sqlite3.IntegrityError): self.repo.conn.execute("INSERT INTO runs(run_id,session_id,initial_world_hash,status,created_at,updated_at) VALUES ('bad','s1','short','CREATED','now','now')")

    def test_cross_run_reference_rejected_and_retry_is_new_attempt_result(self):
        attempt = TaskAttempt(attempt_id="a1", run_id="r1", plan_revision_id="p1", task_id="t1", agent_ref="order@v1", attempt_no=1, status=AttemptStatus.SUCCEEDED, input_hash=sha256_json({"retry": 1}))
        self.repo.create_attempt(attempt)
        self.assertEqual(self.repo.next_attempt_no("t1"), 2)
        retry = attempt.model_copy(update={"attempt_id": "a2", "attempt_no": 2})
        self.repo.create_attempt(retry)
        first = Result(result_id="res1", run_id="r1", plan_revision_id="p1", task_id="t1", attempt_id="a1", status=ResultStatus.SUCCEEDED, output_contract="order.read.v1", payload={"ok": 1})
        second = Result(result_id="res2", run_id="r1", plan_revision_id="p1", task_id="t1", attempt_id="a2", status=ResultStatus.SUCCEEDED, output_contract="order.read.v1", payload={"ok": 2})
        self.repo.append_result(first); self.repo.append_result(second)
        self.assertEqual(self.repo.get_result("res1"), self.repo.get_result_for_attempt("a1"))
        with self.assertRaises(ValueError):
            self.repo.append_result(first.model_copy(update={"result_id": "bad", "run_id": "r2"}))

    def test_checkpoint_insert_failure_rolls_back_shared_state(self):
        sup = Supervisor(self.repo)
        sup.reduce("r1", expected_version=0, patch={"shared_state": {"v": 1}}, checkpoint_id="cp1")
        with self.assertRaises(sqlite3.IntegrityError):
            sup.reduce("r1", expected_version=1, patch={"shared_state": {"v": 2}}, checkpoint_id="cp1")
        state = self.repo.conn.execute("SELECT state_version,shared_state_json FROM runs WHERE run_id='r1'").fetchone()
        self.assertEqual(state[0], 1); self.assertIn('"v":1', state[1])

    def test_result_and_result_written_event_rollback_together(self):
        attempt = TaskAttempt(attempt_id="a_atomic", run_id="r1", plan_revision_id="p1", task_id="t1", agent_ref="order@v1", attempt_no=1, input_hash=sha256_json({"atomic": 1}))
        self.repo.create_attempt(attempt)
        result = Result(result_id="res_atomic", run_id="r1", plan_revision_id="p1", task_id="t1", attempt_id="a_atomic", status=ResultStatus.SUCCEEDED, output_contract="order.read.v1", payload={"ok": True})
        with self.assertRaises(ValueError): self.repo.append_result_with_event(result, event_payload={"token": "raw"})
        self.assertIsNone(self.repo.get_result("res_atomic"))
        self.assertEqual(self.repo.conn.execute("select next_seq_no from runs where run_id='r1'").fetchone()[0], 1)

    def test_reducer_event_reference_failure_rolls_back_state_checkpoint_and_seq(self):
        with self.assertRaises(sqlite3.IntegrityError):
            Supervisor(self.repo).reduce("r1", expected_version=0, patch={"shared_state": {"bad": True}}, checkpoint_id="cp_bad", plan_revision_id="missing")
        row = self.repo.conn.execute("select state_version,shared_state_json,next_seq_no from runs where run_id='r1'").fetchone()
        self.assertEqual(tuple(row), (0, "{}", 1))
        self.assertEqual(self.repo.conn.execute("select count(*) from checkpoints where run_id='r1'").fetchone()[0], 0)
        self.assertEqual(self.repo.conn.execute("select count(*) from trace_outbox where run_id='r1'").fetchone()[0], 0)

    def test_plan_revision_tasks_and_active_pointer_rollback_together(self):
        duplicate = self.task.model_copy(update={"task_id": "t1"})
        bad_plan = self.plan.model_copy(update={"plan_revision_id": "p_bad", "tasks": [self.task.model_copy(update={"plan_revision_id": "p_bad"}), duplicate.model_copy(update={"plan_revision_id": "p_bad"})]})
        with self.assertRaises(sqlite3.IntegrityError): self.repo.create_plan_revision(bad_plan)
        self.assertIsNone(self.repo.conn.execute("select 1 from plan_revisions where plan_revision_id='p_bad'").fetchone())
        self.assertEqual(self.repo.conn.execute("select plan_revision_id from runs where run_id='r1'").fetchone()[0], "p1")

    def test_two_supervisors_race_one_cas_wins(self):
        path = tempfile.mktemp(suffix=".db")
        first = M1Repository(path); first.create_session("s", "u"); first.create_run(Run(run_id="r", session_id="s", initial_world_hash=sha256_json({"race": 1})))
        second = M1Repository(path)
        barrier = threading.Barrier(2); outcomes = []
        def worker(repo, value, checkpoint):
            try:
                barrier.wait()
                Supervisor(repo).reduce("r", expected_version=0, patch={"shared_state": {"winner": value}}, checkpoint_id=checkpoint)
                outcomes.append("won")
            except CASConflict:
                outcomes.append("conflict")
        a = threading.Thread(target=worker, args=(first, "a", "ca")); b = threading.Thread(target=worker, args=(second, "b", "cb")); a.start(); b.start(); a.join(); b.join()
        self.assertEqual(sorted(outcomes), ["conflict", "won"])
        first.close(); second.close()
