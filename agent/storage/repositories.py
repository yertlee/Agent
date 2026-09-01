"""M1 persistence repositories and atomic CAS/checkpoint/outbox operations."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from agent.domain.objects import (
    PlanRevision, Result, Run, Task, TaskAttempt, canonical_json, model_hash,
)
from .migrations.m1 import apply_m1_schema


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class M1Repository:
    """One repository over a dedicated runtime SQLite DB."""
    def __init__(self, db_path: str = ":memory:", *, initialize: bool = True):
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        if initialize:
            apply_m1_schema(self.conn)

    def close(self) -> None:
        self.conn.close()

    def create_session(self, session_id: str, user_id: str, topic_version: str = "v1", client_session_key: Optional[str] = None) -> None:
        self.conn.execute("INSERT INTO sessions VALUES (?,?,?,?,?, ?,?)", (session_id, user_id, topic_version, "ACTIVE", client_session_key or session_id, _now(), _now()))
        self.conn.commit()

    def create_run(self, run: Run, *, client_run_key: Optional[str] = None) -> None:
        self.conn.execute("INSERT INTO runs(run_id,session_id,plan_revision_id,initial_world_hash,status,state_version,next_seq_no,freeze_status,shared_state_json,created_at,updated_at,client_run_key) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", (run.run_id, run.session_id, run.plan_revision_id, run.initial_world_hash, run.status.value, run.state_version, run.next_seq_no, run.freeze_status, canonical_json(run.shared_state), run.created_at.isoformat(), run.updated_at.isoformat(), client_run_key))
        self.conn.commit()

    def create_plan_revision(self, revision: PlanRevision) -> None:
        try:
            self.conn.execute("BEGIN")
            self.conn.execute("INSERT INTO plan_revisions VALUES (?,?,?,?,?,?,?,?,?,?,?)", (revision.plan_revision_id, revision.run_id, revision.schema_version, revision.created_by, revision.revision_reason, revision.supersedes_plan_revision_id, revision.version, revision.status.value, canonical_json(revision.tasks), revision.created_at.isoformat(), _now()))
            for task in revision.tasks:
                self.create_task(task, commit=False)
            self.conn.execute("UPDATE runs SET plan_revision_id=?, updated_at=? WHERE run_id=?", (revision.plan_revision_id, _now(), revision.run_id))
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def create_task(self, task: Task, *, commit: bool = True) -> None:
        self.conn.execute("INSERT INTO tasks VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (task.task_id, self._run_for_plan(task.plan_revision_id), task.plan_revision_id, task.agent_ref, canonical_json(task.capability_refs), canonical_json(task.depends_on), canonical_json(task.input_bindings), task.output_contract, task.failure_strategy, task.side_effect, task.timeout_ms, task.deadline.isoformat() if task.deadline else None, model_hash(task), _now(), _now()))
        if commit: self.conn.commit()

    def _run_for_plan(self, plan_revision_id: str) -> str:
        row = self.conn.execute("SELECT run_id FROM plan_revisions WHERE plan_revision_id=?", (plan_revision_id,)).fetchone()
        if not row: raise ValueError("plan revision parent does not exist")
        return str(row[0])

    def create_attempt(self, attempt: TaskAttempt) -> None:
        parent = self.conn.execute("SELECT t.run_id,t.plan_revision_id FROM tasks t WHERE t.task_id=?", (attempt.task_id,)).fetchone()
        if not parent or (str(parent[0]), str(parent[1])) != (attempt.run_id, attempt.plan_revision_id):
            raise ValueError("attempt parent references do not match task")
        self.conn.execute("INSERT INTO task_attempts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", (attempt.attempt_id, attempt.run_id, attempt.plan_revision_id, attempt.task_id, attempt.agent_ref, attempt.attempt_no, attempt.status.value, attempt.input_hash, attempt.deadline.isoformat() if attempt.deadline else None, attempt.started_at.isoformat() if attempt.started_at else None, attempt.ended_at.isoformat() if attempt.ended_at else None, _now(), _now()))
        self.conn.commit()

    def next_attempt_no(self, task_id: str) -> int:
        row = self.conn.execute("SELECT COALESCE(MAX(attempt_no),0)+1 FROM task_attempts WHERE task_id=?", (task_id,)).fetchone()
        return int(row[0])

    def append_result(self, result: Result) -> None:
        """Safe public result writer; RESULT_WRITTEN is always co-committed."""
        self.append_result_with_event(result)

    def _append_result_locked(self, result: Result) -> None:
        # Parent references are checked explicitly before SQLite FK checks produce a generic error.
        row = self.conn.execute("SELECT run_id,plan_revision_id,task_id FROM task_attempts WHERE attempt_id=?", (result.attempt_id,)).fetchone()
        if not row or tuple(row) != (result.run_id, result.plan_revision_id, result.task_id):
            raise ValueError("result parent references do not match attempt")
        self.conn.execute("INSERT INTO results VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (result.result_id, result.schema_version, result.run_id, result.plan_revision_id, result.task_id, result.attempt_id, result.status.value, result.output_contract, canonical_json(result.payload) if result.payload is not None else None, result.payload_hash, result.business_code, result.error_ref, canonical_json(result.evidence_refs), canonical_json(result.claim_refs), canonical_json(result.usage), result.created_at.isoformat(), _now()))

    def _validate_event_refs_locked(self, event) -> None:
        run = self.conn.execute("SELECT session_id FROM runs WHERE run_id=?", (event.run_id,)).fetchone()
        if not run or str(run[0]) != event.session_id:
            raise ValueError("trace event run/session mismatch")
        if event.plan_revision_id:
            row = self.conn.execute("SELECT run_id FROM plan_revisions WHERE plan_revision_id=?", (event.plan_revision_id,)).fetchone()
            if not row or str(row[0]) != event.run_id: raise ValueError("trace plan reference mismatch")
        if event.task_id:
            row = self.conn.execute("SELECT run_id,plan_revision_id FROM tasks WHERE task_id=?", (event.task_id,)).fetchone()
            if not row or str(row[0]) != event.run_id or (event.plan_revision_id and str(row[1]) != event.plan_revision_id): raise ValueError("trace task reference mismatch")
        if event.attempt_id:
            row = self.conn.execute("SELECT run_id,plan_revision_id,task_id FROM task_attempts WHERE attempt_id=?", (event.attempt_id,)).fetchone()
            if not row or str(row[0]) != event.run_id or (event.plan_revision_id and str(row[1]) != event.plan_revision_id) or (event.task_id and str(row[2]) != event.task_id): raise ValueError("trace attempt reference mismatch")
        if event.parent_event_id:
            row = self.conn.execute("SELECT run_id,seq_no FROM trace_outbox WHERE event_id=?", (event.parent_event_id,)).fetchone()
            if not row or str(row[0]) != event.run_id or int(row[1]) >= event.seq_no: raise ValueError("trace parent must be earlier event in same run")

    def _append_event_locked(self, event) -> None:
        from agent.trace.events import sensitive_surface_scan
        if sensitive_surface_scan(event.payload): raise ValueError("sensitive raw field in trace payload")
        self._validate_event_refs_locked(event)
        now = _now()
        self.conn.execute("INSERT INTO trace_outbox(event_id,run_id,seq_no,envelope_json,payload_hash,created_at,updated_at) VALUES (?,?,?,?,?,?,?)", (event.trace_id, event.run_id, event.seq_no, canonical_json(event), event.payload_hash, now, now))

    def append_result_with_event(self, result: Result, *, event_payload: Optional[dict[str, Any]] = None, parent_event_id: Optional[str] = None):
        """Atomically append Result and its RESULT_WRITTEN outbox event."""
        from agent.trace.events import build_event
        try:
            self.conn.execute("BEGIN IMMEDIATE")
            self._append_result_locked(result)
            # Result publication closes the private attempt in the same
            # transaction, so a terminal Result cannot coexist with RUNNING.
            attempt = self.conn.execute("SELECT status FROM task_attempts WHERE attempt_id=?", (result.attempt_id,)).fetchone()
            if not attempt: raise ValueError("result attempt does not exist")
            if str(attempt[0]) == "RUNNING":
                self.conn.execute("UPDATE task_attempts SET status=?,ended_at=?,updated_at=? WHERE attempt_id=?", (result.status.value, _now(), _now(), result.attempt_id))
            elif str(attempt[0]) != result.status.value:
                raise ValueError("result status does not match terminal attempt")
            row = self.conn.execute("SELECT session_id,next_seq_no FROM runs WHERE run_id=?", (result.run_id,)).fetchone()
            if not row: raise ValueError("result run does not exist")
            payload = {"result_id": result.result_id, "status": result.status.value, "payload_hash": result.payload_hash}
            if event_payload: payload.update(event_payload)
            event = build_event(run_id=result.run_id, session_id=str(row[0]), event_type="RESULT_WRITTEN", seq_no=int(row[1]), payload=payload, plan_revision_id=result.plan_revision_id, task_id=result.task_id, attempt_id=result.attempt_id, parent_event_id=parent_event_id)
            self._append_event_locked(event)
            self.conn.execute("UPDATE runs SET next_seq_no=?,updated_at=? WHERE run_id=?", (int(row[1]) + 1, _now(), result.run_id))
            self.conn.commit()
            return event
        except Exception:
            self.conn.rollback()
            raise

    def get_result(self, result_id: str) -> Optional[dict[str, Any]]:
        row = self.conn.execute("SELECT * FROM results WHERE result_id=?", (result_id,)).fetchone()
        return dict(row) if row else None

    def get_result_for_attempt(self, attempt_id: str) -> Optional[dict[str, Any]]:
        row = self.conn.execute("SELECT * FROM results WHERE attempt_id=?", (attempt_id,)).fetchone()
        return dict(row) if row else None

    def reduce_shared_state_atomic(self, run_id: str, expected_version: int, patch: dict[str, Any], *, checkpoint_id: str, plan_revision_id: Optional[str] = None, task_transition: Optional[dict[str, Any]] = None, parent_event_id: Optional[str] = None):
        """Update run, checkpoint and CHECKPOINT/STATE_DELTA outbox atomically."""
        from agent.trace.events import build_event
        try:
            self.conn.execute("BEGIN IMMEDIATE")
            row = self.conn.execute("SELECT shared_state_json,state_version,next_seq_no,plan_revision_id,session_id FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if not row: raise KeyError(run_id)
            if int(row[1]) != expected_version:
                self.conn.rollback(); return 0, []
            before = json.loads(row[0]); status = patch.get("status"); state = dict(before); state.update({k: v for k, v in patch.items() if k != "status"})
            new_version = expected_version + 1; snapshot = canonical_json(state); snap_hash = __import__("hashlib").sha256(snapshot.encode()).hexdigest()
            cur = self.conn.execute("UPDATE runs SET shared_state_json=?,status=COALESCE(?,status),state_version=?,updated_at=? WHERE run_id=? AND state_version=?", (snapshot, status, new_version, _now(), run_id, expected_version))
            if cur.rowcount != 1: self.conn.rollback(); return 0, []
            self.conn.execute("INSERT INTO checkpoints VALUES (?,?,?,?,?,?,?,?,?)", (checkpoint_id, run_id, plan_revision_id or row[3], new_version, int(row[2]) - 1, snap_hash, snapshot, _now(), _now()))
            first_seq = int(row[2]); checkpoint = build_event(run_id=run_id, session_id=str(row[4]), event_type="CHECKPOINT", seq_no=first_seq, parent_event_id=parent_event_id, payload={"checkpoint_id": checkpoint_id, "state_version": new_version, "snapshot_hash": snap_hash}, plan_revision_id=plan_revision_id)
            delta = build_event(run_id=run_id, session_id=str(row[4]), event_type="STATE_DELTA", seq_no=first_seq + 1, parent_event_id=checkpoint.trace_id, payload={"allowed_paths": sorted(k for k in patch if k != "status"), "before_hash": __import__("hashlib").sha256(canonical_json(before).encode()).hexdigest(), "after_hash": snap_hash, "state_version": new_version}, plan_revision_id=plan_revision_id)
            events = [checkpoint, delta]
            if task_transition:
                task_event = build_event(run_id=run_id, session_id=str(row[4]), event_type="TASK_STATE_CHANGED", seq_no=first_seq + 2, parent_event_id=delta.trace_id, payload=task_transition, plan_revision_id=plan_revision_id, task_id=task_transition["task_id"])
                events.append(task_event)
            for event in events: self._append_event_locked(event)
            self.conn.execute("UPDATE runs SET next_seq_no=?,updated_at=? WHERE run_id=?", (first_seq + len(events), _now(), run_id))
            self.conn.commit(); return new_version, events
        except Exception:
            self.conn.rollback(); raise

    def outbox_empty(self, run_id: str) -> bool:
        return self.conn.execute("SELECT 1 FROM trace_outbox WHERE run_id=? AND delivered_at IS NULL LIMIT 1", (run_id,)).fetchone() is None
