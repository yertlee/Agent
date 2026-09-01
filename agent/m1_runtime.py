"""Supervisor single-writer reducer and legal lifecycle transitions."""
from __future__ import annotations

from typing import Any, Mapping, Optional

from .domain.objects import RunStatus, TaskStatus
from .storage.repositories import M1Repository

RUN_TRANSITIONS = {
    "CREATED": {"READY", "CANCELLED", "FAILED"},
    "READY": {"RUNNING", "CANCELLED", "FAILED"},
    "RUNNING": {"RETRY_WAIT", "WAITING_USER", "WAITING_HUMAN", "BLOCKED", "PARTIAL", "SUCCEEDED", "FAILED", "CANCELLED"},
    "RETRY_WAIT": {"RUNNING", "BLOCKED", "FAILED", "CANCELLED"},
    "WAITING_USER": {"READY", "CANCELLED", "FAILED"},
    "WAITING_HUMAN": {"READY", "CANCELLED", "FAILED"},
    "BLOCKED": {"READY", "FAILED", "CANCELLED"},
    "PARTIAL": {"SUCCEEDED", "FAILED", "CANCELLED"},
    "SUCCEEDED": set(), "FAILED": set(), "CANCELLED": set(),
}
TASK_TRANSITIONS = {
    "CREATED": {"READY", "CANCELLED", "BLOCKED"}, "READY": {"RUNNING", "CANCELLED", "BLOCKED"},
    "RUNNING": {"RETRY_WAIT", "WAITING_USER", "WAITING_HUMAN", "BLOCKED", "SUCCEEDED", "FAILED", "CANCELLED"},
    "RETRY_WAIT": {"RUNNING", "BLOCKED", "FAILED", "CANCELLED"},
    "WAITING_USER": {"READY", "CANCELLED", "FAILED"}, "WAITING_HUMAN": {"READY", "CANCELLED", "FAILED"},
    "BLOCKED": {"READY", "FAILED", "CANCELLED"}, "SUCCEEDED": set(), "FAILED": set(), "CANCELLED": set(),
}


class InvalidTransition(ValueError): pass
class CASConflict(RuntimeError): pass


class Supervisor:
    """Only public writer for shared run state.

    Attempts/results are private records; callers cannot pass arbitrary shared
    fields through attempt APIs.  Every reducer write requires an expected CAS
    version and creates an immutable checkpoint in the same transaction.
    """
    def __init__(self, repository: M1Repository): self.repository = repository

    def reduce(self, run_id: str, *, expected_version: int, patch: Mapping[str, Any], checkpoint_id: str, plan_revision_id: Optional[str] = None, parent_event_id: Optional[str] = None) -> int:
        allowed = {"status", "shared_state"}
        if set(patch) - allowed:
            raise ValueError("shared reducer received unknown field")
        if "status" in patch:
            row = self.repository.conn.execute("SELECT status FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if not row: raise KeyError(run_id)
            current, target = str(row[0]), str(patch["status"])
            if target not in RUN_TRANSITIONS.get(current, set()):
                raise InvalidTransition(f"run transition {current}->{target} is not allowed")
        values = dict(patch)
        if "shared_state" in values:
            state = dict(values.pop("shared_state") or {})
            values.update(state)
        new_version, _events = self.repository.reduce_shared_state_atomic(run_id, expected_version, values, checkpoint_id=checkpoint_id, plan_revision_id=plan_revision_id, parent_event_id=parent_event_id)
        if new_version == 0: raise CASConflict(f"state_version conflict for run {run_id}")
        return new_version

    def transition_task(self, task_id: str, target: TaskStatus, *, expected_current: Optional[TaskStatus] = None, expected_version: Optional[int] = None, checkpoint_id: Optional[str] = None, run_id: Optional[str] = None, plan_revision_id: Optional[str] = None) -> int:
        row = self.repository.conn.execute("SELECT run_id,plan_revision_id FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        if not row: raise KeyError(task_id)
        run_id = run_id or str(row[0]); plan_revision_id = plan_revision_id or str(row[1])
        run = self.repository.conn.execute("SELECT state_version,shared_state_json FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if not run: raise KeyError(run_id)
        state = __import__("json").loads(run[1]); statuses = dict(state.get("task_statuses") or {})
        current = expected_current.value if expected_current else str(statuses.get(task_id, TaskStatus.CREATED.value))
        if target.value not in TASK_TRANSITIONS.get(current, set()):
            raise InvalidTransition(f"task transition {current}->{target.value} is not allowed")
        version = int(run[0]) if expected_version is None else expected_version
        new_version, _ = self.repository.reduce_shared_state_atomic(run_id, version, {"task_statuses": {**statuses, task_id: target.value}}, checkpoint_id=checkpoint_id or f"checkpoint_task_{task_id}_{version + 1}", plan_revision_id=plan_revision_id, task_transition={"task_id": task_id, "from": current, "to": target.value, "reason": "supervisor_reducer", "state_version": version + 1}, parent_event_id=self._last_event_id(run_id))
        if new_version == 0: raise CASConflict(f"state_version conflict for run {run_id}")
        return new_version

    def _last_event_id(self, run_id: str) -> Optional[str]:
        row = self.repository.conn.execute("SELECT event_id FROM trace_outbox WHERE run_id=? ORDER BY seq_no DESC LIMIT 1", (run_id,)).fetchone()
        return str(row[0]) if row else None

    def start_attempt(self, attempt_id: str) -> None:
        self._set_attempt_status(attempt_id, "RUNNING", started=True)

    def complete_attempt(self, attempt_id: str, status: str) -> None:
        if status not in {"SUCCEEDED", "FAILED", "BLOCKED", "CANCELLED"}:
            raise InvalidTransition(f"invalid terminal attempt status: {status}")
        self._set_attempt_status(attempt_id, status, started=False)

    def _set_attempt_status(self, attempt_id: str, status: str, *, started: bool) -> None:
        row = self.repository.conn.execute("SELECT status FROM task_attempts WHERE attempt_id=?", (attempt_id,)).fetchone()
        if not row: raise KeyError(attempt_id)
        current = str(row[0])
        if started and current != "CREATED": raise InvalidTransition(f"attempt transition {current}->RUNNING is not allowed")
        if not started and (current != "RUNNING" or status not in {"SUCCEEDED", "FAILED", "BLOCKED", "CANCELLED"}): raise InvalidTransition(f"attempt transition {current}->{status} is not allowed")
        now = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
        if started:
            self.repository.conn.execute("UPDATE task_attempts SET status=?,started_at=?,updated_at=? WHERE attempt_id=?", (status, now, now, attempt_id))
        else:
            self.repository.conn.execute("UPDATE task_attempts SET status=?,ended_at=?,updated_at=? WHERE attempt_id=?", (status, now, now, attempt_id))
        self.repository.conn.commit()

    def attempt_patch_shared_state(self, *_args, **_kwargs):
        raise PermissionError("TaskAttempt cannot directly mutate shared state")


class StateMachine:
    """Small pure transition checker reusable by legacy adapters."""
    @staticmethod
    def can_transition(current: str, target: str, *, scope: str = "run") -> bool:
        table = TASK_TRANSITIONS if scope == "task" else RUN_TRANSITIONS
        return target in table.get(current, set())
