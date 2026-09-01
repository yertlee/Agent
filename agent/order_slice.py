"""M1 read-only order vertical slice through the canonical object chain."""
from __future__ import annotations

import json
from typing import Any, Optional
from uuid import uuid4

from .domain.objects import AttemptStatus, PlanRevision, PlanStatus, Result, ResultStatus, Run, Task, TaskAttempt, TaskStatus, sha256_json
from .domain.plan_validator import PlanValidator
from .m1_runtime import Supervisor
from .storage.repositories import M1Repository
from .storage.repository import SQLiteOrderRepository
from .trace.events import build_event
from .trace.outbox import OutboxWriter

DEFAULT_RUNTIME_DB = "runtime/m1/runtime.db"


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def run_order_read_slice(*, source_db: str = "ecommerce.db", runtime_db: str = DEFAULT_RUNTIME_DB, order_id: str, phone_last4: str, user_id: str = "user_m1", session_id: Optional[str] = None) -> dict[str, Any]:
    """Execute an ownership-filtered read against source_db and persist M1 facts.

    ``source_db`` is opened read-only by the M0 repository.  All canonical
    object/event writes go to the independent ``runtime_db``.
    """
    session_id = session_id or _id("session")
    run_id, revision_id, task_id, attempt_id = (_id(x) for x in ("run", "plan", "task", "attempt"))
    runtime = M1Repository(runtime_db)
    try:
        runtime.create_session(session_id, user_id, client_session_key=session_id)
        run = Run(run_id=run_id, session_id=session_id, initial_world_hash=sha256_json({"source": "ecommerce", "version": "m1"}))
        runtime.create_run(run)
        task = Task(task_id=task_id, plan_revision_id=revision_id, agent_ref="order@v1", capability_refs=["order/read@v1"], output_contract="order.read.v1", failure_strategy="FAIL_RUN", side_effect="READ_ONLY", timeout_ms=5000)
        revision = PlanRevision(plan_revision_id=revision_id, run_id=run_id, created_by="supervisor", revision_reason="initial", version=1, status=PlanStatus.ACTIVE, tasks=[task])
        PlanValidator(agents={"order@v1"}, capabilities={"order/read@v1"}).validate(revision)
        runtime.create_plan_revision(revision)
        attempt = TaskAttempt(attempt_id=attempt_id, run_id=run_id, plan_revision_id=revision_id, task_id=task_id, agent_ref=task.agent_ref, attempt_no=1, status=AttemptStatus.CREATED, input_hash=sha256_json({"order_id": order_id, "phone_last4": phone_last4}))
        runtime.create_attempt(attempt)
        outbox = OutboxWriter(runtime)
        parent = None
        def emit(kind: str, payload: dict[str, Any], **kwargs):
            nonlocal parent
            seq = runtime.conn.execute("SELECT next_seq_no FROM runs WHERE run_id=?", (run_id,)).fetchone()[0]
            event = build_event(run_id=run_id, session_id=session_id, event_type=kind, seq_no=int(seq), payload=payload, **kwargs)
            outbox.append(event); parent = event.trace_id
        emit("RUN_CREATED", {"run_id": run_id, "scenario_id": "order_read"})
        emit("PLAN_CREATED", {"plan_revision_id": revision_id, "schema_version": revision.schema_version, "supersedes": None}, plan_revision_id=revision_id, parent_event_id=parent)
        emit("PLAN_VALIDATED", {"plan_revision_id": revision_id}, plan_revision_id=revision_id, parent_event_id=parent)
        # Supervisor owns shared-state writes; CAS, checkpoint and the two
        # corresponding outbox events commit atomically.
        supervisor = Supervisor(runtime)
        supervisor.reduce(run_id, expected_version=0, patch={"status": "READY"}, checkpoint_id=_id("checkpoint"), plan_revision_id=revision_id, parent_event_id=parent)
        supervisor.transition_task(task_id, TaskStatus.READY, expected_version=1, plan_revision_id=revision_id)
        parent = runtime.conn.execute("SELECT event_id FROM trace_outbox WHERE run_id=? ORDER BY seq_no DESC LIMIT 1", (run_id,)).fetchone()[0]
        supervisor.reduce(run_id, expected_version=2, patch={"status": "RUNNING"}, checkpoint_id=_id("checkpoint"), plan_revision_id=revision_id, parent_event_id=parent)
        supervisor.transition_task(task_id, TaskStatus.RUNNING, expected_version=3, plan_revision_id=revision_id)
        supervisor.start_attempt(attempt_id)
        parent = runtime.conn.execute("SELECT event_id FROM trace_outbox WHERE run_id=? ORDER BY seq_no DESC LIMIT 1", (run_id,)).fetchone()[0]
        emit("ATTEMPT_STARTED", {"attempt_id": attempt_id, "logical_call_no": 1, "physical_attempt_no": 1}, plan_revision_id=revision_id, task_id=task_id, attempt_id=attempt_id, parent_event_id=parent)
        with SQLiteOrderRepository(source_db) as source:
            order = source.get_order_for_owner(order_id, phone_last4)
        safe_order = None
        business_code = None
        if order:
            safe_order = {k: v for k, v in order.items() if k not in {"phone_last4", "recipient_name", "recipient_phone_mask", "address_summary"}}
            business_code = "ORDER_FOUND"
        else:
            business_code = "ORDER_NOT_FOUND"
        emit("TOOL_CALLED", {"tool_ref": "order/read@v1", "capability_ref": "order/read@v1", "safe_args_hash": sha256_json({"order_id": order_id, "phone_last4": "redacted"}), "implementation_mode": "LIVE"}, plan_revision_id=revision_id, task_id=task_id, attempt_id=attempt_id, parent_event_id=parent)
        result = Result(result_id=_id("result"), run_id=run_id, plan_revision_id=revision_id, task_id=task_id, attempt_id=attempt_id, status=ResultStatus.SUCCEEDED if order else ResultStatus.FAILED, output_contract=task.output_contract, payload=safe_order, business_code=business_code, usage={"logical_calls": 1, "physical_attempts": 1})
        emit("TOOL_RETURNED", {"result_id": result.result_id, "ok": bool(order), "error_code": None if order else "ORDER_NOT_FOUND", "output_hash": result.payload_hash}, plan_revision_id=revision_id, task_id=task_id, attempt_id=attempt_id, parent_event_id=parent)
        result_event = runtime.append_result_with_event(result, parent_event_id=parent)
        parent = result_event.trace_id
        terminal = TaskStatus.SUCCEEDED if order else TaskStatus.FAILED
        supervisor.transition_task(task_id, terminal, expected_version=4, plan_revision_id=revision_id)
        run_row = runtime.conn.execute("SELECT state_version FROM runs WHERE run_id=?", (run_id,)).fetchone()
        parent = runtime.conn.execute("SELECT event_id FROM trace_outbox WHERE run_id=? ORDER BY seq_no DESC LIMIT 1", (run_id,)).fetchone()[0]
        supervisor.reduce(run_id, expected_version=int(run_row[0]), patch={"status": terminal.value}, checkpoint_id=_id("checkpoint"), plan_revision_id=revision_id, parent_event_id=parent)
        if not order:
            emit("ERROR", {"code": "ORDER_NOT_FOUND", "layer": "business", "retryable": False, "safe_details": {}}, plan_revision_id=revision_id, task_id=task_id, attempt_id=attempt_id, parent_event_id=parent)
        trace_rows = runtime.conn.execute("SELECT envelope_json FROM trace_outbox WHERE run_id=? ORDER BY seq_no", (run_id,)).fetchall()
        trace_types = [json.loads(row[0])["event_type"] for row in trace_rows]
        terminal_run = dict(runtime.conn.execute("SELECT run_id,status,state_version,shared_state_json FROM runs WHERE run_id=?", (run_id,)).fetchone())
        terminal_attempt = dict(runtime.conn.execute("SELECT attempt_id,status,ended_at FROM task_attempts WHERE attempt_id=?", (attempt_id,)).fetchone())
        task_status = json.loads(terminal_run["shared_state_json"]).get("task_statuses", {}).get(task_id)
        return {"run": run, "plan_revision": revision, "task": task, "attempt": attempt, "result": result, "terminal_run": terminal_run, "terminal_task_status": task_status, "terminal_attempt": terminal_attempt, "trace_event_count": len(trace_rows), "trace_event_types": trace_types, "business_code": business_code}
    finally:
        runtime.close()
