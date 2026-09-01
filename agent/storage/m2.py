"""M2 repository over the M1 isolated runtime database."""
from __future__ import annotations

from typing import Any
from uuid import uuid4

from agent.domain.objects import canonical_json
from agent.m2_registry import ToolSpec
from .migrations.m2 import M2_SCHEMA_VERSION, apply_m2_schema
from .repositories import M1Repository, _now


class M2Repository(M1Repository):
    def __init__(self, db_path: str = ":memory:", *, initialize: bool = True):
        super().__init__(db_path, initialize=initialize)
        if initialize:
            apply_m2_schema(self.conn)

    def register_tool(self, spec: ToolSpec) -> None:
        self.conn.execute(
            "INSERT INTO registry_entries VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (spec.tool_ref, spec.capability_ref, spec.owner, "m2.tool.v1", spec.args_schema, spec.result_schema, spec.risk, spec.implementation_mode, spec.side_effect, spec.timeout_ms, canonical_json(list(spec.allowed_error_codes)), _now(), _now()),
        )
        self.conn.commit()

    def record_eligibility(self, fact: Any, *, run_id: str, task_id: str) -> None:
        self.conn.execute(
            "INSERT INTO eligibility_facts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (fact.fact_id, run_id, task_id, fact.entity_id, fact.service, fact.decision, fact.rule_id, fact.reason, fact.policy_version, fact.snapshot_hash, canonical_json(fact), _now(), _now()),
        )
        self.conn.commit()

    def _append_m2_event_locked(self, event) -> None:
        row = self.conn.execute("SELECT next_seq_no FROM runs WHERE run_id=?", (event.run_id,)).fetchone()
        if not row or int(row[0]) != event.seq_no:
            raise ValueError("event seq allocator mismatch")
        self._append_event_locked(event)
        self.conn.execute("UPDATE runs SET next_seq_no=?,updated_at=? WHERE run_id=?", (event.seq_no + 1, _now(), event.run_id))

    def append_m2_event(self, event) -> None:
        try:
            self.conn.execute("BEGIN IMMEDIATE")
            self._append_m2_event_locked(event)
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def log_audit_locked(self, *, run_id: str, actor: str, action: str, reason: str, trace_id: str | None = None, before_hash: str | None = None, after_hash: str | None = None) -> None:
        self.conn.execute(
            "INSERT INTO m2_audit_log VALUES (?,?,?,?,?,?,?,?,?,?)",
            (f"audit_{uuid4().hex}", run_id, actor, action, before_hash, after_hash, reason, trace_id, _now(), _now()),
        )


__all__ = ["M2Repository", "M2_SCHEMA_VERSION", "apply_m2_schema"]
