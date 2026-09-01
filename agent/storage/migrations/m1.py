"""M1 SQLite schema migration, safe to apply to an isolated runtime DB.

The migration only creates canonical runtime tables.  It never opens or
modifies the source ecommerce database; callers must pass an explicitly
isolated connection/path.
"""
from __future__ import annotations

import sqlite3
from typing import Optional

M1_SCHEMA_VERSION = "m1.schema.v1"

M1_DDL = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS sessions (
  session_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, topic_version TEXT NOT NULL,
  status TEXT NOT NULL, client_session_key TEXT NOT NULL UNIQUE,
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(session_id),
  plan_revision_id TEXT REFERENCES plan_revisions(plan_revision_id),
  initial_world_hash TEXT NOT NULL CHECK(length(initial_world_hash)=64 AND initial_world_hash NOT GLOB '*[^0-9a-fA-F]*'), status TEXT NOT NULL, state_version INTEGER NOT NULL DEFAULT 0,
  next_seq_no INTEGER NOT NULL DEFAULT 1, freeze_status TEXT NOT NULL DEFAULT 'OPEN',
  shared_state_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
  client_run_key TEXT, UNIQUE(session_id, client_run_key)
);
CREATE TABLE IF NOT EXISTS plan_revisions (
  plan_revision_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id),
  schema_version TEXT NOT NULL, created_by TEXT NOT NULL, revision_reason TEXT NOT NULL,
  supersedes_plan_revision_id TEXT REFERENCES plan_revisions(plan_revision_id), version INTEGER NOT NULL,
  status TEXT NOT NULL, tasks_json TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
  UNIQUE(run_id, version)
);
CREATE TABLE IF NOT EXISTS tasks (
  task_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id),
  plan_revision_id TEXT NOT NULL REFERENCES plan_revisions(plan_revision_id), agent_ref TEXT NOT NULL,
  capability_refs_json TEXT NOT NULL, depends_on_json TEXT NOT NULL, input_bindings_json TEXT NOT NULL,
  output_contract TEXT NOT NULL, failure_strategy TEXT NOT NULL, side_effect TEXT NOT NULL,
  timeout_ms INTEGER NOT NULL, deadline TEXT, task_schema_hash TEXT NOT NULL,
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
  UNIQUE(plan_revision_id, task_id)
);
CREATE TABLE IF NOT EXISTS task_attempts (
  attempt_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id),
  plan_revision_id TEXT NOT NULL REFERENCES plan_revisions(plan_revision_id), task_id TEXT NOT NULL REFERENCES tasks(task_id),
  agent_ref TEXT NOT NULL, attempt_no INTEGER NOT NULL, status TEXT NOT NULL, input_hash TEXT NOT NULL CHECK(length(input_hash)=64 AND input_hash NOT GLOB '*[^0-9a-fA-F]*'),
  deadline TEXT, started_at TEXT, ended_at TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, UNIQUE(task_id, attempt_no)
);
CREATE TABLE IF NOT EXISTS results (
  result_id TEXT PRIMARY KEY, schema_version TEXT NOT NULL, run_id TEXT NOT NULL REFERENCES runs(run_id),
  plan_revision_id TEXT NOT NULL REFERENCES plan_revisions(plan_revision_id), task_id TEXT NOT NULL REFERENCES tasks(task_id),
  attempt_id TEXT NOT NULL UNIQUE REFERENCES task_attempts(attempt_id), status TEXT NOT NULL,
  output_contract TEXT NOT NULL, payload_json TEXT, payload_hash TEXT NOT NULL CHECK(length(payload_hash)=64 AND payload_hash NOT GLOB '*[^0-9a-fA-F]*'), business_code TEXT,
  error_ref TEXT, evidence_refs_json TEXT NOT NULL, claim_refs_json TEXT NOT NULL, usage_json TEXT NOT NULL,
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS checkpoints (
  checkpoint_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id),
  plan_revision_id TEXT REFERENCES plan_revisions(plan_revision_id), state_version INTEGER NOT NULL,
  seq_cursor INTEGER NOT NULL, snapshot_hash TEXT NOT NULL, snapshot_json TEXT NOT NULL,
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL, UNIQUE(run_id, state_version)
);
CREATE TABLE IF NOT EXISTS trace_outbox (
  event_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id), seq_no INTEGER NOT NULL,
  envelope_json TEXT NOT NULL, payload_hash TEXT NOT NULL CHECK(length(payload_hash)=64 AND payload_hash NOT GLOB '*[^0-9a-fA-F]*'), created_at TEXT NOT NULL,
  delivered_at TEXT, updated_at TEXT NOT NULL DEFAULT '', UNIQUE(run_id, seq_no)
);
CREATE TABLE IF NOT EXISTS trace_manifests (
  manifest_id TEXT PRIMARY KEY, run_id TEXT NOT NULL UNIQUE REFERENCES runs(run_id),
  trace_checksum TEXT NOT NULL, event_count INTEGER NOT NULL, version_tuple_json TEXT NOT NULL,
  snapshot_hash TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS late_event_audit (
  audit_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id), event_id TEXT NOT NULL,
  reason TEXT NOT NULL, received_at TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
  UNIQUE(run_id, event_id)
);
CREATE INDEX IF NOT EXISTS idx_runs_session_created ON runs(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_checkpoint_recovery ON checkpoints(run_id, state_version);
CREATE INDEX IF NOT EXISTS idx_attempt_recovery ON task_attempts(task_id, attempt_no);
CREATE INDEX IF NOT EXISTS idx_trace_seq ON trace_outbox(run_id, seq_no);
CREATE TRIGGER IF NOT EXISTS results_immutable_update BEFORE UPDATE ON results
BEGIN SELECT RAISE(ABORT, 'RESULT_IMMUTABLE'); END;
CREATE TRIGGER IF NOT EXISTS results_immutable_delete BEFORE DELETE ON results
BEGIN SELECT RAISE(ABORT, 'RESULT_IMMUTABLE'); END;
CREATE TRIGGER IF NOT EXISTS checkpoints_immutable_update BEFORE UPDATE ON checkpoints
BEGIN SELECT RAISE(ABORT, 'CHECKPOINT_IMMUTABLE'); END;
CREATE TRIGGER IF NOT EXISTS checkpoints_immutable_delete BEFORE DELETE ON checkpoints
BEGIN SELECT RAISE(ABORT, 'CHECKPOINT_IMMUTABLE'); END;
CREATE TRIGGER IF NOT EXISTS active_plan_immutable_update BEFORE UPDATE ON plan_revisions
WHEN OLD.status <> 'DRAFT'
BEGIN SELECT RAISE(ABORT, 'PLAN_REVISION_IMMUTABLE'); END;
CREATE TRIGGER IF NOT EXISTS plan_immutable_delete BEFORE DELETE ON plan_revisions
BEGIN SELECT RAISE(ABORT, 'PLAN_REVISION_IMMUTABLE'); END;
CREATE TRIGGER IF NOT EXISTS task_immutable_update BEFORE UPDATE ON tasks
BEGIN SELECT RAISE(ABORT, 'TASK_IMMUTABLE'); END;
CREATE TRIGGER IF NOT EXISTS task_immutable_delete BEFORE DELETE ON tasks
BEGIN SELECT RAISE(ABORT, 'TASK_IMMUTABLE'); END;
-- envelope rows are append-only; delivery metadata is updated by the recorder.
CREATE TRIGGER IF NOT EXISTS trace_outbox_immutable_delete BEFORE DELETE ON trace_outbox
BEGIN SELECT RAISE(ABORT, 'TRACE_OUTBOX_IMMUTABLE'); END;
CREATE TRIGGER IF NOT EXISTS trace_outbox_immutable_update BEFORE UPDATE ON trace_outbox
WHEN NEW.event_id <> OLD.event_id OR NEW.run_id <> OLD.run_id OR NEW.seq_no <> OLD.seq_no
  OR NEW.envelope_json <> OLD.envelope_json OR NEW.payload_hash <> OLD.payload_hash
  OR NEW.created_at <> OLD.created_at
  OR (NEW.updated_at <> OLD.updated_at AND NOT (OLD.delivered_at IS NULL AND NEW.delivered_at IS NOT NULL))
  OR NEW.delivered_at IS NULL
  OR OLD.delivered_at IS NOT NULL
BEGIN SELECT RAISE(ABORT, 'TRACE_OUTBOX_IMMUTABLE'); END;
"""


def _ddl_statements() -> list[str]:
    """Split DDL without executescript's implicit pre-commit behavior."""
    statements: list[str] = []
    current = ""
    for line in M1_DDL.splitlines():
        current += line + "\n"
        if sqlite3.complete_statement(current):
            statement = current.strip()
            if statement:
                statements.append(statement)
            current = ""
    if current.strip():
        statements.append(current.strip())
    return statements


def apply_m1_schema(conn: sqlite3.Connection, *, fail_after: Optional[int] = None) -> None:
    """Apply atomically; caller owns an isolated runtime connection."""
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        conn.execute("BEGIN")
        for index, statement in enumerate(_ddl_statements(), start=1):
            if fail_after is not None and index > fail_after:
                raise RuntimeError("injected M1 migration failure")
            conn.execute(statement)
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def schema_tables(conn: sqlite3.Connection) -> set[str]:
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
