"""Atomic M2 schema migration for Registry, policy, token and write safety."""
from __future__ import annotations

import sqlite3
from typing import Optional

M2_SCHEMA_VERSION = "m2.schema.v1"
M2_DDL = "\n".join([
    "CREATE TABLE IF NOT EXISTS registry_entries (tool_ref TEXT PRIMARY KEY, capability_ref TEXT NOT NULL, owner TEXT NOT NULL, schema_version TEXT NOT NULL, args_schema TEXT NOT NULL, result_schema TEXT NOT NULL, risk TEXT NOT NULL, mode TEXT NOT NULL, side_effect TEXT NOT NULL, timeout_ms INTEGER NOT NULL, allowed_error_codes_json TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS policy_rules (rule_id TEXT PRIMARY KEY, version TEXT NOT NULL, decision_logic TEXT NOT NULL, source TEXT NOT NULL, effective_from TEXT NOT NULL, effective_to TEXT, scope TEXT NOT NULL, priority INTEGER NOT NULL, supersedes TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS eligibility_facts (fact_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id), task_id TEXT NOT NULL REFERENCES tasks(task_id), entity_id TEXT NOT NULL, service TEXT NOT NULL, decision TEXT NOT NULL, rule_id TEXT NOT NULL, reason TEXT NOT NULL, policy_version TEXT NOT NULL, snapshot_hash TEXT NOT NULL, fact_json TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS confirm_tokens (token_id TEXT PRIMARY KEY, token_hash TEXT NOT NULL UNIQUE, session_id TEXT NOT NULL REFERENCES sessions(session_id), user_id TEXT NOT NULL, run_id TEXT NOT NULL REFERENCES runs(run_id), task_id TEXT NOT NULL REFERENCES tasks(task_id), order_id TEXT NOT NULL, service TEXT NOT NULL, amount TEXT NOT NULL, payload_hash TEXT NOT NULL, topic_version TEXT NOT NULL, issued_at TEXT NOT NULL, expires_at TEXT NOT NULL, status TEXT NOT NULL, consumed_at TEXT, revoked_at TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS aftersales_cases (case_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, order_id TEXT NOT NULL, service TEXT NOT NULL, status TEXT NOT NULL, state_version INTEGER NOT NULL, run_id TEXT NOT NULL REFERENCES runs(run_id), session_id TEXT NOT NULL REFERENCES sessions(session_id), task_id TEXT NOT NULL REFERENCES tasks(task_id), idempotency_key TEXT NOT NULL, request_fingerprint TEXT NOT NULL, reason TEXT NOT NULL, amount TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)",
    "CREATE UNIQUE INDEX IF NOT EXISTS ux_aftersales_active_user_order ON aftersales_cases(user_id, order_id) WHERE status IN ('REQUESTED','UNDER_REVIEW','APPROVED','RETURN_PENDING','RETURNED','REFUND_PENDING','EXCHANGE_PENDING','HUMAN_REVIEW')",
    "CREATE TABLE IF NOT EXISTS review_tickets (ticket_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id), task_id TEXT NOT NULL REFERENCES tasks(task_id), idempotency_key TEXT NOT NULL UNIQUE, risk TEXT NOT NULL, reason TEXT NOT NULL, pending_action_hash TEXT NOT NULL, status TEXT NOT NULL, sla_deadline TEXT NOT NULL, decision TEXT, decision_by TEXT, decision_at TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS tool_submit_log (submit_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, run_id TEXT NOT NULL REFERENCES runs(run_id), task_id TEXT NOT NULL REFERENCES tasks(task_id), tool_ref TEXT NOT NULL, request_fingerprint TEXT NOT NULL, idempotency_key TEXT NOT NULL UNIQUE, result_id TEXT REFERENCES results(result_id), status TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, UNIQUE(user_id, tool_ref, request_fingerprint))",
    "CREATE TABLE IF NOT EXISTS m2_audit_log (audit_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id), actor TEXT NOT NULL, action TEXT NOT NULL, before_hash TEXT, after_hash TEXT, reason TEXT NOT NULL, trace_id TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)",
    "CREATE INDEX IF NOT EXISTS idx_m2_token_lookup ON confirm_tokens(token_hash, status)",
    "CREATE INDEX IF NOT EXISTS idx_m2_case_user_order ON aftersales_cases(user_id, order_id)",
    "CREATE INDEX IF NOT EXISTS idx_m2_submit_fingerprint ON tool_submit_log(tool_ref, request_fingerprint)",
])


def apply_m2_schema(conn: sqlite3.Connection, *, fail_after: Optional[int] = None) -> None:
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        conn.execute("BEGIN")
        for index, statement in enumerate(M2_DDL.splitlines(), start=1):
            if fail_after is not None and index > fail_after:
                raise RuntimeError("injected M2 migration failure")
            conn.execute(statement)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
