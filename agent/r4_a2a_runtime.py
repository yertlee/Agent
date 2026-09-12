"""R4-A read-only A2A runtime.

This module is intentionally a small vertical slice.  It owns a separate
message ledger, deterministic envelope/result verification, and three read
only Specialist adapters.  It does not create evaluation data, call an
external model, or write the user order database.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Protocol

from pydantic import BaseModel, ConfigDict, Field

from .agents.ports import LogisticsAgent, OrderAgent, PolicyAgent, TypedAgentResult
from .domain.objects import Result, ResultStatus, Usage, sha256_json
from .m2_context import InvocationContext
from .m2_errors import ErrorCatalog
from .m2_registry import Registry, ToolSpec
from .m3_registry import M3Registry
from .r2_order_logistics import OrderLogisticsRepository
from .r4_a2a_contracts import (
    A2AContractError,
    A2AErrorEnvelopeV1,
    A2AMessageEnvelopeV1,
    request_fingerprint,
)
from .storage.repository import SQLiteOrderRepository


COORDINATOR_REFS = frozenset({"supervisor@v1", "router@v1", "runtime@v1"})
TRUSTED_PAYLOAD_KEYS = frozenset(
    {
        "session_id",
        "user_id",
        "run_id",
        "plan_revision_id",
        "task_id",
        "attempt_id",
        "agent_ref",
        "auth_scope",
        "idempotency_key",
        "deadline",
        "cancellation",
        "confirm_token",
        "trace_id",
    }
)
RETRYABLE_A2A_CODES = frozenset(
    {"INFRA_TIMEOUT", "INFRA_UNAVAILABLE", "INFRA_RATE_LIMITED", "TOOL_EXECUTION_FAILED"}
)
PHYSICAL_RETRY_FAULTS = {
    "PHYSICAL_TIMEOUT": "INFRA_TIMEOUT",
    "PHYSICAL_UNAVAILABLE": "INFRA_UNAVAILABLE",
    "PHYSICAL_FAILURE": "TOOL_EXECUTION_FAILED",
}
# A message is still live until it reaches one of these durable states.  The
# set is shared by freeze, reservation/duplicate handling, and rehydration so
# each path observes the same lifecycle boundary.
A2A_TERMINAL_MESSAGE_STATUSES = frozenset(
    {"SUCCEEDED", "FAILED", "BLOCKED", "REJECTED", "CANCELLED", "LATE", "CORRUPT"}
)


@dataclass(frozen=True)
class _Route:
    capability_ref: str
    agent_ref: str
    tool_ref: str
    output_contract: str
    args_schema: str


ROUTES: dict[str, _Route] = {
    "order/read@v1": _Route(
        "order/read@v1", "order-agent@v1", "order/get_info@v1", "order.result.v1", "order.read.v1"
    ),
    "logistics/read@v1": _Route(
        "logistics/read@v1", "logistics-agent@v1", "logistics/query@v1", "logistics.result.v1", "logistics.query.v1"
    ),
    "policy/read@v1": _Route(
        "policy/read@v1", "policy-agent@v1", "policy/search@v1", "policy.result.v1", "policy.query.v1"
    ),
}


class R4TraceEventV1(BaseModel):
    """Safe message/attempt trace row; payload is already redacted."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    seq_no: int = Field(gt=0)
    event_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    trace_id: str = Field(min_length=1)
    event_type: str = Field(min_length=1)
    message_id: Optional[str] = None
    attempt_id: Optional[str] = None
    actor: str = Field(min_length=1)
    scene_clock: datetime
    payload: dict[str, Any] = Field(default_factory=dict)
    payload_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class A2ADispatchResult:
    status: str
    request: A2AMessageEnvelopeV1
    response: A2AMessageEnvelopeV1 | None = None
    specialist_result: TypedAgentResult | None = None
    canonical_result: Result | None = None
    error_code: str | None = None
    attempt_count: int = 0
    physical_call_count: int = 0
    duplicate: bool = False
    pending: bool = False
    blocked: bool = False
    late: bool = False
    duplicate_of: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "SUCCEEDED"


@dataclass(frozen=True)
class R4RunResult:
    run_id: str
    plan_revision_id: str
    topology: str
    status: str
    answer: str
    dispatches: Mapping[str, A2ADispatchResult]
    canonical_results: Mapping[str, Result]
    trace: tuple[R4TraceEventV1, ...]


class R4SpecialistAdapter(Protocol):
    agent_ref: str
    capability_ref: str

    def invoke(self, payload: Mapping[str, Any], *, context: InvocationContext) -> TypedAgentResult: ...


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _normalized_scene_clock(value: datetime | str) -> str:
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return _utc(parsed).isoformat()


def project_trace_event(
    *,
    seq_no: int,
    run_id: str,
    trace_id: str,
    event_type: str,
    message_id: str | None,
    attempt_id: str | None,
    actor: str,
    scene_clock: datetime | str,
    payload_hash: str,
) -> dict[str, Any]:
    """Return the canonical identity projection for one trace event."""

    return {
        "seq_no": int(seq_no),
        "run_id": str(run_id),
        "trace_id": str(trace_id),
        "event_type": str(event_type),
        "message_id": None if message_id is None else str(message_id),
        "attempt_id": None if attempt_id is None else str(attempt_id),
        "actor": str(actor),
        "scene_clock": _normalized_scene_clock(scene_clock),
        "payload_hash": str(payload_hash),
    }


def project_trace_event_id(**projection: Any) -> str:
    """Derive a deterministic event id from a canonical event projection."""

    return f"event_{sha256_json(project_trace_event(**projection))}"


def _redact(value: Any, *, key: str = "") -> Any:
    """Redact raw identity/secrets before a value enters ledger/trace JSON."""

    lowered = str(key).lower()
    safe_reference = lowered.endswith(("_hash", "_ref")) or lowered in {"snapshot_hash", "payload_hash"}
    if lowered in {"phone", "phone_last4", "recipient_phone", "address", "secret", "api_key", "token", "password"} and not safe_reference:
        return "<redacted>"
    if isinstance(value, BaseModel):
        return _redact(value.model_dump(mode="python"), key=key)
    if isinstance(value, Mapping):
        return {str(k): _redact(v, key=str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact(item, key=key) for item in value]
    if isinstance(value, datetime):
        return _utc(value).isoformat().replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    return value


@dataclass(frozen=True)
class PolicyAuthoritySnapshot:
    """Immutable authority projection used by the R4 policy adapter.

    The snapshot is derived from a verified R3/R3.5 manifest for a real
    retriever, or must be supplied explicitly by a test reader.  Evidence is
    represented as JSON-safe records so the same projection can be hashed,
    traced, and compared after a runtime restart.
    """

    source: str
    source_version: str
    version_tuple: tuple[str, ...]
    strategy_checksum: str
    evidence: tuple[Mapping[str, str], ...]
    snapshot_hash: str

    @property
    def evidence_by_id(self) -> dict[str, Mapping[str, str]]:
        return {str(item["evidence_id"]): item for item in self.evidence}


def _authority_projection(
    *,
    source: str,
    source_version: str,
    version_tuple: tuple[str, ...],
    strategy_checksum: str,
    evidence: tuple[Mapping[str, str], ...],
) -> dict[str, Any]:
    return {
        "source": source,
        "source_version": source_version,
        "version_tuple": list(version_tuple),
        "strategy_checksum": strategy_checksum,
        "evidence": [dict(item) for item in evidence],
    }


def _authority_from_mapping(value: Mapping[str, Any] | PolicyAuthoritySnapshot) -> PolicyAuthoritySnapshot:
    if isinstance(value, PolicyAuthoritySnapshot):
        return value
    if not isinstance(value, Mapping):
        raise A2AContractError("A2A_POLICY_AUTHORITY_INVALID")
    source = str(value.get("source") or "")
    if source != "r3.5.project-authored-kb":
        raise A2AContractError("A2A_POLICY_SOURCE_UNAUTHORIZED")
    raw_tuple = value.get("version_tuple")
    if not isinstance(raw_tuple, (list, tuple)) or not raw_tuple or any(not str(item) for item in raw_tuple):
        raise A2AContractError("A2A_POLICY_VERSION_UNAUTHORIZED")
    version_tuple = tuple(str(item) for item in raw_tuple)
    source_version = str(value.get("source_version") or "")
    if source_version != sha256_json(version_tuple):
        raise A2AContractError("A2A_POLICY_VERSION_UNAUTHORIZED")
    strategy_checksum = str(value.get("strategy_checksum") or "")
    if not strategy_checksum or len(strategy_checksum) != 64 or any(ch not in "0123456789abcdef" for ch in strategy_checksum.lower()):
        raise A2AContractError("A2A_POLICY_STRATEGY_CHECKSUM_INVALID")
    raw_evidence = value.get("evidence")
    if isinstance(raw_evidence, Mapping):
        raw_evidence = list(raw_evidence.values())
    if not isinstance(raw_evidence, (list, tuple)) or not raw_evidence:
        raise A2AContractError("A2A_POLICY_AUTHORITY_EVIDENCE_INVALID")
    evidence: list[Mapping[str, str]] = []
    required = ("evidence_id", "source_id", "version", "chunk_id", "text_hash", "locator")
    for item in raw_evidence:
        if not isinstance(item, Mapping) or not all(str(item.get(key) or "") for key in required):
            raise A2AContractError("A2A_POLICY_AUTHORITY_EVIDENCE_INVALID")
        record = {key: str(item[key]) for key in required}
        if len(record["text_hash"]) != 64 or any(ch not in "0123456789abcdef" for ch in record["text_hash"].lower()):
            raise A2AContractError("A2A_POLICY_AUTHORITY_EVIDENCE_INVALID")
        evidence.append(record)
    evidence_tuple = tuple(sorted(evidence, key=lambda item: item["evidence_id"]))
    projection = _authority_projection(
        source=source,
        source_version=source_version,
        version_tuple=version_tuple,
        strategy_checksum=strategy_checksum,
        evidence=evidence_tuple,
    )
    snapshot_hash = sha256_json(projection)
    supplied_hash = value.get("snapshot_hash")
    if supplied_hash is not None and str(supplied_hash) != snapshot_hash:
        raise A2AContractError("A2A_POLICY_AUTHORITY_HASH_MISMATCH")
    return PolicyAuthoritySnapshot(
        source=source,
        source_version=source_version,
        version_tuple=version_tuple,
        strategy_checksum=strategy_checksum,
        evidence=evidence_tuple,
        snapshot_hash=snapshot_hash,
    )


def _authority_from_retriever(retriever: Any) -> PolicyAuthoritySnapshot:
    manifest = getattr(retriever, "manifest", None)
    if manifest is None:
        raise A2AContractError("A2A_POLICY_AUTHORITY_REQUIRED")
    version_tuple = tuple(str(item) for item in getattr(manifest, "version_tuple", ()))
    strategy_checksum = str(getattr(manifest, "strategy_checksum", ""))
    records = []
    for chunk in getattr(manifest, "chunks", ()):
        records.append(
            {
                "evidence_id": f"ev-{chunk.chunk_id}",
                "source_id": str(chunk.source_id),
                "version": str(chunk.version),
                "chunk_id": str(chunk.chunk_id),
                "text_hash": str(chunk.text_hash),
                "locator": str(chunk.locator),
            }
        )
    return _authority_from_mapping(
        {
            "source": "r3.5.project-authored-kb",
            "source_version": sha256_json(version_tuple),
            "version_tuple": list(version_tuple),
            "strategy_checksum": strategy_checksum,
            "evidence": records,
        }
    )


class A2AMessageLedger:
    """SQLite message owner isolated from ``ecommerce.db`` and M2 tables."""

    SCHEMA = """
    PRAGMA foreign_keys = ON;
    CREATE TABLE IF NOT EXISTS a2a_runs (
      run_id TEXT PRIMARY KEY,
      plan_revision_id TEXT NOT NULL,
      status TEXT NOT NULL,
      live_event_count INTEGER,
      live_head TEXT,
      live_checksum TEXT,
      freeze_checksum TEXT,
      freeze_event_count INTEGER,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS a2a_superseded_revisions (
      run_id TEXT NOT NULL,
      plan_revision_id TEXT NOT NULL,
      superseded_by_plan_revision_id TEXT,
      created_at TEXT NOT NULL,
      PRIMARY KEY(run_id, plan_revision_id)
    );
    CREATE TABLE IF NOT EXISTS a2a_messages (
      message_id TEXT PRIMARY KEY,
      idempotency_key TEXT NOT NULL UNIQUE,
      request_fingerprint TEXT NOT NULL,
      request_checksum TEXT NOT NULL,
      message_kind TEXT,
      schema_version TEXT,
      attempt_id TEXT,
      trace_id TEXT,
      envelope_created_at TEXT,
      envelope_deadline TEXT,
      run_id TEXT NOT NULL,
      plan_revision_id TEXT NOT NULL,
      task_id TEXT NOT NULL,
      correlation_id TEXT NOT NULL,
      parent_message_id TEXT,
      dependency_message_ids_json TEXT NOT NULL,
      sender_ref TEXT NOT NULL,
      receiver_ref TEXT NOT NULL,
      capability_ref TEXT NOT NULL,
      payload_hash TEXT NOT NULL,
      request_payload_hash TEXT,
      status TEXT NOT NULL,
      request_json TEXT NOT NULL,
      response_json TEXT,
      response_checksum TEXT,
      response_message_id TEXT,
      response_message_kind TEXT,
      response_schema_version TEXT,
      response_run_id TEXT,
      response_plan_revision_id TEXT,
      response_task_id TEXT,
      response_correlation_id TEXT,
      response_parent_message_id TEXT,
      response_trace_id TEXT,
      response_attempt_id TEXT,
      response_sender_ref TEXT,
      response_receiver_ref TEXT,
      response_capability_ref TEXT,
      response_idempotency_key TEXT,
      response_created_at TEXT,
      response_deadline TEXT,
      response_payload_hash TEXT,
      specialist_json TEXT,
      specialist_checksum TEXT,
      canonical_json TEXT,
      canonical_checksum TEXT,
      duplicate_of TEXT,
      error_code TEXT,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS a2a_attempts (
      attempt_id TEXT PRIMARY KEY,
      message_id TEXT NOT NULL REFERENCES a2a_messages(message_id),
      attempt_no INTEGER NOT NULL,
      status TEXT NOT NULL,
      error_code TEXT,
      physical_call INTEGER NOT NULL DEFAULT 0,
      started_at TEXT NOT NULL,
      scene_clock TEXT,
      ended_at TEXT,
      UNIQUE(message_id, attempt_no)
    );
    CREATE TABLE IF NOT EXISTS a2a_trace (
      run_id TEXT NOT NULL,
      seq_no INTEGER NOT NULL,
      event_id TEXT NOT NULL UNIQUE,
      trace_id TEXT NOT NULL,
      event_type TEXT NOT NULL,
      message_id TEXT,
      attempt_id TEXT,
      actor TEXT NOT NULL,
      scene_clock TEXT NOT NULL,
      payload_json TEXT NOT NULL,
      payload_hash TEXT NOT NULL,
      PRIMARY KEY(run_id, seq_no)
    );
    CREATE TABLE IF NOT EXISTS a2a_late_events (
      audit_id TEXT PRIMARY KEY,
      run_id TEXT NOT NULL,
      message_id TEXT NOT NULL,
      reason TEXT NOT NULL,
      trace_id TEXT,
      payload_json TEXT,
      payload_hash TEXT,
      received_at TEXT NOT NULL
    );
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(self.SCHEMA)
        self._ensure_columns()
        self._lock = threading.RLock()

    def _ensure_columns(self) -> None:
        """Keep an already-created isolated ledger forward compatible."""

        required = {
            "a2a_runs": {
                "live_event_count": "INTEGER",
                "live_head": "TEXT",
                "live_checksum": "TEXT",
                "freeze_checksum": "TEXT",
                "freeze_event_count": "INTEGER",
            },
            "a2a_messages": {
                "request_checksum": "TEXT",
                "message_kind": "TEXT",
                "schema_version": "TEXT",
                "attempt_id": "TEXT",
                "trace_id": "TEXT",
                "envelope_created_at": "TEXT",
                "envelope_deadline": "TEXT",
                "correlation_id": "TEXT",
                "parent_message_id": "TEXT",
                "dependency_message_ids_json": "TEXT",
                "sender_ref": "TEXT",
                "receiver_ref": "TEXT",
                "capability_ref": "TEXT",
                "payload_hash": "TEXT",
                "request_payload_hash": "TEXT",
                "response_checksum": "TEXT",
                "response_message_id": "TEXT",
                "response_message_kind": "TEXT",
                "response_schema_version": "TEXT",
                "response_run_id": "TEXT",
                "response_plan_revision_id": "TEXT",
                "response_task_id": "TEXT",
                "response_correlation_id": "TEXT",
                "response_parent_message_id": "TEXT",
                "response_trace_id": "TEXT",
                "response_attempt_id": "TEXT",
                "response_sender_ref": "TEXT",
                "response_receiver_ref": "TEXT",
                "response_capability_ref": "TEXT",
                "response_idempotency_key": "TEXT",
                "response_created_at": "TEXT",
                "response_deadline": "TEXT",
                "response_payload_hash": "TEXT",
                "specialist_json": "TEXT",
                "specialist_checksum": "TEXT",
                "canonical_json": "TEXT",
                "canonical_checksum": "TEXT",
                "duplicate_of": "TEXT",
            },
            "a2a_attempts": {"scene_clock": "TEXT"},
            "a2a_late_events": {"trace_id": "TEXT", "payload_json": "TEXT", "payload_hash": "TEXT"},
        }
        for table, columns in required.items():
            existing = {str(row[1]) for row in self.conn.execute(f"PRAGMA table_info({table})").fetchall()}
            for name, column_type in columns.items():
                if name not in existing:
                    self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {column_type}")
        empty_checksum = sha256_json([])
        self.conn.execute(
            "UPDATE a2a_runs SET live_event_count=0,live_head='',live_checksum=? WHERE live_event_count IS NULL AND NOT EXISTS (SELECT 1 FROM a2a_trace WHERE a2a_trace.run_id=a2a_runs.run_id)",
            (empty_checksum,),
        )
        self.conn.execute(
            "UPDATE a2a_runs SET live_event_count=-1,live_head='__UNVERIFIED__',live_checksum='' WHERE live_event_count IS NULL",
        )

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def close(self) -> None:
        self.conn.close()

    def ensure_run(self, run_id: str, plan_revision_id: str, *, status: str = "OPEN") -> None:
        now = self._now()
        with self._lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                row = self.conn.execute("SELECT plan_revision_id,status FROM a2a_runs WHERE run_id=?", (run_id,)).fetchone()
                if row is None:
                    self.conn.execute(
                        "INSERT INTO a2a_runs(run_id,plan_revision_id,status,live_event_count,live_head,live_checksum,freeze_checksum,freeze_event_count,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (run_id, plan_revision_id, status, 0, "", sha256_json([]), None, None, now, now),
                    )
                else:
                    if str(row["status"]) in {"FROZEN", "CANCELLED"}:
                        raise A2AContractError("A2A_RUN_TERMINAL")
                    if str(row["plan_revision_id"]) != plan_revision_id:
                        raise A2AContractError("A2A_RUN_PLAN_BINDING_MISMATCH")
                    if self.conn.execute(
                        "SELECT 1 FROM a2a_superseded_revisions WHERE run_id=? AND plan_revision_id=?",
                        (run_id, plan_revision_id),
                    ).fetchone() is not None:
                        raise A2AContractError("A2A_REVISION_SUPERSEDED")
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise

    def run_status(self, run_id: str) -> str | None:
        with self._lock:
            row = self.conn.execute("SELECT status FROM a2a_runs WHERE run_id=?", (run_id,)).fetchone()
        return str(row[0]) if row else None

    def run_plan(self, run_id: str) -> str | None:
        with self._lock:
            row = self.conn.execute("SELECT plan_revision_id FROM a2a_runs WHERE run_id=?", (run_id,)).fetchone()
        return str(row[0]) if row else None

    def set_run_status(self, run_id: str, status: str) -> None:
        with self._lock:
            self.conn.execute("UPDATE a2a_runs SET status=?,updated_at=? WHERE run_id=?", (status, self._now(), run_id))

    def _converge_obsolete_messages_locked(
        self,
        *,
        run_id: str,
        reason: str,
        error_code: str,
        plan_revision_id: str | None = None,
    ) -> None:
        """Close messages that cannot start after a durable lifecycle change.

        The caller owns a ``BEGIN IMMEDIATE`` transaction.  Messages already
        in an adapter attempt remain live until ``finalize_terminal`` performs
        its lifecycle recheck; pre-dispatch states are closed here so a plan
        switch or cancellation cannot leave a permanently executable row.
        """

        clauses = ["run_id=?", "status IN ('REGISTERED','PENDING','DISPATCHED')"]
        params: list[Any] = [run_id]
        if plan_revision_id is not None:
            clauses.append("plan_revision_id=?")
            params.append(plan_revision_id)
        rows = self.conn.execute(
            f"SELECT message_id,status,trace_id,payload_hash FROM a2a_messages WHERE {' AND '.join(clauses)} ORDER BY created_at,message_id",
            tuple(params),
        ).fetchall()
        now = self._now()
        received_at = datetime.now(timezone.utc)
        for row in rows:
            self.conn.execute(
                "UPDATE a2a_messages SET status='LATE',error_code=?,updated_at=? WHERE message_id=? AND status IN ('REGISTERED','PENDING','DISPATCHED')",
                (error_code, now, str(row["message_id"])),
            )
            self._record_late_locked(
                run_id=run_id,
                message_id=str(row["message_id"]),
                reason=reason,
                received_at=received_at,
                trace_id=row["trace_id"],
                payload={
                    "reason": reason,
                    "previous_status": str(row["status"]),
                    "message_hash": row["payload_hash"],
                },
            )

    def switch_plan(self, run_id: str, old_plan_revision_id: str, new_plan_revision_id: str) -> None:
        with self._lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                row = self.conn.execute("SELECT plan_revision_id,status FROM a2a_runs WHERE run_id=?", (run_id,)).fetchone()
                if row is None or str(row["plan_revision_id"]) != old_plan_revision_id or str(row["status"]) in {"FROZEN", "CANCELLED"}:
                    raise A2AContractError("A2A_SUPERSEDE_BINDING_MISMATCH")
                now = self._now()
                self.conn.execute(
                    "INSERT OR REPLACE INTO a2a_superseded_revisions(run_id,plan_revision_id,superseded_by_plan_revision_id,created_at) VALUES (?,?,?,?)",
                    (run_id, old_plan_revision_id, new_plan_revision_id, now),
                )
                self._converge_obsolete_messages_locked(
                    run_id=run_id,
                    plan_revision_id=old_plan_revision_id,
                    reason="REVISION_SUPERSEDED",
                    error_code="A2A_REVISION_SUPERSEDED",
                )
                self.conn.execute("UPDATE a2a_runs SET plan_revision_id=?,updated_at=? WHERE run_id=?", (new_plan_revision_id, now, run_id))
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise

    def mark_superseded(self, run_id: str, plan_revision_id: str) -> None:
        with self._lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                row = self.conn.execute("SELECT plan_revision_id,status FROM a2a_runs WHERE run_id=?", (run_id,)).fetchone()
                if row is None or str(row["plan_revision_id"]) != plan_revision_id or str(row["status"]) in {"FROZEN", "CANCELLED"}:
                    raise A2AContractError("A2A_SUPERSEDE_BINDING_MISMATCH")
                self.conn.execute(
                    "INSERT OR REPLACE INTO a2a_superseded_revisions(run_id,plan_revision_id,superseded_by_plan_revision_id,created_at) VALUES (?,?,?,?)",
                    (run_id, plan_revision_id, None, self._now()),
                )
                self._converge_obsolete_messages_locked(
                    run_id=run_id,
                    plan_revision_id=plan_revision_id,
                    reason="REVISION_SUPERSEDED",
                    error_code="A2A_REVISION_SUPERSEDED",
                )
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise

    def cancel_run(self, run_id: str) -> None:
        """Atomically cancel a run and close messages that have not started."""

        with self._lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                row = self.conn.execute("SELECT status FROM a2a_runs WHERE run_id=?", (run_id,)).fetchone()
                if row is None:
                    raise KeyError(run_id)
                if str(row["status"]) == "FROZEN":
                    raise A2AContractError("A2A_RUN_TERMINAL")
                self._converge_obsolete_messages_locked(
                    run_id=run_id,
                    reason="RUN_CANCELLED",
                    error_code="A2A_RUN_CANCELLED",
                )
                self.conn.execute(
                    "UPDATE a2a_runs SET status='CANCELLED',updated_at=? WHERE run_id=?",
                    (self._now(), run_id),
                )
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise

    def is_superseded(self, run_id: str, plan_revision_id: str) -> bool:
        with self._lock:
            row = self.conn.execute(
                "SELECT 1 FROM a2a_superseded_revisions WHERE run_id=? AND plan_revision_id=?",
                (run_id, plan_revision_id),
            ).fetchone()
        return row is not None

    def lifecycle(self, run_id: str, plan_revision_id: str) -> tuple[str | None, str | None, bool]:
        """Read the durable run/plan boundary used by dispatch diagnostics."""

        with self._lock:
            row = self.conn.execute("SELECT status,plan_revision_id FROM a2a_runs WHERE run_id=?", (run_id,)).fetchone()
            if row is None:
                return None, None, False
            superseded = self.conn.execute(
                "SELECT 1 FROM a2a_superseded_revisions WHERE run_id=? AND plan_revision_id=?",
                (run_id, plan_revision_id),
            ).fetchone() is not None
        return str(row["status"]), str(row["plan_revision_id"]), superseded

    def set_freeze(self, run_id: str, checksum: str, event_count: int) -> None:
        with self._lock:
            self.conn.execute("UPDATE a2a_runs SET status='FROZEN',freeze_checksum=?,freeze_event_count=?,updated_at=? WHERE run_id=?", (checksum, int(event_count), self._now(), run_id))

    def freeze_if_idle(self, run_id: str) -> dict[str, Any]:
        """Atomically check durable in-flight state and seal the main trace."""

        with self._lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                run = self.conn.execute("SELECT status,freeze_checksum,freeze_event_count FROM a2a_runs WHERE run_id=?", (run_id,)).fetchone()
                if run is None:
                    raise KeyError(run_id)
                status = str(run["status"])
                terminal_sql = ",".join("?" for _ in A2A_TERMINAL_MESSAGE_STATUSES)
                busy_message = self.conn.execute(
                    f"SELECT 1 FROM a2a_messages WHERE run_id=? AND status NOT IN ({terminal_sql}) LIMIT 1",
                    (run_id, *sorted(A2A_TERMINAL_MESSAGE_STATUSES)),
                ).fetchone()
                busy_attempt = self.conn.execute(
                    "SELECT 1 FROM a2a_attempts AS a JOIN a2a_messages AS m ON m.message_id=a.message_id WHERE m.run_id=? AND a.status='RUNNING' LIMIT 1",
                    (run_id,),
                ).fetchone()
                if busy_message is not None or busy_attempt is not None:
                    raise A2AContractError("A2A_RUN_BUSY")
                if status == "CANCELLED":
                    raise A2AContractError("A2A_RUN_TERMINAL")
                try:
                    events, trace = self._verify_live_chain_locked(run_id)
                    self._verify_terminal_messages_locked(run_id, events)
                except A2AContractError as exc:
                    if exc.code.startswith("A2A_TRACE_") or exc.code == "A2A_LIVE_TRACE_CHAIN_MISMATCH":
                        raise A2AContractError("A2A_FREEZE_CHECKSUM_MISMATCH") from exc
                    raise
                checksum = sha256_json(trace)
                if status == "FROZEN":
                    stored_event_count = run["freeze_event_count"]
                    if (
                        str(run["freeze_checksum"] or "") != checksum
                        or stored_event_count is None
                        or int(stored_event_count) != len(trace)
                    ):
                        raise A2AContractError("A2A_FREEZE_CHECKSUM_MISMATCH")
                else:
                    self.conn.execute(
                        "UPDATE a2a_runs SET status='FROZEN',freeze_checksum=?,freeze_event_count=?,updated_at=? WHERE run_id=?",
                        (checksum, len(trace), self._now(), run_id),
                    )
                self.conn.commit()
                return {"run_id": run_id, "status": "FROZEN", "trace": trace, "trace_checksum": checksum, "late_events": [dict(item) for item in self.conn.execute("SELECT * FROM a2a_late_events WHERE run_id=? ORDER BY received_at", (run_id,)).fetchall()]}
            except Exception:
                self.conn.rollback()
                raise

    def freeze_projection(self, run_id: str) -> tuple[str | None, int | None]:
        with self._lock:
            row = self.conn.execute("SELECT freeze_checksum,freeze_event_count FROM a2a_runs WHERE run_id=?", (run_id,)).fetchone()
        return (str(row[0]) if row and row[0] is not None else None, int(row[1]) if row and row[1] is not None else None)

    @staticmethod
    def _safe_envelope(envelope: A2AMessageEnvelopeV1) -> A2AMessageEnvelopeV1:
        updates: dict[str, Any] = {
            "dependency_message_ids": tuple(sorted(str(item) for item in envelope.dependency_message_ids)),
        }
        if envelope.payload is not None:
            safe_payload = _redact(envelope.payload)
            updates.update({"payload": safe_payload, "payload_hash": sha256_json(safe_payload)})
        return envelope.model_copy(update=updates)

    @staticmethod
    def _timestamp(value: datetime) -> str:
        return _utc(value).isoformat().replace("+00:00", "Z")

    @staticmethod
    def _row_dependencies(row: sqlite3.Row) -> tuple[str, ...]:
        raw = row["dependency_message_ids_json"] if "dependency_message_ids_json" in row.keys() else "[]"
        try:
            values = json.loads(str(raw or "[]"))
        except json.JSONDecodeError as exc:
            raise A2AContractError("A2A_REGISTERED_REQUEST_INVALID") from exc
        if not isinstance(values, list) or any(not str(item) for item in values):
            raise A2AContractError("A2A_REGISTERED_REQUEST_INVALID")
        return tuple(sorted(str(item) for item in values))

    @classmethod
    def _row_binding_matches(cls, row: sqlite3.Row, envelope: A2AMessageEnvelopeV1) -> bool:
        """Match the logical request identity used for idempotent replay.

        ``message_id``, ``attempt_id``, and delivery timestamps are transport
        specific.  A same-key replay may receive fresh values for those
        fields, while every routing, authority, correlation, and payload
        field remains bound to the original registration.
        """

        expected = {
            "message_kind": envelope.message_kind,
            "schema_version": envelope.schema_version,
            "idempotency_key": envelope.idempotency_key,
            "run_id": envelope.run_id,
            "plan_revision_id": envelope.plan_revision_id,
            "task_id": envelope.task_id,
            "correlation_id": envelope.correlation_id,
            "parent_message_id": envelope.parent_message_id,
            "sender_ref": envelope.sender_ref,
            "receiver_ref": envelope.receiver_ref,
            "capability_ref": envelope.capability_ref,
            "trace_id": envelope.trace_id,
            "payload_hash": envelope.payload_hash,
        }
        try:
            if not all(str(row[key]) == str(value) for key, value in expected.items()):
                return False
            return cls._row_dependencies(row) == tuple(sorted(str(item) for item in envelope.dependency_message_ids))
        except (KeyError, A2AContractError):
            return False

    @classmethod
    def _row_full_binding_matches(
        cls,
        row: sqlite3.Row,
        envelope: A2AMessageEnvelopeV1,
        *,
        persisted_projection: bool = False,
    ) -> bool:
        """Match every independently persisted envelope identity field.

        The request JSON is a redacted projection, so its payload hash is
        checked against ``request_payload_hash``.  The separate ``payload_hash``
        column remains the original caller hash and is checked against the
        incoming request by :meth:`_row_binding_matches`.
        """

        expected = {
            "message_id": envelope.message_id,
            "message_kind": envelope.message_kind,
            "schema_version": envelope.schema_version,
            "attempt_id": envelope.attempt_id,
            "trace_id": envelope.trace_id,
            "idempotency_key": envelope.idempotency_key,
            "run_id": envelope.run_id,
            "plan_revision_id": envelope.plan_revision_id,
            "task_id": envelope.task_id,
            "correlation_id": envelope.correlation_id,
            "parent_message_id": envelope.parent_message_id,
            "sender_ref": envelope.sender_ref,
            "receiver_ref": envelope.receiver_ref,
            "capability_ref": envelope.capability_ref,
            "envelope_created_at": cls._timestamp(envelope.created_at),
            "envelope_deadline": cls._timestamp(envelope.deadline),
        }
        payload_column = "request_payload_hash" if persisted_projection else "payload_hash"
        expected[payload_column] = envelope.payload_hash
        try:
            if not all(str(row[key]) == str(value) for key, value in expected.items()):
                return False
            return cls._row_dependencies(row) == tuple(sorted(str(item) for item in envelope.dependency_message_ids))
        except (KeyError, A2AContractError):
            return False

    @classmethod
    def _incoming_binding_matches(cls, row: sqlite3.Row, envelope: A2AMessageEnvelopeV1) -> bool:
        if str(row["message_id"]) == envelope.message_id:
            return cls._row_full_binding_matches(row, envelope)
        return cls._row_binding_matches(row, envelope)

    @classmethod
    def _response_row_matches(cls, row: sqlite3.Row, response: A2AMessageEnvelopeV1) -> bool:
        expected = {
            "response_message_id": response.message_id,
            "response_message_kind": response.message_kind,
            "response_schema_version": response.schema_version,
            "response_run_id": response.run_id,
            "response_plan_revision_id": response.plan_revision_id,
            "response_task_id": response.task_id,
            "response_correlation_id": response.correlation_id,
            "response_parent_message_id": response.parent_message_id,
            "response_trace_id": response.trace_id,
            "response_attempt_id": response.attempt_id,
            "response_sender_ref": response.sender_ref,
            "response_receiver_ref": response.receiver_ref,
            "response_capability_ref": response.capability_ref,
            "response_idempotency_key": response.idempotency_key,
            "response_created_at": cls._timestamp(response.created_at),
            "response_deadline": cls._timestamp(response.deadline),
            "response_payload_hash": response.payload_hash,
        }
        try:
            return all(str(row[key]) == str(value) for key, value in expected.items())
        except KeyError:
            return False

    def register_request(self, envelope: A2AMessageEnvelopeV1) -> sqlite3.Row | None:
        """Register trusted runtime context before an envelope is dispatchable."""

        fp = request_fingerprint(envelope)
        now = self._now()
        safe = self._safe_envelope(envelope)
        safe_value = safe.model_dump(mode="json")
        request_checksum = sha256_json(safe_value)
        encoded = json.dumps(safe_value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        dependency_json = json.dumps(sorted(str(item) for item in envelope.dependency_message_ids), ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                run = self.conn.execute("SELECT status,plan_revision_id FROM a2a_runs WHERE run_id=?", (envelope.run_id,)).fetchone()
                if run is None:
                    raise A2AContractError("A2A_RUN_UNREGISTERED")
                if str(run["status"]) != "OPEN":
                    raise A2AContractError("A2A_RUN_TERMINAL")
                if str(run["plan_revision_id"]) != envelope.plan_revision_id:
                    raise A2AContractError("A2A_PLAN_NOT_CURRENT")
                if self.conn.execute(
                    "SELECT 1 FROM a2a_superseded_revisions WHERE run_id=? AND plan_revision_id=?",
                    (envelope.run_id, envelope.plan_revision_id),
                ).fetchone() is not None:
                    raise A2AContractError("A2A_REVISION_SUPERSEDED")
                by_message = self.conn.execute("SELECT * FROM a2a_messages WHERE message_id=?", (envelope.message_id,)).fetchone()
                if by_message is not None:
                    if str(by_message["request_fingerprint"]) != fp or not self._row_full_binding_matches(by_message, envelope):
                        raise A2AContractError("A2A_REGISTERED_REQUEST_MISMATCH")
                    self.conn.commit()
                    return by_message
                by_key = self.conn.execute("SELECT * FROM a2a_messages WHERE idempotency_key=?", (envelope.idempotency_key,)).fetchone()
                if by_key is not None:
                    if str(by_key["request_fingerprint"]) != fp or not self._row_binding_matches(by_key, envelope):
                        raise A2AContractError("A2A_IDEMPOTENCY_CONFLICT")
                    self.conn.commit()
                    return by_key
                self.conn.execute(
                    "INSERT INTO a2a_messages(message_id,idempotency_key,request_fingerprint,request_checksum,message_kind,schema_version,attempt_id,trace_id,envelope_created_at,envelope_deadline,run_id,plan_revision_id,task_id,correlation_id,parent_message_id,dependency_message_ids_json,sender_ref,receiver_ref,capability_ref,payload_hash,request_payload_hash,status,request_json,response_json,response_checksum,response_message_id,response_message_kind,response_schema_version,response_run_id,response_plan_revision_id,response_task_id,response_correlation_id,response_parent_message_id,response_trace_id,response_attempt_id,response_sender_ref,response_receiver_ref,response_capability_ref,response_idempotency_key,response_created_at,response_deadline,response_payload_hash,specialist_json,specialist_checksum,canonical_json,canonical_checksum,duplicate_of,error_code,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (envelope.message_id, envelope.idempotency_key, fp, request_checksum, envelope.message_kind, envelope.schema_version, envelope.attempt_id, envelope.trace_id, self._timestamp(envelope.created_at), self._timestamp(envelope.deadline), envelope.run_id, envelope.plan_revision_id, envelope.task_id, envelope.correlation_id, envelope.parent_message_id, dependency_json, envelope.sender_ref, envelope.receiver_ref, envelope.capability_ref, envelope.payload_hash, safe.payload_hash, "REGISTERED", encoded, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, now, now),
                )
                self.conn.commit()
                return self.conn.execute("SELECT * FROM a2a_messages WHERE message_id=?", (envelope.message_id,)).fetchone()
            except Exception:
                self.conn.rollback()
                raise

    def reserve(self, envelope: A2AMessageEnvelopeV1) -> tuple[str, sqlite3.Row | None]:
        """Reserve one request with the durable run boundary in one transaction."""

        fp = request_fingerprint(envelope)
        now = self._now()
        with self._lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                run = self.conn.execute("SELECT status,plan_revision_id FROM a2a_runs WHERE run_id=?", (envelope.run_id,)).fetchone()
                if run is None:
                    self.conn.commit()
                    return "RUN_UNREGISTERED", None
                superseded = self.conn.execute(
                    "SELECT 1 FROM a2a_superseded_revisions WHERE run_id=? AND plan_revision_id=?",
                    (envelope.run_id, envelope.plan_revision_id),
                ).fetchone() is not None
                row = self.conn.execute("SELECT * FROM a2a_messages WHERE idempotency_key=?", (envelope.idempotency_key,)).fetchone()
                if str(run["status"]) in {"CANCELLED", "FROZEN"} or superseded:
                    self.conn.commit()
                    return "LATE", row
                if str(run["plan_revision_id"]) != envelope.plan_revision_id:
                    self.conn.commit()
                    return "PLAN_NOT_CURRENT", row
                if row is None:
                    self.conn.commit()
                    return "UNREGISTERED", None
                if str(row["request_fingerprint"]) != fp or not self._incoming_binding_matches(row, envelope):
                    self.conn.commit()
                    return "CONFLICT", row
                status = str(row["status"])
                if status in {"REGISTERED", "PENDING"}:
                    self.conn.execute("UPDATE a2a_messages SET status='DISPATCHED',updated_at=? WHERE message_id=?", (now, str(row["message_id"])))
                    self.conn.commit()
                    return "NEW", row
                self.conn.commit()
                if status in {"PENDING", "DISPATCHED", "RUNNING"}:
                    return "PENDING", row
                if status == "LATE":
                    return "LATE", row
                return "DUPLICATE", row
            except Exception:
                self.conn.rollback()
                raise

    def validate_registered(self, envelope: A2AMessageEnvelopeV1) -> tuple[str, sqlite3.Row | None]:
        """Validate a dispatch against trusted registration, never self-asserted fields."""

        fp = request_fingerprint(envelope)
        with self._lock:
            row = self.conn.execute("SELECT * FROM a2a_messages WHERE message_id=?", (envelope.message_id,)).fetchone()
            if row is None:
                row = self.conn.execute("SELECT * FROM a2a_messages WHERE idempotency_key=?", (envelope.idempotency_key,)).fetchone()
            if row is None:
                return "UNREGISTERED", None
            if str(row["request_fingerprint"]) != fp or not self._incoming_binding_matches(row, envelope):
                return "CONFLICT", row
            return "REGISTERED", row

    def set_message_status(self, message_id: str, status: str, *, error_code: str | None = None) -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE a2a_messages SET status=?,error_code=?,updated_at=? WHERE message_id=?",
                (status, error_code, self._now(), message_id),
            )

    def _record_late_locked(
        self,
        *,
        run_id: str,
        message_id: str,
        reason: str,
        received_at: datetime,
        trace_id: str | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        existing = self.conn.execute(
            "SELECT 1 FROM a2a_late_events WHERE run_id=? AND message_id=? AND reason=? LIMIT 1",
            (run_id, message_id, reason),
        ).fetchone()
        if existing is not None:
            return
        safe = _redact(dict(payload or {}))
        self.conn.execute(
            "INSERT INTO a2a_late_events(audit_id,run_id,message_id,reason,trace_id,payload_json,payload_hash,received_at) VALUES (?,?,?,?,?,?,?,?)",
            (f"late_{uuid.uuid4().hex}", run_id, message_id, reason, trace_id, json.dumps(safe, ensure_ascii=False, sort_keys=True, separators=(",", ":")), sha256_json(safe), _utc(received_at).isoformat()),
        )

    def finalize_terminal(
        self,
        message_id: str,
        response: A2AMessageEnvelopeV1,
        *,
        specialist: TypedAgentResult,
        canonical: Result,
        status: str,
        error_code: str | None = None,
        received_at: datetime | None = None,
        scene_clock: datetime | None = None,
        terminal_event_type: str | None = None,
        terminal_event_payload: Mapping[str, Any] | None = None,
    ) -> str:
        """Atomically finalize artifacts, attempt, message, and terminal trace.

        The physical adapter call happens outside the ledger transaction.  A
        supervisor may therefore cancel or supersede the run while that call
        is blocked.  This method is the single atomic commit boundary: it
        rechecks lifecycle, converts a stale return to a durable late audit,
        and closes the message/attempt without writing canonical artifacts.
        """

        safe_response = self._safe_envelope(response)
        response_value = safe_response.model_dump(mode="json")
        # Hash the final JSON-safe redacted projection.  ``mode="python"``
        # would leave datetimes/enums for ``json.dumps(default=str)`` and
        # could produce bytes that no longer match the pre-serialization
        # canonical hash after a runtime restart.
        specialist_value = _redact(specialist.model_dump(mode="json"))
        canonical_value = _redact(canonical.model_dump(mode="json"))
        response_checksum = sha256_json(response_value)
        specialist_checksum = sha256_json(specialist_value)
        canonical_checksum = sha256_json(canonical_value)
        with self._lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                row = self.conn.execute("SELECT * FROM a2a_messages WHERE message_id=?", (message_id,)).fetchone()
                if row is None:
                    raise A2AContractError("A2A_MESSAGE_UNREGISTERED")
                if response.attempt_id != canonical.attempt_id:
                    raise A2AContractError("A2A_RESPONSE_ATTEMPT_MISMATCH")
                run = self.conn.execute("SELECT status,plan_revision_id FROM a2a_runs WHERE run_id=?", (str(row["run_id"]),)).fetchone()
                if run is None:
                    raise A2AContractError("A2A_RUN_UNREGISTERED")
                superseded = self.conn.execute(
                    "SELECT 1 FROM a2a_superseded_revisions WHERE run_id=? AND plan_revision_id=?",
                    (str(row["run_id"]), str(row["plan_revision_id"])),
                ).fetchone() is not None
                run_status = str(run["status"])
                current_plan = str(run["plan_revision_id"])
                late_reason: str | None = None
                if run_status == "CANCELLED":
                    late_reason = "RUN_CANCELLED"
                elif run_status == "FROZEN":
                    late_reason = "RUN_TERMINAL"
                elif superseded or current_plan != str(row["plan_revision_id"]):
                    late_reason = "REVISION_SUPERSEDED"

                now = self._now()
                if late_reason is not None:
                    # Close every outstanding attempt, including an attempt
                    # whose adapter returned just before this transaction.
                    self.conn.execute(
                        "UPDATE a2a_attempts SET status='LATE',error_code=?,ended_at=COALESCE(ended_at,?) WHERE message_id=? AND status IN ('RUNNING','SUCCEEDED','FAILED')",
                        (f"A2A_{late_reason}", now, message_id),
                    )
                    self.conn.execute(
                        "UPDATE a2a_messages SET status='LATE',response_json=NULL,response_checksum=NULL,response_message_id=NULL,response_message_kind=NULL,response_schema_version=NULL,response_run_id=NULL,response_plan_revision_id=NULL,response_task_id=NULL,response_correlation_id=NULL,response_parent_message_id=NULL,response_trace_id=NULL,response_attempt_id=NULL,response_sender_ref=NULL,response_receiver_ref=NULL,response_capability_ref=NULL,response_idempotency_key=NULL,response_created_at=NULL,response_deadline=NULL,response_payload_hash=NULL,specialist_json=NULL,specialist_checksum=NULL,canonical_json=NULL,canonical_checksum=NULL,error_code=?,updated_at=? WHERE message_id=?",
                        (f"A2A_{late_reason}", now, message_id),
                    )
                    self._record_late_locked(
                        run_id=str(row["run_id"]),
                        message_id=message_id,
                        reason=late_reason,
                        received_at=_utc(received_at or datetime.now(timezone.utc)),
                        trace_id=response.trace_id,
                        payload={"reason": late_reason, "response_message_id": response.message_id, "response_payload_hash": response.payload_hash},
                    )
                    self.conn.commit()
                    return "LATE"

                if str(row["status"]) in A2A_TERMINAL_MESSAGE_STATUSES:
                    raise A2AContractError("A2A_MESSAGE_LIFECYCLE_MISMATCH")
                if status not in {"SUCCEEDED", "FAILED"}:
                    raise A2AContractError("A2A_TERMINAL_STATUS_INVALID")
                expected_event_type = "A2A_RESULT_VERIFIED" if status == "SUCCEEDED" else "A2A_MESSAGE_FAILED"
                if terminal_event_type != expected_event_type:
                    raise A2AContractError("A2A_TERMINAL_TRACE_REQUIRED")
                if canonical.status.value != status or (status == "SUCCEEDED" and (not specialist.ok or error_code is not None or canonical.error_ref is not None)):
                    raise A2AContractError("A2A_TERMINAL_RESULT_MISMATCH")
                if status == "FAILED" and (specialist.ok or not (error_code or canonical.error_ref) or canonical.error_ref != (error_code or canonical.error_ref)):
                    raise A2AContractError("A2A_TERMINAL_ERROR_MISMATCH")
                attempt = self.conn.execute(
                    "SELECT * FROM a2a_attempts WHERE message_id=? AND attempt_id=?",
                    (message_id, response.attempt_id),
                ).fetchone()
                if attempt is None:
                    raise A2AContractError("A2A_RESPONSE_ATTEMPT_MISMATCH")
                if str(attempt["status"]) not in {"RUNNING", status}:
                    raise A2AContractError("A2A_ATTEMPT_STATUS_MISMATCH")
                attempts = self.conn.execute("SELECT * FROM a2a_attempts WHERE message_id=? ORDER BY attempt_no", (message_id,)).fetchall()
                physical_attempts = sum(1 for item in attempts if int(item["physical_call"]))
                if canonical.usage.logical_calls != 1 or canonical.usage.physical_attempts != physical_attempts:
                    raise A2AContractError("A2A_TERMINAL_USAGE_MISMATCH")
                expected_error = error_code or canonical.error_ref
                if status == "FAILED" and str(attempt["error_code"] or expected_error) != str(expected_error):
                    raise A2AContractError("A2A_ATTEMPT_ERROR_MISMATCH")
                terminal_clock = _utc(scene_clock or received_at or datetime.now(timezone.utc))
                self.conn.execute(
                    "UPDATE a2a_attempts SET status=?,error_code=?,ended_at=COALESCE(ended_at,?),scene_clock=? WHERE attempt_id=? AND message_id=?",
                    (status, expected_error, now, terminal_clock.isoformat(), response.attempt_id, message_id),
                )
                self.conn.execute(
                    "UPDATE a2a_messages SET status=?,response_json=?,response_checksum=?,response_message_id=?,response_message_kind=?,response_schema_version=?,response_run_id=?,response_plan_revision_id=?,response_task_id=?,response_correlation_id=?,response_parent_message_id=?,response_trace_id=?,response_attempt_id=?,response_sender_ref=?,response_receiver_ref=?,response_capability_ref=?,response_idempotency_key=?,response_created_at=?,response_deadline=?,response_payload_hash=?,specialist_json=?,specialist_checksum=?,canonical_json=?,canonical_checksum=?,error_code=?,updated_at=? WHERE message_id=?",
                    (status, json.dumps(response_value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str), response_checksum, safe_response.message_id, safe_response.message_kind, safe_response.schema_version, safe_response.run_id, safe_response.plan_revision_id, safe_response.task_id, safe_response.correlation_id, safe_response.parent_message_id, safe_response.trace_id, safe_response.attempt_id, safe_response.sender_ref, safe_response.receiver_ref, safe_response.capability_ref, safe_response.idempotency_key, self._timestamp(safe_response.created_at), self._timestamp(safe_response.deadline), safe_response.payload_hash, json.dumps(specialist_value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str), specialist_checksum, json.dumps(canonical_value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str), canonical_checksum, expected_error, now, message_id),
                )
                event_payload = dict(terminal_event_payload or {})
                event_payload.setdefault("attempt_no", int(attempt["attempt_no"]))
                if status == "SUCCEEDED":
                    event_payload.setdefault("result_id", canonical.result_id)
                    event_payload.setdefault("payload_hash", specialist.payload_hash)
                else:
                    if "error_code" in event_payload and str(event_payload["error_code"]) != str(expected_error):
                        raise A2AContractError("A2A_TERMINAL_TRACE_ERROR_MISMATCH")
                    event_payload["error_code"] = expected_error
                    event_payload.setdefault("attempts", len(attempts))
                self._append_trace_locked(
                    run_id=str(row["run_id"]),
                    trace_id=response.trace_id,
                    event_type=expected_event_type,
                    scene_clock=terminal_clock,
                    payload=event_payload,
                    message_id=message_id,
                    attempt_id=response.attempt_id,
                )
                self.conn.commit()
                return "SAVED"
            except Exception:
                self.conn.rollback()
                raise

    def save_response(
        self,
        message_id: str,
        response: A2AMessageEnvelopeV1,
        *,
        specialist: TypedAgentResult,
        canonical: Result,
        status: str,
        error_code: str | None = None,
        received_at: datetime | None = None,
    ) -> str:
        """Compatibility entry point using the shared terminal primitive."""

        event_type = "A2A_RESULT_VERIFIED" if status == "SUCCEEDED" else "A2A_MESSAGE_FAILED"
        return self.finalize_terminal(
            message_id,
            response,
            specialist=specialist,
            canonical=canonical,
            status=status,
            error_code=error_code,
            received_at=received_at,
            scene_clock=received_at,
            terminal_event_type=event_type,
        )

    def _verify_terminal_trace(
        self,
        *,
        row: sqlite3.Row,
        request: A2AMessageEnvelopeV1,
        response: A2AMessageEnvelopeV1,
        specialist: TypedAgentResult,
        canonical: Result,
        latest_attempt: sqlite3.Row,
        expected_status: str,
    ) -> tuple[R4TraceEventV1, Mapping[str, Any]]:
        """Verify one terminal trace row and its complete provenance.

        Terminal trace metadata is part of the durable result graph.  Query
        by the registered run/message pair first, then validate every
        terminal discriminator together so changing one column cannot turn a
        valid artifact bundle into a replayable result.
        """

        expected_event_type = "A2A_RESULT_VERIFIED" if expected_status == "SUCCEEDED" else "A2A_MESSAGE_FAILED"
        all_trace, trace_projection = self._verify_live_chain_locked(str(row["run_id"]))
        trace_rows = [
            event
            for event in all_trace
            if event.message_id == str(row["message_id"])
        ]
        run_projection = self.conn.execute(
            "SELECT status,freeze_checksum,freeze_event_count FROM a2a_runs WHERE run_id=?",
            (str(row["run_id"]),),
        ).fetchone()
        if run_projection is not None and str(run_projection["status"]) == "FROZEN":
            stored_event_count = run_projection["freeze_event_count"]
            if (
                stored_event_count is None
                or int(stored_event_count) != len(trace_projection)
                or str(run_projection["freeze_checksum"] or "") != sha256_json(trace_projection)
            ):
                raise A2AContractError("A2A_TRACE_FROZEN_CHECKSUM_MISMATCH")
        terminal_rows = [
            item
            for item in trace_rows
            if item.event_type in {"A2A_RESULT_VERIFIED", "A2A_MESSAGE_FAILED"}
        ]
        if len(terminal_rows) != 1:
            raise A2AContractError("A2A_PERSISTED_TERMINAL_TRACE_MISSING")
        terminal_trace = terminal_rows[0]
        expected_provenance = {
            "run_id": str(row["run_id"]),
            "message_id": str(row["message_id"]),
            "attempt_id": str(latest_attempt["attempt_id"]),
            "trace_id": request.trace_id,
            "event_type": expected_event_type,
        }
        actual_provenance = {
            "run_id": terminal_trace.run_id,
            "message_id": terminal_trace.message_id,
            "attempt_id": terminal_trace.attempt_id,
            "trace_id": terminal_trace.trace_id,
            "event_type": terminal_trace.event_type,
        }
        if actual_provenance != expected_provenance or response.trace_id != request.trace_id or response.attempt_id != str(latest_attempt["attempt_id"]):
            raise A2AContractError("A2A_PERSISTED_TERMINAL_TRACE_PROVENANCE_MISMATCH")
        terminal_payload = terminal_trace.payload
        if sha256_json(terminal_payload) != terminal_trace.payload_hash:
            raise A2AContractError("A2A_PERSISTED_TERMINAL_TRACE_CHECKSUM_MISMATCH")
        if int(terminal_payload.get("attempt_no", -1)) != int(latest_attempt["attempt_no"]):
            raise A2AContractError("A2A_PERSISTED_TERMINAL_TRACE_ATTEMPT_MISMATCH")
        latest_error = str(latest_attempt["error_code"] or "")
        if expected_status == "SUCCEEDED":
            if (
                str(terminal_payload.get("result_id") or "") != canonical.result_id
                or str(terminal_payload.get("payload_hash") or "") != specialist.payload_hash
            ):
                raise A2AContractError("A2A_PERSISTED_TERMINAL_TRACE_RESULT_MISMATCH")
        elif str(terminal_payload.get("error_code") or "") != latest_error:
            raise A2AContractError("A2A_PERSISTED_TERMINAL_TRACE_ERROR_MISMATCH")
        return terminal_trace, terminal_payload

    def load_cached(self, row: sqlite3.Row, *, request: A2AMessageEnvelopeV1, verifier: Any) -> tuple[A2AMessageEnvelopeV1, TypedAgentResult, Result, str | None]:
        """Rehydrate from SQLite and cross-verify every persisted artifact.

        This method intentionally ignores all process-local caches.  The
        request, response, specialist result, canonical result, and attempt
        ledger form one durable graph; a checksum-valid but semantically
        inconsistent graph is still rejected.
        """

        fields = (
            row["request_json"], row["request_checksum"], row["request_fingerprint"],
            row["response_json"], row["response_checksum"],
            row["specialist_json"], row["specialist_checksum"],
            row["canonical_json"], row["canonical_checksum"],
        )
        if any(value is None for value in fields):
            raise A2AContractError("A2A_PERSISTED_RESULT_MISSING")
        try:
            request_value = json.loads(str(row["request_json"]))
            response_value = json.loads(str(row["response_json"]))
            specialist_value = json.loads(str(row["specialist_json"]))
            canonical_value = json.loads(str(row["canonical_json"]))
            if sha256_json(request_value) != str(row["request_checksum"]):
                raise A2AContractError("A2A_PERSISTED_REQUEST_CHECKSUM_MISMATCH")
            if sha256_json(response_value) != str(row["response_checksum"]):
                raise A2AContractError("A2A_PERSISTED_RESPONSE_CHECKSUM_MISMATCH")
            if sha256_json(specialist_value) != str(row["specialist_checksum"]):
                raise A2AContractError("A2A_PERSISTED_SPECIALIST_CHECKSUM_MISMATCH")
            if sha256_json(canonical_value) != str(row["canonical_checksum"]):
                raise A2AContractError("A2A_PERSISTED_RESULT_CHECKSUM_MISMATCH")
            registered_request = A2AMessageEnvelopeV1.model_validate(request_value)
            response = A2AMessageEnvelopeV1.model_validate(response_value)
            specialist = TypedAgentResult.model_validate(specialist_value)
            canonical = Result.model_validate(canonical_value)

            if registered_request.message_kind != "REQUEST" or registered_request.message_id != str(row["message_id"]):
                raise A2AContractError("A2A_PERSISTED_REQUEST_OWNERSHIP_MISMATCH")
            if request_fingerprint(request) != str(row["request_fingerprint"]):
                raise A2AContractError("A2A_PERSISTED_REQUEST_FINGERPRINT_MISMATCH")
            if not self._row_binding_matches(row, request):
                raise A2AContractError("A2A_PERSISTED_REQUEST_BINDING_MISMATCH")
            if registered_request.parent_message_id != request.parent_message_id or tuple(sorted(registered_request.dependency_message_ids)) != tuple(sorted(request.dependency_message_ids)):
                raise A2AContractError("A2A_PERSISTED_REQUEST_DEPENDENCY_MISMATCH")
            if registered_request.payload_hash != sha256_json(registered_request.payload):
                raise A2AContractError("A2A_PERSISTED_REQUEST_HASH_MISMATCH")
            safe_incoming = self._safe_envelope(request)
            if registered_request.payload != safe_incoming.payload or registered_request.payload_hash != safe_incoming.payload_hash:
                raise A2AContractError("A2A_PERSISTED_REQUEST_PAYLOAD_MISMATCH")
            if not self._row_full_binding_matches(row, registered_request, persisted_projection=True):
                raise A2AContractError("A2A_PERSISTED_REQUEST_IDENTITY_MISMATCH")

            # Re-run the deterministic result and specialist verifiers with
            # the incoming request context, not with self-asserted artifact
            # fields.
            # The persisted response belongs to the original registered
            # request.  A same-key replay may carry a new transport message
            # id, so verify parent/correlation against that durable request;
            # the incoming request fingerprint was checked above.
            verifier.verify_result_envelope(registered_request, response)
            if not self._response_row_matches(row, response):
                raise A2AContractError("A2A_PERSISTED_RESPONSE_IDENTITY_MISMATCH")
            verifier.verify_specialist_result(registered_request, specialist)
            expected_inner = {
                "contract": specialist.contract,
                "ok": specialist.ok,
                "payload": specialist.payload,
                "payload_hash": specialist.payload_hash,
                "agent_ref": specialist.agent_ref,
                "tool_ref": specialist.tool_ref,
            }
            if response.message_kind == "RESULT":
                if response.payload is None or sha256_json(response.payload) != sha256_json(expected_inner) or response.payload != expected_inner:
                    raise A2AContractError("A2A_PERSISTED_RESPONSE_SPECIALIST_MISMATCH")
                if not specialist.ok:
                    raise A2AContractError("A2A_PERSISTED_RESPONSE_STATUS_MISMATCH")
            else:
                if specialist.ok or response.error is None or response.error.code != specialist.error_code:
                    raise A2AContractError("A2A_PERSISTED_RESPONSE_STATUS_MISMATCH")

            attempts = self.attempts(str(row["message_id"]))
            if not attempts:
                raise A2AContractError("A2A_PERSISTED_ATTEMPT_MISSING")
            attempt_ids = {str(item["attempt_id"]) for item in attempts}
            if any(str(item["status"]) == "RUNNING" or item["ended_at"] is None for item in attempts):
                raise A2AContractError("A2A_PERSISTED_ATTEMPT_NOT_TERMINAL")
            latest_attempt = max(attempts, key=lambda item: int(item["attempt_no"]))
            physical_attempts = sum(1 for item in attempts if int(item["physical_call"]))
            if canonical.attempt_id not in attempt_ids:
                raise A2AContractError("A2A_PERSISTED_ATTEMPT_OWNERSHIP_MISMATCH")
            if response.attempt_id != canonical.attempt_id:
                raise A2AContractError("A2A_PERSISTED_RESPONSE_ATTEMPT_MISMATCH")
            if canonical.attempt_id != str(latest_attempt["attempt_id"]):
                raise A2AContractError("A2A_PERSISTED_LATEST_ATTEMPT_MISMATCH")
            if str(row["status"]) not in A2A_TERMINAL_MESSAGE_STATUSES or str(row["status"]) not in {"SUCCEEDED", "FAILED"}:
                raise A2AContractError("A2A_PERSISTED_RESULT_STATUS_MISMATCH")
            expected_status = "SUCCEEDED" if str(row["status"]) == "SUCCEEDED" else "FAILED"
            if str(canonical.status.value) != expected_status:
                raise A2AContractError("A2A_PERSISTED_RESULT_STATUS_MISMATCH")
            if str(latest_attempt["status"]) != expected_status:
                raise A2AContractError("A2A_PERSISTED_LATEST_ATTEMPT_STATUS_MISMATCH")
            try:
                terminal_trace, terminal_payload = self._verify_terminal_trace(
                    row=row,
                    request=registered_request,
                    response=response,
                    specialist=specialist,
                    canonical=canonical,
                    latest_attempt=latest_attempt,
                    expected_status=expected_status,
                )
            except A2AContractError as exc:
                if exc.code.startswith("A2A_TRACE_"):
                    raise A2AContractError(f"A2A_PERSISTED_TERMINAL_TRACE_{exc.code[len('A2A_TRACE_'):]}") from exc
                if exc.code == "A2A_LIVE_TRACE_CHAIN_MISMATCH":
                    raise A2AContractError("A2A_PERSISTED_TERMINAL_TRACE_LIVE_CHAIN_MISMATCH") from exc
                raise
            if canonical.run_id != str(row["run_id"]) or canonical.plan_revision_id != str(row["plan_revision_id"]) or canonical.task_id != str(row["task_id"]):
                raise A2AContractError("A2A_PERSISTED_RESULT_OWNERSHIP_MISMATCH")
            if canonical.output_contract != specialist.contract or canonical.payload != specialist.payload or canonical.payload_hash != sha256_json(specialist.payload):
                raise A2AContractError("A2A_PERSISTED_RESULT_SPECIALIST_MISMATCH")
            expected_evidence = [str(item["evidence_id"]) for item in (specialist.payload or {}).get("evidence", []) if isinstance(item, Mapping) and item.get("evidence_id")] if specialist.ok and isinstance(specialist.payload, Mapping) else []
            if canonical.evidence_refs != expected_evidence or canonical.error_ref != (specialist.error_code if not specialist.ok else None):
                raise A2AContractError("A2A_PERSISTED_RESULT_SEMANTIC_MISMATCH")
            latest_error = str(latest_attempt["error_code"] or "")
            if expected_status == "FAILED":
                if not latest_error or canonical.error_ref != latest_error or specialist.error_code != latest_error or response.error is None or response.error.code != latest_error:
                    raise A2AContractError("A2A_PERSISTED_LATEST_ATTEMPT_ERROR_MISMATCH")
            elif latest_error or response.error is not None:
                raise A2AContractError("A2A_PERSISTED_LATEST_ATTEMPT_ERROR_MISMATCH")
            if canonical.usage.logical_calls != 1 or canonical.usage.physical_attempts != physical_attempts:
                raise A2AContractError("A2A_PERSISTED_USAGE_MISMATCH")
            return response, specialist, canonical, str(row["message_id"])
        except A2AContractError:
            raise
        except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise A2AContractError("A2A_PERSISTED_RESULT_INVALID") from exc

    def row(self, message_id: str) -> sqlite3.Row | None:
        with self._lock:
            return self.conn.execute("SELECT * FROM a2a_messages WHERE message_id=?", (message_id,)).fetchone()

    def registered_deadline(self, idempotency_key: str) -> datetime | None:
        """Return the durable logical deadline for a same-key replay."""

        with self._lock:
            row = self.conn.execute("SELECT envelope_deadline FROM a2a_messages WHERE idempotency_key=?", (idempotency_key,)).fetchone()
        if row is None or row[0] is None:
            return None
        try:
            return _utc(datetime.fromisoformat(str(row[0]).replace("Z", "+00:00")))
        except ValueError as exc:
            raise A2AContractError("A2A_REGISTERED_REQUEST_INVALID") from exc

    def record_attempt(self, *, message_id: str, attempt_id: str, attempt_no: int, status: str, physical_call: bool, scene_clock: datetime | None = None, error_code: str | None = None, ended: bool = False) -> None:
        now = self._now()
        with self._lock:
            if ended:
                self.conn.execute(
                    "UPDATE a2a_attempts SET status=?,error_code=?,physical_call=?,scene_clock=?,ended_at=? WHERE message_id=? AND attempt_no=?",
                    (status, error_code, int(physical_call), _utc(scene_clock).isoformat() if scene_clock else None, now, message_id, attempt_no),
                )
                return
            self.conn.execute(
                "INSERT INTO a2a_attempts(attempt_id,message_id,attempt_no,status,error_code,physical_call,started_at,scene_clock,ended_at) VALUES (?,?,?,?,?,?,?,?,NULL)",
                (attempt_id, message_id, attempt_no, status, error_code, int(physical_call), now, _utc(scene_clock).isoformat() if scene_clock else None),
            )

    def attempts(self, message_id: str) -> list[sqlite3.Row]:
        with self._lock:
            return list(self.conn.execute("SELECT * FROM a2a_attempts WHERE message_id=? ORDER BY attempt_no", (message_id,)).fetchall())

    @staticmethod
    def _trace_event_from_row(row: sqlite3.Row, *, run_id: str) -> R4TraceEventV1:
        try:
            if str(row["run_id"]) != run_id:
                raise A2AContractError("A2A_TRACE_PROVENANCE_MISMATCH")
            payload = json.loads(str(row["payload_json"]))
            if not isinstance(payload, Mapping):
                raise A2AContractError("A2A_TRACE_INVALID")
            payload_hash = str(row["payload_hash"])
            if sha256_json(payload) != payload_hash:
                raise A2AContractError("A2A_TRACE_PAYLOAD_HASH_MISMATCH")
            scene_clock = _utc(datetime.fromisoformat(str(row["scene_clock"]).replace("Z", "+00:00")))
            event_id = project_trace_event_id(
                seq_no=int(row["seq_no"]),
                run_id=run_id,
                trace_id=str(row["trace_id"]),
                event_type=str(row["event_type"]),
                message_id=row["message_id"],
                attempt_id=row["attempt_id"],
                actor=str(row["actor"]),
                scene_clock=scene_clock,
                payload_hash=payload_hash,
            )
            if str(row["event_id"]) != event_id:
                raise A2AContractError("A2A_TRACE_EVENT_ID_MISMATCH")
            return R4TraceEventV1(
                seq_no=int(row["seq_no"]),
                event_id=event_id,
                run_id=run_id,
                trace_id=str(row["trace_id"]),
                event_type=str(row["event_type"]),
                message_id=row["message_id"],
                attempt_id=row["attempt_id"],
                actor=str(row["actor"]),
                scene_clock=scene_clock,
                payload=dict(payload),
                payload_hash=payload_hash,
            )
        except A2AContractError:
            raise
        except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise A2AContractError("A2A_TRACE_INVALID") from exc

    def _read_trace_locked(self, run_id: str) -> tuple[R4TraceEventV1, ...]:
        rows = self.conn.execute("SELECT * FROM a2a_trace WHERE run_id=? ORDER BY seq_no", (run_id,)).fetchall()
        seqs = [int(row["seq_no"]) for row in rows]
        if seqs != list(range(1, len(seqs) + 1)) or len(seqs) != len(set(seqs)):
            raise A2AContractError("A2A_TRACE_SEQUENCE_INVALID")
        return tuple(self._trace_event_from_row(row, run_id=run_id) for row in rows)

    def _verify_live_chain_locked(self, run_id: str) -> tuple[tuple[R4TraceEventV1, ...], list[dict[str, Any]]]:
        events = self._read_trace_locked(run_id)
        projection = [event.model_dump(mode="json") for event in events]
        run = self.conn.execute(
            "SELECT live_event_count,live_head,live_checksum FROM a2a_runs WHERE run_id=?",
            (run_id,),
        ).fetchone()
        if run is None:
            raise A2AContractError("A2A_RUN_UNREGISTERED")
        stored_count = run["live_event_count"]
        stored_head = run["live_head"]
        stored_checksum = run["live_checksum"]
        expected_head = events[-1].event_id if events else ""
        expected_checksum = sha256_json(projection)
        if (
            stored_count is None
            or int(stored_count) != len(events)
            or stored_head is None
            or str(stored_head) != expected_head
            or stored_checksum is None
            or str(stored_checksum) != expected_checksum
        ):
            raise A2AContractError("A2A_LIVE_TRACE_CHAIN_MISMATCH")
        return events, projection

    @staticmethod
    def _terminal_trace_integrity_error() -> A2AContractError:
        return A2AContractError("A2A_TERMINAL_TRACE_INTEGRITY_MISMATCH")

    def _verify_terminal_messages_locked(self, run_id: str, events: tuple[R4TraceEventV1, ...]) -> None:
        messages = self.conn.execute("SELECT * FROM a2a_messages WHERE run_id=? ORDER BY created_at,message_id", (run_id,)).fetchall()
        by_message: dict[str, list[R4TraceEventV1]] = {}
        for event in events:
            if event.message_id is not None:
                by_message.setdefault(str(event.message_id), []).append(event)
            if event.event_type in {"A2A_RESULT_VERIFIED", "A2A_MESSAGE_FAILED"} and event.message_id is None:
                raise self._terminal_trace_integrity_error()
        known_messages = {str(row["message_id"]): row for row in messages}
        for event in events:
            if event.event_type in {"A2A_RESULT_VERIFIED", "A2A_MESSAGE_FAILED"} and str(event.message_id) not in known_messages:
                raise self._terminal_trace_integrity_error()
        for row in messages:
            message_id = str(row["message_id"])
            related = by_message.get(message_id, [])
            terminal_events = [event for event in related if event.event_type in {"A2A_RESULT_VERIFIED", "A2A_MESSAGE_FAILED"}]
            status = str(row["status"])
            if status in {"SUCCEEDED", "FAILED"}:
                expected_event_type = "A2A_RESULT_VERIFIED" if status == "SUCCEEDED" else "A2A_MESSAGE_FAILED"
                if len(terminal_events) != 1 or terminal_events[0].event_type != expected_event_type:
                    raise self._terminal_trace_integrity_error()
                event = terminal_events[0]
                if any(row[key] is None for key in ("canonical_json", "canonical_checksum", "response_trace_id", "response_attempt_id")):
                    raise self._terminal_trace_integrity_error()
                try:
                    canonical = json.loads(str(row["canonical_json"]))
                    specialist = json.loads(str(row["specialist_json"]))
                    payload = event.payload
                    latest = self.conn.execute(
                        "SELECT * FROM a2a_attempts WHERE message_id=? ORDER BY attempt_no DESC LIMIT 1",
                        (message_id,),
                    ).fetchone()
                except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
                    raise self._terminal_trace_integrity_error() from exc
                if not isinstance(canonical, Mapping) or not isinstance(specialist, Mapping) or latest is None:
                    raise self._terminal_trace_integrity_error()
                if (
                    event.trace_id != str(row["response_trace_id"])
                    or event.message_id != message_id
                    or event.attempt_id != str(row["response_attempt_id"])
                    or event.attempt_id != str(canonical.get("attempt_id"))
                    or event.attempt_id != str(latest["attempt_id"])
                    or str(latest["status"]) != status
                    or int(payload.get("attempt_no", -1)) != int(latest["attempt_no"])
                ):
                    raise self._terminal_trace_integrity_error()
                if status == "SUCCEEDED":
                    if (
                        str(payload.get("result_id") or "") != str(canonical.get("result_id") or "")
                        or str(payload.get("payload_hash") or "") != str(specialist.get("payload_hash") or "")
                    ):
                        raise self._terminal_trace_integrity_error()
                elif str(payload.get("error_code") or "") != str(latest["error_code"] or row["error_code"] or ""):
                    raise self._terminal_trace_integrity_error()
            elif terminal_events:
                raise self._terminal_trace_integrity_error()

    def _append_trace_locked(
        self,
        *,
        run_id: str,
        trace_id: str,
        event_type: str,
        scene_clock: datetime,
        payload: Mapping[str, Any],
        message_id: str | None = None,
        attempt_id: str | None = None,
        actor: str = "r4-runtime",
    ) -> R4TraceEventV1:
        own_transaction = not self.conn.in_transaction
        if own_transaction:
            self.conn.execute("BEGIN IMMEDIATE")
        try:
            safe = _redact(dict(payload))
            payload_hash = sha256_json(safe)
            run = self.conn.execute("SELECT status FROM a2a_runs WHERE run_id=?", (run_id,)).fetchone()
            if run is None:
                raise A2AContractError("A2A_RUN_UNREGISTERED")
            if str(run[0]) in {"FROZEN", "CANCELLED"}:
                raise A2AContractError("A2A_TRACE_SEALED")
            events, projection = self._verify_live_chain_locked(run_id)
            seq = len(events) + 1
            normalized_clock = _utc(scene_clock)
            event = R4TraceEventV1(
                seq_no=seq,
                event_id=project_trace_event_id(
                    seq_no=seq,
                    run_id=run_id,
                    trace_id=trace_id,
                    event_type=event_type,
                    message_id=message_id,
                    attempt_id=attempt_id,
                    actor=actor,
                    scene_clock=normalized_clock,
                    payload_hash=payload_hash,
                ),
                run_id=run_id,
                trace_id=trace_id,
                event_type=event_type,
                message_id=message_id,
                attempt_id=attempt_id,
                actor=actor,
                scene_clock=normalized_clock,
                payload=safe,
                payload_hash=payload_hash,
            )
            self.conn.execute(
                "INSERT INTO a2a_trace(run_id,seq_no,event_id,trace_id,event_type,message_id,attempt_id,actor,scene_clock,payload_json,payload_hash) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (run_id, seq, event.event_id, trace_id, event_type, message_id, attempt_id, actor, event.scene_clock.isoformat(), json.dumps(safe, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str), payload_hash),
            )
            new_projection = [*projection, event.model_dump(mode="json")]
            self.conn.execute(
                "UPDATE a2a_runs SET live_event_count=?,live_head=?,live_checksum=?,updated_at=? WHERE run_id=?",
                (len(new_projection), event.event_id, sha256_json(new_projection), self._now(), run_id),
            )
            if own_transaction:
                self.conn.commit()
            return event
        except Exception:
            if own_transaction and self.conn.in_transaction:
                self.conn.rollback()
            raise

    def append_trace(self, *, run_id: str, trace_id: str, event_type: str, scene_clock: datetime, payload: Mapping[str, Any], message_id: str | None = None, attempt_id: str | None = None, actor: str = "r4-runtime") -> R4TraceEventV1:
        with self._lock:
            return self._append_trace_locked(run_id=run_id, trace_id=trace_id, event_type=event_type, scene_clock=scene_clock, payload=payload, message_id=message_id, attempt_id=attempt_id, actor=actor)

    def trace(self, run_id: str) -> tuple[R4TraceEventV1, ...]:
        with self._lock:
            events, _ = self._verify_live_chain_locked(run_id)
            return events

    def record_late(self, *, run_id: str, message_id: str, reason: str, received_at: datetime, trace_id: str | None = None, payload: Mapping[str, Any] | None = None) -> None:
        with self._lock:
            self._record_late_locked(
                run_id=run_id,
                message_id=message_id,
                reason=reason,
                received_at=received_at,
                trace_id=trace_id,
                payload=payload,
            )

    def late_events(self, run_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(row) for row in self.conn.execute("SELECT * FROM a2a_late_events WHERE run_id=? ORDER BY received_at", (run_id,)).fetchall()]


class A2AMessageVerifier:
    """Deterministic envelope and Specialist result verifier."""

    def __init__(self, registry: M3Registry, authority_snapshot: PolicyAuthoritySnapshot | None = None):
        self.registry = registry
        self.authority_snapshot = authority_snapshot

    def _route(self, capability_ref: str) -> _Route:
        route = ROUTES.get(capability_ref)
        if route is None:
            raise A2AContractError("A2A_CAPABILITY_UNKNOWN")
        spec = self.registry.get(route.tool_ref)
        if spec.capability_ref != capability_ref or spec.owner != route.agent_ref.split("@", 1)[0]:
            raise A2AContractError("A2A_CAPABILITY_OWNER_MISMATCH")
        if spec.side_effect != "READ_ONLY" or spec.risk != "READ":
            raise A2AContractError("A2A_WRITE_CAPABILITY_FORBIDDEN")
        return route

    @staticmethod
    def _check_trusted_payload(payload: Mapping[str, Any]) -> None:
        forbidden = TRUSTED_PAYLOAD_KEYS.intersection(str(key) for key in payload)
        if forbidden:
            raise A2AContractError("A2A_TRUSTED_FIELD_OVERRIDE")

    def verify_request(self, envelope: A2AMessageEnvelopeV1, *, now: datetime, expected_context: Mapping[str, str] | None = None) -> _Route:
        if envelope.message_kind != "REQUEST":
            raise A2AContractError("A2A_REQUEST_KIND_REQUIRED")
        if envelope.schema_version != "r4.a2a.message.v1":
            raise A2AContractError("A2A_SCHEMA_VERSION_MISMATCH")
        route = self._route(envelope.capability_ref)
        if envelope.sender_ref not in COORDINATOR_REFS:
            raise A2AContractError("A2A_SENDER_FORBIDDEN")
        if envelope.receiver_ref != route.agent_ref:
            raise A2AContractError("A2A_RECEIVER_MISMATCH")
        if _utc(now) >= envelope.deadline:
            raise A2AContractError("A2A_DEADLINE_EXPIRED")
        if expected_context:
            for key in ("run_id", "plan_revision_id", "task_id"):
                if key in expected_context and getattr(envelope, key) != expected_context[key]:
                    raise A2AContractError("A2A_OWNERSHIP_MISMATCH")
        payload = envelope.payload or {}
        self._check_trusted_payload(payload)
        try:
            self.registry.get(route.tool_ref).validate_candidate_args(payload)
        except (KeyError, ValueError) as exc:
            raise A2AContractError("A2A_REQUEST_SCHEMA_INVALID") from exc
        if envelope.payload_hash != sha256_json(payload):
            raise A2AContractError("A2A_PAYLOAD_HASH_MISMATCH")
        return route

    def verify_specialist_result(self, request: A2AMessageEnvelopeV1, result: TypedAgentResult) -> _Route:
        route = self._route(request.capability_ref)
        if result.agent_ref != route.agent_ref or result.tool_ref != route.tool_ref or result.contract != route.output_contract:
            raise A2AContractError("A2A_RESULT_OWNER_MISMATCH")
        if not result.ok:
            if not result.error_code:
                raise A2AContractError("A2A_RESULT_ERROR_MISSING")
            return route
        if not isinstance(result.payload, Mapping):
            raise A2AContractError("A2A_RESULT_PAYLOAD_INVALID")
        if result.payload_hash != sha256_json(result.payload):
            raise A2AContractError("A2A_RESULT_PAYLOAD_HASH_MISMATCH")
        payload = dict(result.payload)
        self._check_trusted_payload(payload)
        if not str(payload.get("source_version") or ""):
            raise A2AContractError("A2A_RESULT_SOURCE_VERSION_MISSING")
        request_payload = request.payload or {}
        if route.capability_ref == "order/read@v1" and str(payload.get("order_id") or "") != str(request_payload.get("order_id") or ""):
            raise A2AContractError("A2A_SEMANTIC_WRONG")
        if route.capability_ref == "logistics/read@v1":
            for key in ("carrier_code", "tracking_no"):
                if str(payload.get(key) or "") != str(request_payload.get(key) or ""):
                    raise A2AContractError("A2A_SEMANTIC_WRONG")
        if route.capability_ref == "policy/read@v1" and str(payload.get("query") or "") != str(request_payload.get("query") or ""):
            raise A2AContractError("A2A_SEMANTIC_WRONG")
        if route.capability_ref == "policy/read@v1":
            authority = self.authority_snapshot
            if authority is None:
                raise A2AContractError("A2A_POLICY_AUTHORITY_REQUIRED")
            status = str(payload.get("status") or "")
            if status == "FAILED":
                raise A2AContractError("A2A_POLICY_FAILED_STATUS")
            if status not in {"ANSWERED", "WEAK_EVIDENCE", "NO_HITS", "STALE_ONLY", "CONFLICT"}:
                raise A2AContractError("A2A_POLICY_STATUS_INVALID")
            if str(payload.get("query_hash") or "") != sha256_json(str(payload.get("query") or "")):
                raise A2AContractError("A2A_POLICY_QUERY_HASH_MISMATCH")
            if str(payload.get("source") or "") != authority.source:
                raise A2AContractError("A2A_POLICY_SOURCE_UNAUTHORIZED")
            version_tuple = payload.get("source_version_tuple")
            if not isinstance(version_tuple, (list, tuple)) or tuple(str(item) for item in version_tuple) != authority.version_tuple or str(payload.get("source_version") or "") != authority.source_version:
                raise A2AContractError("A2A_POLICY_VERSION_UNAUTHORIZED")
            if str(payload.get("strategy_checksum") or "") != authority.strategy_checksum:
                raise A2AContractError("A2A_POLICY_STRATEGY_CHECKSUM_INVALID")
            if str(payload.get("authority_snapshot_hash") or "") != authority.snapshot_hash:
                raise A2AContractError("A2A_POLICY_AUTHORITY_HASH_MISMATCH")
            evidence = payload.get("evidence")
            claims = payload.get("claims")
            if not isinstance(evidence, list) or not isinstance(claims, list):
                raise A2AContractError("A2A_POLICY_EVIDENCE_INVALID")
            evidence_ids: set[str] = set()
            authority_evidence = authority.evidence_by_id
            for item in evidence:
                if not isinstance(item, Mapping) or not all(str(item.get(key) or "") for key in ("evidence_id", "source_id", "version", "chunk_id", "text_hash", "locator")):
                    raise A2AContractError("A2A_POLICY_EVIDENCE_INVALID")
                evidence_id = str(item["evidence_id"])
                authorized = authority_evidence.get(evidence_id)
                if authorized is None or any(str(item.get(key) or "") != str(authorized[key]) for key in ("source_id", "version", "chunk_id", "text_hash", "locator")):
                    raise A2AContractError("A2A_POLICY_EVIDENCE_UNAUTHORIZED")
                evidence_ids.add(evidence_id)
            if status == "ANSWERED":
                if not evidence_ids or not claims:
                    raise A2AContractError("A2A_POLICY_ANSWERED_WITHOUT_EVIDENCE")
                for claim in claims:
                    if not isinstance(claim, Mapping) or not str(claim.get("claim_id") or "") or not str(claim.get("text") or ""):
                        raise A2AContractError("A2A_POLICY_CLAIM_INVALID")
                    claim_ids = claim.get("evidence_ids")
                    if not isinstance(claim_ids, list) or not claim_ids or not set(map(str, claim_ids)).issubset(evidence_ids):
                        raise A2AContractError("A2A_POLICY_CLAIM_BINDING_INVALID")
            elif claims:
                raise A2AContractError("A2A_POLICY_REFUSAL_WITH_CLAIMS")
        return route

    def verify_result_envelope(self, request: A2AMessageEnvelopeV1, envelope: A2AMessageEnvelopeV1) -> _Route:
        """Verify the result/error envelope before it can release a task."""

        if envelope.schema_version != "r4.a2a.message.v1":
            raise A2AContractError("A2A_SCHEMA_VERSION_MISMATCH")
        if envelope.run_id != request.run_id or envelope.plan_revision_id != request.plan_revision_id or envelope.task_id != request.task_id:
            raise A2AContractError("A2A_OWNERSHIP_MISMATCH")
        if envelope.correlation_id != request.correlation_id or envelope.parent_message_id != request.message_id:
            raise A2AContractError("A2A_CORRELATION_MISMATCH")
        if envelope.dependency_message_ids != request.dependency_message_ids:
            raise A2AContractError("A2A_DEPENDENCY_MISMATCH")
        if envelope.trace_id != request.trace_id or envelope.created_at != request.created_at or envelope.deadline != request.deadline:
            raise A2AContractError("A2A_RESULT_REQUEST_BINDING_MISMATCH")
        if envelope.capability_ref != request.capability_ref or envelope.receiver_ref != "supervisor@v1":
            raise A2AContractError("A2A_RESULT_RECEIVER_MISMATCH")
        route = self._route(request.capability_ref)
        if envelope.message_kind == "ERROR":
            if envelope.sender_ref != route.agent_ref or envelope.error is None:
                raise A2AContractError("A2A_RESULT_OWNER_MISMATCH")
            expected_message_id = f"{request.message_id}:error:{envelope.attempt_id}"
            expected_idempotency_key = f"{request.idempotency_key}:error"
        elif envelope.message_kind == "RESULT":
            if envelope.sender_ref != route.agent_ref:
                raise A2AContractError("A2A_RESULT_OWNER_MISMATCH")
            expected_message_id = f"{request.message_id}:result:{envelope.attempt_id}"
            expected_idempotency_key = f"{request.idempotency_key}:result"
        else:
            raise A2AContractError("A2A_RESULT_KIND_INVALID")
        if envelope.message_id != expected_message_id or envelope.idempotency_key != expected_idempotency_key:
            raise A2AContractError("A2A_RESULT_IDENTITY_MISMATCH")
        if envelope.attempt_id == request.attempt_id:
            raise A2AContractError("A2A_RESULT_ATTEMPT_MISMATCH")
        if envelope.message_kind == "ERROR":
            return route
        if envelope.result_ref is not None:
            raise A2AContractError("A2A_RESULT_REFERENCE_DISABLED")
        if envelope.payload is None:
            raise A2AContractError("A2A_RESULT_KIND_INVALID")
        payload = envelope.payload
        if str(payload.get("agent_ref") or "") != route.agent_ref or str(payload.get("tool_ref") or "") != route.tool_ref or str(payload.get("contract") or "") != route.output_contract:
            raise A2AContractError("A2A_RESULT_OWNER_MISMATCH")
        inner = TypedAgentResult.model_validate(payload)
        return self.verify_specialist_result(request, inner)


class _OrderReadAdapter:
    agent_ref = "order-agent@v1"
    capability_ref = "order/read@v1"

    def __init__(self, registry: M3Registry, db_path: str | Path):
        self.port = OrderAgent(registry=registry)
        self.db_path = str(db_path)

    def invoke(self, payload: Mapping[str, Any], *, context: InvocationContext) -> TypedAgentResult:
        return self.port.invoke(payload, context=context)


class _LogisticsReadAdapter:
    agent_ref = "logistics-agent@v1"
    capability_ref = "logistics/read@v1"

    def __init__(self, registry: M3Registry):
        self.port = LogisticsAgent(registry=registry)

    def invoke(self, payload: Mapping[str, Any], *, context: InvocationContext) -> TypedAgentResult:
        return self.port.invoke(payload, context=context)


class _PolicyRAGReadAdapter:
    agent_ref = "policy-agent@v1"
    capability_ref = "policy/read@v1"

    def __init__(self, registry: M3Registry):
        self.port = PolicyAgent(registry=registry)

    def invoke(self, payload: Mapping[str, Any], *, context: InvocationContext) -> TypedAgentResult:
        return self.port.invoke(payload, context=context)


class PolicyRAGReader:
    """Read-only canonical adapter over an injected R3/R3.5 retriever.

    It intentionally performs retrieval only.  R4-A does not invoke a model;
    a later REAL adapter may wrap R3GroundedQARuntime without changing this
    message contract.
    """

    def __init__(self, *, retriever: Any = None, reader: Callable[[str], Mapping[str, Any]] | None = None, authority_snapshot: Mapping[str, Any] | PolicyAuthoritySnapshot | None = None, as_of: date | None = None, mode: str = "bm25"):
        self.retriever = retriever
        self.reader = reader
        self.as_of = as_of or date(2026, 1, 1)
        self.mode = mode
        if retriever is not None:
            self.authority_snapshot = _authority_from_retriever(retriever)
        elif reader is not None:
            if authority_snapshot is None:
                raise A2AContractError("A2A_POLICY_AUTHORITY_REQUIRED")
            self.authority_snapshot = _authority_from_mapping(authority_snapshot)
        else:
            self.authority_snapshot = _authority_from_mapping(authority_snapshot) if authority_snapshot is not None else None

    def _authority_defaults(self, data: Mapping[str, Any]) -> dict[str, Any]:
        value = dict(data)
        authority = self.authority_snapshot
        if authority is None:
            return value
        value.setdefault("source", authority.source)
        value.setdefault("source_version", authority.source_version)
        value.setdefault("source_version_tuple", list(authority.version_tuple))
        value.setdefault("strategy_checksum", authority.strategy_checksum)
        value.setdefault("authority_snapshot_hash", authority.snapshot_hash)
        return value

    def __call__(self, query: str, *, top_k: int = 5) -> dict[str, Any]:
        if self.reader is not None:
            value = dict(self.reader(query))
            if "success" in value:
                data = value.get("data")
                if value.get("success") and isinstance(data, Mapping):
                    data = self._authority_defaults(data)
                    data.setdefault("query_hash", sha256_json(query))
                    value["data"] = data
                return value
            value = self._authority_defaults(value)
            value.setdefault("query_hash", sha256_json(query))
            return {"success": True, "code": "OK", "data": value}
        if self.retriever is None:
            return {"success": False, "code": "CONFIG_MISSING", "data": None}
        if top_k != 5:
            return {"success": False, "code": "TOOL_CONTRACT_VIOLATION", "data": None}
        try:
            retrieval = self.retriever.retrieve(query, as_of=self.as_of, mode=self.mode, top_k=top_k)
        except Exception:
            return {"success": False, "code": "INFRA_UNAVAILABLE", "data": None}
        evidence = [
            {
                "evidence_id": str(ref.evidence_id),
                "source_id": str(ref.source_id),
                "version": str(ref.version),
                "chunk_id": str(ref.chunk_id),
                "text_hash": str(ref.text_hash),
                "locator": str(ref.locator),
            }
            for ref in getattr(retrieval, "evidence", [])
        ]
        claims = [
            {"claim_id": str(claim.claim_id), "text": str(claim.text), "evidence_ids": list(claim.evidence_ids)}
            for claim in getattr(retrieval, "claims", [])
        ]
        authority = self.authority_snapshot
        if authority is None:
            return {"success": False, "code": "A2A_POLICY_AUTHORITY_REQUIRED", "data": None}
        return {
            "success": True,
            "code": "OK",
            "data": {
                "query": query,
                "query_hash": sha256_json(query),
                "status": str(getattr(retrieval, "status", "FAILED")),
                "source": authority.source,
                "source_version": authority.source_version,
                "source_version_tuple": list(authority.version_tuple),
                "strategy_checksum": authority.strategy_checksum,
                "authority_snapshot_hash": authority.snapshot_hash,
                "evidence": evidence,
                "claims": claims,
                "failure_code": getattr(retrieval, "failure_code", None),
            },
        }


def _order_callable(db_path: str | Path) -> Callable[..., Mapping[str, Any]]:
    def read_order(*, order_id: str, phone_last4: str) -> Mapping[str, Any]:
        try:
            with SQLiteOrderRepository(str(db_path)) as repository:
                row = repository.get_order_for_owner(order_id, phone_last4)
                if row is None:
                    existing = repository.get_order_by_id(order_id)
                    code = "AUTH_IDENTITY_MISMATCH" if existing is not None else "ORDER_NOT_FOUND"
                    return {"success": False, "code": code, "data": None}
        except (OSError, sqlite3.Error, FileNotFoundError):
            return {"success": False, "code": "INFRA_UNAVAILABLE", "data": None}
        safe = {key: value for key, value in row.items() if key not in {"phone_last4"}}
        safe.update({"order_id": order_id, "source": "ecommerce.sqlite.read", "source_version": "ecommerce.sqlite.read.v1"})
        safe["snapshot_hash"] = sha256_json(safe)
        return {"success": True, "code": "OK", "data": safe}

    return read_order


def _logistics_callable(db_path: str | Path, logistics_db_path: str | Path | None) -> Callable[..., Mapping[str, Any]]:
    def read_logistics(*, carrier_code: str, tracking_no: str, phone_last4: str | None = None) -> Mapping[str, Any]:
        if logistics_db_path is None:
            return {"success": False, "code": "INFRA_UNAVAILABLE", "data": None}
        try:
            result = OrderLogisticsRepository(db_path, logistics_db_path=logistics_db_path).read_logistics(carrier_code, tracking_no, phone_last4)
        except (OSError, sqlite3.Error, FileNotFoundError):
            return {"success": False, "code": "INFRA_UNAVAILABLE", "data": None}
        if not result.success:
            return {"success": False, "code": result.code, "data": None}
        data = dict(result.data or {})
        data["source_kind"] = "local_derived_snapshot"
        return {"success": True, "code": "OK", "data": data}

    return read_logistics


def build_r4_registry(*, db_path: str | Path, logistics_db_path: str | Path | None = None, policy_reader: Callable[[str], Mapping[str, Any]] | None = None, policy_retriever: Any = None, policy_authority: Mapping[str, Any] | PolicyAuthoritySnapshot | None = None, policy_as_of: date | None = None, policy_mode: str = "bm25", implementation_mode: str = "SIMULATED") -> M3Registry:
    policy = PolicyRAGReader(retriever=policy_retriever, reader=policy_reader, authority_snapshot=policy_authority, as_of=policy_as_of, mode=policy_mode)
    specs = [
        ToolSpec(tool_ref="order/get_info@v1", capability_ref="order/read@v1", owner="order-agent", args_schema="order.read.v1", result_schema="order.result.v1", risk="READ", implementation_mode=implementation_mode, side_effect="READ_ONLY", timeout_ms=5000, allowed_error_codes=("AUTH_IDENTITY_MISMATCH", "ORDER_NOT_FOUND", "DATA_MISSING", "INFRA_UNAVAILABLE", "INFRA_TIMEOUT"), callable=_order_callable(db_path)),
        ToolSpec(tool_ref="logistics/query@v1", capability_ref="logistics/read@v1", owner="logistics-agent", args_schema="logistics.query.v1", result_schema="logistics.result.v1", risk="READ", implementation_mode=implementation_mode, side_effect="READ_ONLY", timeout_ms=5000, allowed_error_codes=("AUTH_IDENTITY_MISMATCH", "ORDER_NOT_FOUND", "DATA_MISSING", "DATA_STALE", "DATA_CONFLICT", "INFRA_UNAVAILABLE", "INFRA_TIMEOUT"), callable=_logistics_callable(db_path, logistics_db_path)),
        ToolSpec(tool_ref="policy/search@v1", capability_ref="policy/read@v1", owner="policy-agent", args_schema="policy.query.v1", result_schema="policy.result.v1", risk="READ", implementation_mode=implementation_mode, side_effect="READ_ONLY", timeout_ms=5000, allowed_error_codes=("DATA_MISSING", "POLICY_CONFLICT", "INFRA_UNAVAILABLE", "CONFIG_MISSING", "TOOL_CONTRACT_VIOLATION"), callable=lambda *, query, top_k=5: policy(query, top_k=top_k)),
    ]
    return M3Registry(canonical=Registry(specs))


class R4A2ARuntime:
    """Small read-only runtime proving message-level ownership and recovery."""

    def __init__(self, *, db_path: str | Path, logistics_db_path: str | Path | None = None, ledger_path: str | Path = ":memory:", scene_clock: datetime | Callable[[], datetime] | None = None, timeout_seconds: float = 5.0, policy_reader: Callable[[str], Mapping[str, Any]] | None = None, policy_retriever: Any = None, policy_authority: Mapping[str, Any] | PolicyAuthoritySnapshot | None = None, policy_as_of: date | None = None, policy_mode: str = "bm25", failure_script: Mapping[str, Any] | None = None, registry: M3Registry | None = None, retry_budget: int = 2, disabled_capabilities: Iterable[str] = ()):
        self.db_path = str(db_path)
        self.logistics_db_path = str(logistics_db_path) if logistics_db_path is not None else None
        self.timeout_seconds = float(timeout_seconds)
        # Public R4-B ablation hook: the default preserves the R4-A bounded
        # retry contract; ``retry_budget=1`` removes only retry while keeping
        # the A2A ledger, verifier and terminal lifecycle intact.
        if int(retry_budget) not in {1, 2}:
            raise ValueError("retry_budget must be 1 or 2")
        self.retry_budget = int(retry_budget)
        # R4-B's disabled-specialist ablation keeps the A2A verifier, ledger,
        # lifecycle and trace active while withholding one admitted
        # capability from execution.  The default is empty for full R4-A
        # backward compatibility.
        self.disabled_capabilities = frozenset(str(item) for item in disabled_capabilities)
        self._scene_clock = scene_clock
        self.failure_script = dict(failure_script or {})
        if str(ledger_path) != ":memory:" and Path(ledger_path).resolve() == Path(self.db_path).resolve():
            raise ValueError("R4 A2A ledger must be isolated from the order database")
        self.ledger = A2AMessageLedger(ledger_path)
        if policy_retriever is not None:
            self.authority_snapshot = _authority_from_retriever(policy_retriever)
        elif policy_reader is not None:
            if policy_authority is None:
                raise A2AContractError("A2A_POLICY_AUTHORITY_REQUIRED")
            self.authority_snapshot = _authority_from_mapping(policy_authority)
        elif policy_authority is not None:
            self.authority_snapshot = _authority_from_mapping(policy_authority)
        else:
            self.authority_snapshot = None
        self.runtime_version = "r4-a2a.v1" + (f"+authority.{self.authority_snapshot.snapshot_hash}" if self.authority_snapshot else "")
        self.registry = registry or build_r4_registry(db_path=self.db_path, logistics_db_path=self.logistics_db_path, policy_reader=policy_reader, policy_retriever=policy_retriever, policy_authority=self.authority_snapshot, policy_as_of=policy_as_of, policy_mode=policy_mode)
        self.verifier = A2AMessageVerifier(self.registry, authority_snapshot=self.authority_snapshot)
        self.adapters: dict[str, R4SpecialistAdapter] = {
            "order/read@v1": _OrderReadAdapter(self.registry, self.db_path),
            "logistics/read@v1": _LogisticsReadAdapter(self.registry),
            "policy/read@v1": _PolicyRAGReadAdapter(self.registry),
        }
        self._response_cache: dict[str, A2AMessageEnvelopeV1] = {}
        self._result_cache: dict[str, Result] = {}
        self._specialist_cache: dict[str, TypedAgentResult] = {}
        self._fault_counts: dict[tuple[str, str], int] = {}
        self.physical_call_counts: dict[str, int] = {key: 0 for key in ROUTES}
        self._superseded_revisions: set[tuple[str, str]] = set()
        self._state_lock = threading.RLock()

    def now(self) -> datetime:
        value = self._scene_clock() if callable(self._scene_clock) else self._scene_clock
        return _utc(value if value is not None else datetime.now(timezone.utc))

    def close(self) -> None:
        self.ledger.close()

    def start_run(self, *, run_id: str, plan_revision_id: str) -> None:
        """Trusted runtime entry point for run/plan identity binding."""

        self.ledger.ensure_run(run_id, plan_revision_id)

    def _context(self, envelope: A2AMessageEnvelopeV1, route: _Route, attempt_id: str) -> InvocationContext:
        dataset_version = "runtime-read-only" + (f"+authority.{self.authority_snapshot.snapshot_hash}" if self.authority_snapshot else "")
        return InvocationContext(session_id=f"r4-session-{envelope.run_id}", user_id="r4-runtime-user", run_id=envelope.run_id, plan_revision_id=envelope.plan_revision_id, task_id=envelope.task_id, attempt_id=attempt_id, agent_ref=route.agent_ref, auth_scope=route.capability_ref, idempotency_key=envelope.idempotency_key, deadline=envelope.deadline, cancellation=False, config_version=self.runtime_version, registry_version="r4.registry.v1", dataset_version=dataset_version, trace_id=envelope.trace_id)

    def build_request(self, *, run_id: str, plan_revision_id: str, task_id: str, capability_ref: str, payload: Mapping[str, Any], sender_ref: str = "supervisor@v1", message_id: str | None = None, correlation_id: str | None = None, parent_message_id: str | None = None, dependency_message_ids: tuple[str, ...] = (), idempotency_key: str | None = None, created_at: datetime | None = None, deadline: datetime | None = None) -> A2AMessageEnvelopeV1:
        route = ROUTES.get(capability_ref)
        if route is None:
            raise A2AContractError("A2A_CAPABILITY_UNKNOWN")
        logical_key = idempotency_key or f"idem_{run_id}_{task_id}"
        created = _utc(created_at or self.now())
        stored_deadline = self.ledger.registered_deadline(logical_key) if deadline is None else None
        due = _utc(deadline or stored_deadline or (created + timedelta(seconds=self.timeout_seconds)))
        mid = message_id or f"msg_{uuid.uuid4().hex}"
        envelope = A2AMessageEnvelopeV1.request(message_id=mid, correlation_id=correlation_id or f"corr_{run_id}", run_id=run_id, plan_revision_id=plan_revision_id, task_id=task_id, attempt_id=f"attempt_{mid}", trace_id=f"trace_{run_id}", sender_ref=sender_ref, receiver_ref=route.agent_ref, capability_ref=capability_ref, created_at=created, deadline=due, idempotency_key=logical_key, payload=dict(payload), parent_message_id=parent_message_id, dependency_message_ids=dependency_message_ids)
        self.start_run(run_id=run_id, plan_revision_id=plan_revision_id)
        self.ledger.register_request(envelope)
        return envelope

    def _trace(self, envelope: A2AMessageEnvelopeV1, event_type: str, payload: Mapping[str, Any], *, attempt_id: str | None = None, actor: str = "r4-runtime") -> R4TraceEventV1:
        trace_payload = self._trace_payload(payload)
        return self.ledger.append_trace(run_id=envelope.run_id, trace_id=envelope.trace_id, event_type=event_type, scene_clock=self.now(), payload=trace_payload, message_id=envelope.message_id, attempt_id=attempt_id, actor=actor)

    def _trace_payload(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        trace_payload = dict(payload)
        if self.authority_snapshot is not None:
            trace_payload.setdefault("authority_snapshot_hash", self.authority_snapshot.snapshot_hash)
            trace_payload.setdefault("authority_strategy_checksum", self.authority_snapshot.strategy_checksum)
        return trace_payload

    def _trace_nonterminal(self, envelope: A2AMessageEnvelopeV1, event_type: str, payload: Mapping[str, Any], *, attempt_id: str | None = None, actor: str = "r4-runtime") -> R4TraceEventV1 | None:
        """Record pre-finalization diagnostics when the run is still open.

        Terminal result tracing is owned by ``finalize_terminal`` and never
        passes through this best-effort path.
        """

        status, current_plan, superseded = self.ledger.lifecycle(envelope.run_id, envelope.plan_revision_id)
        if status in {"FROZEN", "CANCELLED"} or superseded or (current_plan is not None and current_plan != envelope.plan_revision_id):
            return None
        try:
            return self._trace(envelope, event_type, payload, attempt_id=attempt_id, actor=actor)
        except A2AContractError as exc:
            if exc.code == "A2A_TRACE_SEALED":
                return None
            raise

    def _dependency_state(self, envelope: A2AMessageEnvelopeV1) -> str | None:
        if not envelope.dependency_message_ids:
            return None
        states = []
        for dependency in envelope.dependency_message_ids:
            row = self.ledger.row(dependency)
            if row is None:
                return "MISMATCH"
            for key in ("run_id", "plan_revision_id", "correlation_id"):
                if str(row[key]) != str(getattr(envelope, key)):
                    return "MISMATCH"
            states.append(str(row["status"]))
        if any(state in {"FAILED", "BLOCKED", "CANCELLED", "LATE", "REJECTED"} for state in states):
            return "BLOCKED"
        if not all(state == "SUCCEEDED" for state in states):
            return "PENDING"
        return None

    def _late_reason(self, envelope: A2AMessageEnvelopeV1) -> str:
        run_status, current_plan, superseded = self.ledger.lifecycle(envelope.run_id, envelope.plan_revision_id)
        if run_status == "CANCELLED":
            return "RUN_CANCELLED"
        if run_status == "FROZEN":
            return "RUN_TERMINAL"
        if superseded or (current_plan is not None and current_plan != envelope.plan_revision_id):
            return "REVISION_SUPERSEDED"
        return "DEADLINE_EXPIRED"

    def _fault_rule(self, envelope: A2AMessageEnvelopeV1, attempt_no: int) -> str | None:
        keys = (envelope.message_id, envelope.task_id, envelope.capability_ref, envelope.receiver_ref)
        rule: Any = None
        for key in keys:
            if key in self.failure_script:
                rule = self.failure_script[key]
                break
        if rule is None:
            return None
        if isinstance(rule, str):
            kind = rule
            limit = 1
        else:
            kind = str(rule.get("kind") or rule.get("code") or "")
            limit = int(rule.get("count", 1))
        seen = self._fault_counts.get((envelope.message_id, kind), 0)
        if seen >= limit or attempt_no > limit:
            return None
        self._fault_counts[(envelope.message_id, kind)] = seen + 1
        return kind.upper()

    def _mutate_semantic_result(self, result: TypedAgentResult) -> TypedAgentResult:
        if not isinstance(result.payload, Mapping):
            return result
        value = dict(result.payload)
        if "order_id" in value:
            value["order_id"] = "semantic-mismatch"
        elif "tracking_no" in value:
            value["tracking_no"] = "semantic-mismatch"
        elif "query" in value:
            value["query"] = "semantic-mismatch"
        else:
            value["source_version"] = "semantic-mismatch"
        return result.model_copy(update={"payload": value, "payload_hash": sha256_json(value)})

    @staticmethod
    def _safe_specialist_result(result: TypedAgentResult) -> TypedAgentResult:
        if not result.ok:
            return result
        payload = _redact(result.payload)
        return result.model_copy(update={"payload": payload, "payload_hash": sha256_json(payload)})

    def _canonical_result(self, envelope: A2AMessageEnvelopeV1, attempt_id: str, specialist: TypedAgentResult, *, status: ResultStatus, error_code: str | None = None, physical_attempts: int = 0) -> Result:
        payload = dict(specialist.payload) if specialist.ok and isinstance(specialist.payload, Mapping) else None
        evidence_refs = [str(item["evidence_id"]) for item in (payload or {}).get("evidence", []) if isinstance(item, Mapping) and item.get("evidence_id")] if payload else []
        return Result(result_id=f"result_{uuid.uuid4().hex}", run_id=envelope.run_id, plan_revision_id=envelope.plan_revision_id, task_id=envelope.task_id, attempt_id=attempt_id, status=status, output_contract=specialist.contract, payload=payload, business_code="OK" if specialist.ok else None, error_ref=error_code, evidence_refs=evidence_refs, usage=Usage(logical_calls=1, physical_attempts=physical_attempts))

    def _duplicate_result(self, envelope: A2AMessageEnvelopeV1, row: sqlite3.Row) -> A2ADispatchResult:
        original_id = str(row["message_id"])
        # Process-local caches are deliberately ignored.  A duplicate is a
        # fresh rehydration from the durable ledger on every call.
        response, specialist, canonical, duplicate_of = self.ledger.load_cached(row, request=envelope, verifier=self.verifier)
        error_code = row["error_code"]
        attempts = len(self.ledger.attempts(original_id))
        physical = sum(1 for attempt in self.ledger.attempts(original_id) if int(attempt["physical_call"]))
        return A2ADispatchResult(status=str(row["status"]), request=envelope, response=response, specialist_result=specialist, canonical_result=canonical, error_code=error_code, attempt_count=attempts, physical_call_count=physical, duplicate=True, duplicate_of=duplicate_of)

    def dispatch(self, envelope: A2AMessageEnvelopeV1) -> A2ADispatchResult:
        """Validate and dispatch one request with one bounded retry."""

        with self._state_lock:
            return self._dispatch(envelope)

    def _dispatch(self, envelope: A2AMessageEnvelopeV1) -> A2ADispatchResult:
        """Locked dispatcher implementation."""

        run_status, current_plan, superseded = self.ledger.lifecycle(envelope.run_id, envelope.plan_revision_id)
        if run_status is None:
            return A2ADispatchResult(status="FAILED", request=envelope, error_code="A2A_RUN_UNREGISTERED")
        if current_plan != envelope.plan_revision_id and not superseded:
            return A2ADispatchResult(status="FAILED", request=envelope, error_code="A2A_PLAN_NOT_CURRENT")
        now = self.now()
        if run_status in {"CANCELLED", "FROZEN"} or superseded or now >= envelope.deadline:
            reason = self._late_reason(envelope)
            self.ledger.record_late(run_id=envelope.run_id, message_id=envelope.message_id, reason=reason, received_at=now, trace_id=envelope.trace_id, payload={"reason": reason, "message_hash": envelope.payload_hash})
            return A2ADispatchResult(status="LATE", request=envelope, error_code="A2A_LATE_MESSAGE", late=True)
        try:
            route = self.verifier.verify_request(envelope, now=now, expected_context={"run_id": envelope.run_id, "plan_revision_id": envelope.plan_revision_id, "task_id": envelope.task_id})
        except A2AContractError as exc:
            self._trace_nonterminal(envelope, "A2A_MESSAGE_REJECTED", {"code": exc.code, "message_hash": envelope.payload_hash})
            return A2ADispatchResult(status="FAILED", request=envelope, error_code=exc.code)
        registration, registered_row = self.ledger.validate_registered(envelope)
        if registration == "UNREGISTERED":
            return A2ADispatchResult(status="FAILED", request=envelope, error_code="A2A_MESSAGE_UNREGISTERED")
        if registration == "CONFLICT":
            if registered_row is not None and str(registered_row["idempotency_key"]) == envelope.idempotency_key:
                return A2ADispatchResult(status="FAILED", request=envelope, error_code="A2A_IDEMPOTENCY_CONFLICT")
            return A2ADispatchResult(status="FAILED", request=envelope, error_code="A2A_REGISTERED_REQUEST_MISMATCH")
        dependency = self._dependency_state(envelope)
        if dependency == "MISMATCH":
            self.ledger.set_message_status(envelope.message_id, "REJECTED", error_code="A2A_DEPENDENCY_MISMATCH")
            self._trace_nonterminal(envelope, "A2A_DEPENDENCY_REJECTED", {"dependencies": list(envelope.dependency_message_ids), "code": "A2A_DEPENDENCY_MISMATCH"})
            return A2ADispatchResult(status="FAILED", request=envelope, error_code="A2A_DEPENDENCY_MISMATCH")
        if dependency == "PENDING":
            reservation, _ = self.ledger.reserve(envelope)
            if reservation == "CONFLICT":
                self._trace_nonterminal(envelope, "A2A_IDEMPOTENCY_CONFLICT", {"message_hash": envelope.payload_hash})
                return A2ADispatchResult(status="FAILED", request=envelope, error_code="A2A_IDEMPOTENCY_CONFLICT")
            if reservation == "RUN_UNREGISTERED":
                return A2ADispatchResult(status="FAILED", request=envelope, error_code="A2A_MESSAGE_UNREGISTERED")
            if reservation == "PLAN_NOT_CURRENT":
                return A2ADispatchResult(status="FAILED", request=envelope, error_code="A2A_PLAN_NOT_CURRENT")
            if reservation == "LATE":
                reason = self._late_reason(envelope)
                self.ledger.record_late(run_id=envelope.run_id, message_id=envelope.message_id, reason=reason, received_at=self.now(), trace_id=envelope.trace_id, payload={"reason": reason, "message_hash": envelope.payload_hash})
                return A2ADispatchResult(status="LATE", request=envelope, error_code="A2A_LATE_MESSAGE", late=True)
            if reservation == "UNREGISTERED":
                return A2ADispatchResult(status="FAILED", request=envelope, error_code="A2A_MESSAGE_UNREGISTERED")
            self.ledger.set_message_status(envelope.message_id, "PENDING")
            self._trace_nonterminal(envelope, "A2A_MESSAGE_PENDING", {"dependencies": list(envelope.dependency_message_ids)})
            return A2ADispatchResult(status="PENDING", request=envelope, pending=True)
        if dependency == "BLOCKED":
            reservation, _ = self.ledger.reserve(envelope)
            if reservation == "CONFLICT":
                self._trace_nonterminal(envelope, "A2A_IDEMPOTENCY_CONFLICT", {"message_hash": envelope.payload_hash})
                return A2ADispatchResult(status="FAILED", request=envelope, error_code="A2A_IDEMPOTENCY_CONFLICT")
            if reservation == "RUN_UNREGISTERED":
                return A2ADispatchResult(status="FAILED", request=envelope, error_code="A2A_MESSAGE_UNREGISTERED")
            if reservation == "PLAN_NOT_CURRENT":
                return A2ADispatchResult(status="FAILED", request=envelope, error_code="A2A_PLAN_NOT_CURRENT")
            if reservation == "LATE":
                reason = self._late_reason(envelope)
                self.ledger.record_late(run_id=envelope.run_id, message_id=envelope.message_id, reason=reason, received_at=self.now(), trace_id=envelope.trace_id, payload={"reason": reason, "message_hash": envelope.payload_hash})
                return A2ADispatchResult(status="LATE", request=envelope, error_code="A2A_LATE_MESSAGE", late=True)
            if reservation == "UNREGISTERED":
                return A2ADispatchResult(status="FAILED", request=envelope, error_code="A2A_MESSAGE_UNREGISTERED")
            self.ledger.set_message_status(envelope.message_id, "BLOCKED", error_code="A2A_DEPENDENCY_BLOCKED")
            self._trace_nonterminal(envelope, "A2A_MESSAGE_BLOCKED", {"dependencies": list(envelope.dependency_message_ids)})
            return A2ADispatchResult(status="BLOCKED", request=envelope, error_code="A2A_DEPENDENCY_BLOCKED", blocked=True)
        reservation, row = self.ledger.reserve(envelope)
        if reservation == "CONFLICT":
            self._trace_nonterminal(envelope, "A2A_IDEMPOTENCY_CONFLICT", {"message_hash": envelope.payload_hash})
            return A2ADispatchResult(status="FAILED", request=envelope, error_code="A2A_IDEMPOTENCY_CONFLICT")
        if reservation == "RUN_UNREGISTERED":
            return A2ADispatchResult(status="FAILED", request=envelope, error_code="A2A_MESSAGE_UNREGISTERED")
        if reservation == "PLAN_NOT_CURRENT":
            return A2ADispatchResult(status="FAILED", request=envelope, error_code="A2A_PLAN_NOT_CURRENT")
        if reservation == "LATE":
            reason = self._late_reason(envelope)
            self.ledger.record_late(run_id=envelope.run_id, message_id=envelope.message_id, reason=reason, received_at=self.now(), trace_id=envelope.trace_id, payload={"reason": reason, "message_hash": envelope.payload_hash})
            return A2ADispatchResult(status="LATE", request=envelope, error_code="A2A_LATE_MESSAGE", late=True)
        if reservation == "UNREGISTERED":
            return A2ADispatchResult(status="FAILED", request=envelope, error_code="A2A_MESSAGE_UNREGISTERED")
        if reservation in {"DUPLICATE", "PENDING"} and row is not None:
            if reservation == "PENDING":
                self._trace_nonterminal(envelope, "A2A_DUPLICATE", {"cached_status": str(row["status"]), "message_hash": envelope.payload_hash})
                return A2ADispatchResult(status="PENDING", request=envelope, pending=True, duplicate=True)
            try:
                result = self._duplicate_result(envelope, row)
            except A2AContractError as exc:
                self.ledger.set_message_status(str(row["message_id"]), "CORRUPT", error_code=exc.code)
                try:
                    self._trace_nonterminal(envelope, "A2A_PERSISTED_RESULT_REJECTED", {"code": exc.code, "duplicate_of": str(row["message_id"])})
                except A2AContractError as trace_exc:
                    if not (trace_exc.code.startswith("A2A_TRACE_") or trace_exc.code == "A2A_LIVE_TRACE_CHAIN_MISMATCH"):
                        raise
                return A2ADispatchResult(status="FAILED", request=envelope, error_code=exc.code, duplicate=True, duplicate_of=str(row["message_id"]))
            self._trace_nonterminal(envelope, "A2A_DUPLICATE", {"cached_status": str(row["status"]), "message_hash": envelope.payload_hash})
            return result
        self._trace_nonterminal(envelope, "A2A_REQUEST_ACCEPTED", {"capability_ref": route.capability_ref, "message_hash": envelope.payload_hash})
        adapter = self.adapters[route.capability_ref]
        last_error: str | None = None
        for attempt_no in range(1, self.retry_budget + 1):
            attempt_id = f"{envelope.message_id}:attempt:{attempt_no}"
            self.ledger.record_attempt(message_id=envelope.message_id, attempt_id=attempt_id, attempt_no=attempt_no, status="RUNNING", physical_call=False, scene_clock=self.now())
            self._trace_nonterminal(envelope, "A2A_ATTEMPT_STARTED", {"attempt_no": attempt_no, "capability_ref": route.capability_ref}, attempt_id=attempt_id)
            fault = self._fault_rule(envelope, attempt_no)
            physical = False
            specialist: TypedAgentResult | None = None
            if route.capability_ref in self.disabled_capabilities:
                # Keep the request in the normal A2A lifecycle, but do not
                # invoke the disabled adapter and do not count a physical
                # call.  The common terminal finalizer below records the
                # stable failure envelope/canonical result and trace.
                last_error = "SPECIALIST_DISABLED"
                specialist = TypedAgentResult(
                    contract=route.output_contract,
                    ok=False,
                    payload=None,
                    payload_hash=sha256_json({"error_code": last_error}),
                    error_code=last_error,
                    agent_ref=route.agent_ref,
                    tool_ref=route.tool_ref,
                )
                self.ledger.record_attempt(
                    message_id=envelope.message_id,
                    attempt_id=attempt_id,
                    attempt_no=attempt_no,
                    status="FAILED",
                    physical_call=False,
                    scene_clock=self.now(),
                    error_code=last_error,
                    ended=True,
                )
                self._trace_nonterminal(
                    envelope,
                    "A2A_ATTEMPT_FAILED",
                    {"attempt_no": attempt_no, "error_code": last_error, "physical_call": False},
                    attempt_id=attempt_id,
                )
            elif fault == "NON_RETRYABLE":
                # A generic non-retryable protocol fault is intentionally a
                # separate family from semantic mismatch.  It does not call
                # the specialist and therefore cannot be retried by the
                # bounded retry loop.
                last_error = "NON_RETRYABLE_FAULT"
                specialist = TypedAgentResult(
                    contract=route.output_contract,
                    ok=False,
                    payload=None,
                    payload_hash=sha256_json({"error_code": last_error}),
                    error_code=last_error,
                    agent_ref=route.agent_ref,
                    tool_ref=route.tool_ref,
                )
                self.ledger.record_attempt(
                    message_id=envelope.message_id,
                    attempt_id=attempt_id,
                    attempt_no=attempt_no,
                    status="FAILED",
                    physical_call=False,
                    scene_clock=self.now(),
                    error_code=last_error,
                    ended=True,
                )
                self._trace_nonterminal(
                    envelope,
                    "A2A_ATTEMPT_FAILED",
                    {"attempt_no": attempt_no, "error_code": last_error, "physical_call": False},
                    attempt_id=attempt_id,
                )
            elif fault in {"LOST", "TIMEOUT"}:
                error_code = "INFRA_UNAVAILABLE" if fault == "LOST" else "INFRA_TIMEOUT"
                last_error = error_code
                self.ledger.record_attempt(message_id=envelope.message_id, attempt_id=attempt_id, attempt_no=attempt_no, status="FAILED", physical_call=False, scene_clock=self.now(), error_code=error_code, ended=True)
                self._trace_nonterminal(envelope, "A2A_ATTEMPT_FAILED", {"attempt_no": attempt_no, "error_code": error_code, "physical_call": False}, attempt_id=attempt_id)
            else:
                self.physical_call_counts[route.capability_ref] = self.physical_call_counts.get(route.capability_ref, 0) + 1
                physical = True
                try:
                    specialist = adapter.invoke(envelope.payload or {}, context=self._context(envelope, route, attempt_id))
                    if fault in PHYSICAL_RETRY_FAULTS:
                        last_error = PHYSICAL_RETRY_FAULTS[fault]
                        specialist = TypedAgentResult(contract=route.output_contract, ok=False, payload=None, payload_hash=sha256_json({"error_code": last_error}), error_code=last_error, agent_ref=route.agent_ref, tool_ref=route.tool_ref)
                    elif fault == "SEMANTIC_WRONG":
                        specialist = self._mutate_semantic_result(specialist)
                    if fault not in PHYSICAL_RETRY_FAULTS:
                        self.verifier.verify_specialist_result(envelope, specialist)
                    if not specialist.ok:
                        last_error = specialist.error_code or "TOOL_EXECUTION_FAILED"
                        status = "FAILED"
                    else:
                        status = "SUCCEEDED"
                        last_error = None
                    self.ledger.record_attempt(message_id=envelope.message_id, attempt_id=attempt_id, attempt_no=attempt_no, status=status, physical_call=physical, scene_clock=self.now(), error_code=last_error, ended=True)
                    self._trace_nonterminal(envelope, "A2A_ATTEMPT_RETURNED", {"attempt_no": attempt_no, "ok": specialist.ok, "error_code": last_error, "physical_call": physical}, attempt_id=attempt_id)
                except A2AContractError as exc:
                    last_error = exc.code
                    self.ledger.record_attempt(message_id=envelope.message_id, attempt_id=attempt_id, attempt_no=attempt_no, status="FAILED", physical_call=physical, scene_clock=self.now(), error_code=last_error, ended=True)
                    self._trace_nonterminal(envelope, "A2A_SEMANTIC_REJECTED", {"attempt_no": attempt_no, "error_code": last_error, "physical_call": physical}, attempt_id=attempt_id)
                    # Never pass a semantically rejected payload downstream,
                    # even when its original shape was schema-valid.
                    specialist = TypedAgentResult(contract=route.output_contract, ok=False, payload=None, payload_hash=sha256_json({"error_code": last_error}), error_code=last_error, agent_ref=route.agent_ref, tool_ref=route.tool_ref)
                except Exception as exc:
                    last_error = "TOOL_EXECUTION_FAILED"
                    self.ledger.record_attempt(message_id=envelope.message_id, attempt_id=attempt_id, attempt_no=attempt_no, status="FAILED", physical_call=physical, scene_clock=self.now(), error_code=last_error, ended=True)
                    self._trace_nonterminal(envelope, "A2A_ATTEMPT_FAILED", {"attempt_no": attempt_no, "error_code": last_error, "exception_type": type(exc).__name__, "physical_call": physical}, attempt_id=attempt_id)
                    specialist = TypedAgentResult(contract=route.output_contract, ok=False, payload=None, payload_hash=sha256_json({"error_code": last_error}), error_code=last_error, agent_ref=route.agent_ref, tool_ref=route.tool_ref)
            if specialist is not None and specialist.ok:
                specialist = self._safe_specialist_result(specialist)
                physical_attempts = sum(1 for attempt in self.ledger.attempts(envelope.message_id) if int(attempt["physical_call"]))
                response = A2AMessageEnvelopeV1.result(request=envelope, sender_ref=route.agent_ref, receiver_ref="supervisor@v1", attempt_id=attempt_id, payload={"contract": specialist.contract, "ok": True, "payload": specialist.payload, "payload_hash": specialist.payload_hash, "agent_ref": specialist.agent_ref, "tool_ref": specialist.tool_ref})
                self.verifier.verify_result_envelope(envelope, response)
                canonical = self._canonical_result(envelope, attempt_id, specialist, status=ResultStatus.SUCCEEDED, physical_attempts=physical_attempts)
                self._response_cache[envelope.message_id] = response
                self._specialist_cache[envelope.message_id] = specialist
                self._result_cache[envelope.message_id] = canonical
                persisted = self.ledger.finalize_terminal(
                    envelope.message_id,
                    response,
                    specialist=specialist,
                    canonical=canonical,
                    status="SUCCEEDED",
                    received_at=self.now(),
                    scene_clock=self.now(),
                    terminal_event_type="A2A_RESULT_VERIFIED",
                    terminal_event_payload=self._trace_payload({"attempt_no": attempt_no, "result_id": canonical.result_id, "payload_hash": specialist.payload_hash}),
                )
                if persisted == "LATE":
                    self._response_cache.pop(envelope.message_id, None)
                    self._specialist_cache.pop(envelope.message_id, None)
                    self._result_cache.pop(envelope.message_id, None)
                    return A2ADispatchResult(status="LATE", request=envelope, error_code="A2A_LATE_MESSAGE", late=True)
                return A2ADispatchResult(status="SUCCEEDED", request=envelope, response=response, specialist_result=specialist, canonical_result=canonical, attempt_count=attempt_no, physical_call_count=physical_attempts)
            if last_error not in RETRYABLE_A2A_CODES or attempt_no == self.retry_budget:
                break
            lifecycle_status, lifecycle_plan, lifecycle_superseded = self.ledger.lifecycle(envelope.run_id, envelope.plan_revision_id)
            if (
                lifecycle_status in {"CANCELLED", "FROZEN"}
                or lifecycle_superseded
                or (lifecycle_plan is not None and lifecycle_plan != envelope.plan_revision_id)
                or self.now() >= envelope.deadline
            ):
                break
            self._trace_nonterminal(envelope, "A2A_BOUNDED_RETRY", {"attempt_no": attempt_no, "error_code": last_error})
        error = A2AErrorEnvelopeV1(code=last_error or "TOOL_EXECUTION_FAILED", message_key=(last_error or "TOOL_EXECUTION_FAILED").lower(), retryable=False, details={"attempts": len(self.ledger.attempts(envelope.message_id))})
        response = A2AMessageEnvelopeV1.error_message(request=envelope, sender_ref=route.agent_ref, receiver_ref="supervisor@v1", attempt_id=f"{envelope.message_id}:attempt:{len(self.ledger.attempts(envelope.message_id)) or 1}", error=error)
        self.verifier.verify_result_envelope(envelope, response)
        specialist = specialist or TypedAgentResult(contract=route.output_contract, ok=False, payload=None, payload_hash=sha256_json({"error_code": last_error or "TOOL_EXECUTION_FAILED"}), error_code=last_error or "TOOL_EXECUTION_FAILED", agent_ref=route.agent_ref, tool_ref=route.tool_ref)
        attempt_rows = self.ledger.attempts(envelope.message_id)
        physical_attempts = sum(1 for attempt in attempt_rows if int(attempt["physical_call"]))
        canonical_attempt_id = str(attempt_rows[-1]["attempt_id"]) if attempt_rows else response.attempt_id
        canonical = self._canonical_result(envelope, canonical_attempt_id, specialist, status=ResultStatus.FAILED, error_code=last_error or "TOOL_EXECUTION_FAILED", physical_attempts=physical_attempts)
        self._response_cache[envelope.message_id] = response
        self._specialist_cache[envelope.message_id] = specialist
        self._result_cache[envelope.message_id] = canonical
        persisted = self.ledger.finalize_terminal(
            envelope.message_id,
            response,
            specialist=specialist,
            canonical=canonical,
            status="FAILED",
            error_code=last_error or "TOOL_EXECUTION_FAILED",
            received_at=self.now(),
            scene_clock=self.now(),
            terminal_event_type="A2A_MESSAGE_FAILED",
            terminal_event_payload=self._trace_payload({"error_code": last_error or "TOOL_EXECUTION_FAILED", "attempts": len(self.ledger.attempts(envelope.message_id))}),
        )
        if persisted == "LATE":
            self._response_cache.pop(envelope.message_id, None)
            self._specialist_cache.pop(envelope.message_id, None)
            self._result_cache.pop(envelope.message_id, None)
            return A2ADispatchResult(status="LATE", request=envelope, error_code="A2A_LATE_MESSAGE", late=True)
        return A2ADispatchResult(status="FAILED", request=envelope, response=response, specialist_result=specialist, canonical_result=canonical, error_code=last_error or "TOOL_EXECUTION_FAILED", attempt_count=len(self.ledger.attempts(envelope.message_id)), physical_call_count=sum(1 for row in self.ledger.attempts(envelope.message_id) if int(row["physical_call"])))

    send = dispatch
    dispatch_message = dispatch

    def cancel(self, run_id: str) -> None:
        """Close a run under the same state boundary used by dispatch."""

        with self._state_lock:
            self.ledger.cancel_run(run_id)

    def freeze(self, run_id: str) -> dict[str, Any]:
        """Atomically seal the current main trace and persist its checksum.

        Late deliveries are written by :meth:`dispatch` to the independent
        late-event table.  They therefore cannot change the projection or
        checksum returned here, including when ``freeze`` is called again.
        """

        with self._state_lock:
            return self.ledger.freeze_if_idle(run_id)

    def supersede_revision(self, run_id: str, plan_revision_id: str, new_plan_revision_id: str | None = None) -> None:
        """Mark an old revision late and optionally bind a new trusted plan."""

        with self._state_lock:
            current = self.ledger.run_plan(run_id)
            if current is None or current != plan_revision_id:
                raise A2AContractError("A2A_SUPERSEDE_BINDING_MISMATCH")
            if new_plan_revision_id is not None:
                self.ledger.switch_plan(run_id, plan_revision_id, new_plan_revision_id)
            else:
                self.ledger.mark_superseded(run_id, plan_revision_id)

    def trace(self, run_id: str) -> tuple[R4TraceEventV1, ...]:
        return self.ledger.trace(run_id)

    def run(self, *, topology: str, order_id: str | None = None, phone_last4: str | None = None, carrier_code: str | None = None, tracking_no: str | None = None, policy_query: str | None = None, run_id: str | None = None, plan_revision_id: str | None = None) -> R4RunResult:
        """Execute one explicit read-only topology; no model-generated routing."""

        normalized = topology.lower().replace(" ", "").replace("→", "->")
        run = run_id or f"run_{uuid.uuid4().hex}"
        plan = plan_revision_id or f"plan_{run}"
        self.ledger.ensure_run(run, plan)
        dispatches: dict[str, A2ADispatchResult] = {}
        canonical: dict[str, Result] = {}

        def submit(task_id: str, capability: str, payload: Mapping[str, Any], deps: tuple[str, ...] = ()) -> A2ADispatchResult:
            request = self.build_request(run_id=run, plan_revision_id=plan, task_id=task_id, capability_ref=capability, payload=payload, dependency_message_ids=deps)
            outcome = self.dispatch(request)
            dispatches[task_id] = outcome
            if outcome.canonical_result is not None:
                canonical[task_id] = outcome.canonical_result
            return outcome

        order_outcome: A2ADispatchResult | None = None
        if normalized in {"order_only", "order", "order->logistics", "order+policy", "order->logistics+policy", "order+logistics+policy", "order->logistics+policy"}:
            if not order_id or not phone_last4:
                raise ValueError("order topology requires order_id and phone_last4")
            order_outcome = submit("order", "order/read@v1", {"order_id": order_id, "phone_last4": phone_last4})
        if normalized in {"logistics_only", "logistics", "order->logistics", "order->logistics+policy", "order+logistics+policy"}:
            order_payload = order_outcome.specialist_result.payload if order_outcome and order_outcome.specialist_result and order_outcome.specialist_result.ok else {}
            carrier = carrier_code or (str(order_payload.get("carrier_code")) if isinstance(order_payload, Mapping) else "")
            tracking = tracking_no or (str(order_payload.get("tracking_no")) if isinstance(order_payload, Mapping) else "")
            if not carrier or not tracking:
                if order_outcome is not None and order_outcome.status != "SUCCEEDED":
                    request = self.build_request(run_id=run, plan_revision_id=plan, task_id="logistics", capability_ref="logistics/read@v1", payload={"carrier_code": carrier or "missing", "tracking_no": tracking or "missing", **({"phone_last4": phone_last4} if phone_last4 else {})}, dependency_message_ids=(order_outcome.request.message_id,))
                    dispatches["logistics"] = self.dispatch(request)
                else:
                    raise ValueError("logistics topology requires carrier_code and tracking_no")
            else:
                logistics_payload: dict[str, Any] = {"carrier_code": carrier, "tracking_no": tracking}
                if phone_last4:
                    logistics_payload["phone_last4"] = phone_last4
                deps = (order_outcome.request.message_id,) if order_outcome is not None else ()
                submit("logistics", "logistics/read@v1", logistics_payload, deps)
        if normalized in {"order+policy", "order->logistics+policy", "order+logistics+policy", "policy_only", "policy"}:
            if not policy_query:
                raise ValueError("policy topology requires policy_query")
            submit("policy", "policy/read@v1", {"query": policy_query, "top_k": 5})
        if not dispatches:
            raise ValueError(f"unsupported R4 topology: {topology}")
        values = list(dispatches.values())
        if all(item.status == "SUCCEEDED" for item in values):
            status = "SUCCEEDED"
        elif any(item.status == "SUCCEEDED" for item in values):
            status = "PARTIAL"
        elif any(item.status == "PENDING" for item in values):
            status = "BLOCKED"
        else:
            status = "FAILED"
        answer_parts: list[str] = []
        for task_id, result in dispatches.items():
            if result.specialist_result is not None and result.specialist_result.ok:
                payload = result.specialist_result.payload or {}
                if task_id == "policy":
                    answer_parts.append(f"policy:{payload.get('status', 'VERIFIED')}")
                else:
                    answer_parts.append(f"{task_id}:verified")
            elif result.error_code:
                answer_parts.append(f"{task_id}:{result.error_code}")
        return R4RunResult(run_id=run, plan_revision_id=plan, topology=topology, status=status, answer="; ".join(answer_parts), dispatches=dispatches, canonical_results=canonical, trace=self.trace(run))


__all__ = [
    "A2ADispatchResult",
    "A2AMessageLedger",
    "A2AMessageVerifier",
    "A2A_TERMINAL_MESSAGE_STATUSES",
    "PolicyAuthoritySnapshot",
    "PolicyRAGReader",
    "R4A2ARuntime",
    "R4RunResult",
    "R4TraceEventV1",
    "build_r4_registry",
]
