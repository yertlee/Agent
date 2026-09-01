from __future__ import annotations

import json
from typing import Optional

from agent.domain.objects import canonical_json
from agent.storage.repositories import M1Repository, _now
from .events import TraceEvent, sensitive_surface_scan


class OutboxWriter:
    """Append-only event writer with per-run sequence allocation."""
    def __init__(self, repository: M1Repository): self.repository = repository

    def append(self, event: TraceEvent) -> TraceEvent:
        if sensitive_surface_scan(event.payload):
            raise ValueError("sensitive raw field in trace payload")
        conn = self.repository.conn
        # Sequence allocation and insert share one transaction/lock.
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT next_seq_no FROM runs WHERE run_id=?", (event.run_id,)).fetchone()
        if not row: conn.rollback(); raise KeyError(event.run_id)
        session = conn.execute("SELECT session_id FROM runs WHERE run_id=?", (event.run_id,)).fetchone()[0]
        if str(session) != event.session_id:
            conn.rollback(); raise ValueError("trace event session does not match run")
        expected = int(row[0])
        if event.seq_no != expected:
            conn.rollback(); raise ValueError(f"seq_no must be {expected}")
        self.repository._append_event_locked(event)
        conn.execute("UPDATE runs SET next_seq_no=?,updated_at=? WHERE run_id=?", (expected + 1, _now(), event.run_id))
        conn.commit()
        return event

    def pending(self, run_id: str):
        return self.repository.conn.execute("SELECT * FROM trace_outbox WHERE run_id=? AND delivered_at IS NULL ORDER BY seq_no", (run_id,)).fetchall()

    def mark_delivered(self, event_id: str) -> None:
        # Outbox rows are immutable; delivery metadata is the sole operational update.
        now = _now()
        self.repository.conn.execute("UPDATE trace_outbox SET delivered_at=?,updated_at=? WHERE event_id=? AND delivered_at IS NULL", (now, now, event_id))
        self.repository.conn.commit()
