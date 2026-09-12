"""Versioned read-only R2 logistics snapshot repository.

The production order database remains untouched.  ``seed_from_orders`` creates
an explicit R2 snapshot database with a schema/version marker and event rows;
the runtime opens that database read-only and never falls back to the legacy
logistics simulator.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Mapping

from .domain.objects import canonical_json, sha256_json


R2_LOGISTICS_SCHEMA_VERSION = "r2.logistics.sqlite.schema.v1"
R2_LOGISTICS_SOURCE_VERSION = "versioned_order_derived_logistics_snapshot.v1"


SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS r2_metadata (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS logistics_snapshots (
  snapshot_id TEXT PRIMARY KEY,
  order_ref_hash TEXT NOT NULL,
  carrier_code TEXT NOT NULL,
  tracking_no TEXT NOT NULL,
  delivery_state TEXT NOT NULL,
  shipment_status TEXT NOT NULL,
  observed_at TEXT NOT NULL,
  data_quality TEXT NOT NULL CHECK(data_quality IN ('FRESH','STALE','MISSING','CONFLICT')),
  source_name TEXT NOT NULL,
  source_version TEXT NOT NULL,
  snapshot_hash TEXT NOT NULL,
  UNIQUE(order_ref_hash, carrier_code, tracking_no, snapshot_hash)
);
CREATE TABLE IF NOT EXISTS logistics_events (
  event_id TEXT PRIMARY KEY,
  snapshot_id TEXT NOT NULL REFERENCES logistics_snapshots(snapshot_id),
  ordinal INTEGER NOT NULL,
  event_code TEXT NOT NULL,
  event_time TEXT NOT NULL,
  event_hash TEXT NOT NULL,
  UNIQUE(snapshot_id, ordinal)
);
CREATE INDEX IF NOT EXISTS idx_r2_logistics_order_ref ON logistics_snapshots(order_ref_hash);
CREATE INDEX IF NOT EXISTS idx_r2_logistics_tracking ON logistics_snapshots(carrier_code, tracking_no);
"""


def order_ref_hash(order_id: str) -> str:
    return hashlib.sha256(("r2-order-ref-v1:" + str(order_id)).encode("utf-8")).hexdigest()


def _connect_rw(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def initialize_logistics_db(path: str | Path, *, dataset_version: str = "dev-order-logistics-dag-v1", replace: bool = False) -> Path:
    target = Path(path)
    if target.exists() and not replace:
        raise FileExistsError(str(target))
    if replace and target.exists():
        target.unlink()
    conn = _connect_rw(target)
    try:
        conn.executescript(SCHEMA)
        conn.executemany("INSERT INTO r2_metadata(key,value) VALUES (?,?)", [("schema_version", R2_LOGISTICS_SCHEMA_VERSION), ("source_version", R2_LOGISTICS_SOURCE_VERSION), ("dataset_version", dataset_version)])
        conn.commit()
    finally:
        conn.close()
    return target


def seed_rows(path: str | Path, rows: Iterable[Mapping[str, Any]], *, dataset_version: str = "dev-order-logistics-dag-v1") -> Path:
    """Create a deterministic versioned snapshot from already-safe row mappings.

    This helper is used by tests/evaluator setup.  It never accepts phone or
    raw order IDs; callers provide the stable order reference hash instead.
    """
    target = initialize_logistics_db(path, dataset_version=dataset_version)
    conn = _connect_rw(target)
    try:
        for row in rows:
            payload = {
                "order_ref_hash": str(row["order_ref_hash"]),
                "carrier_code": str(row["carrier_code"]),
                "tracking_no": str(row["tracking_no"]),
                "delivery_state": str(row.get("delivery_state") or "UNKNOWN"),
                "shipment_status": str(row.get("shipment_status") or ""),
                "observed_at": str(row.get("observed_at") or "2026-01-01T00:00:00Z"),
                "data_quality": str(row.get("data_quality") or "FRESH"),
                "source_name": str(row.get("source_name") or "versioned_order_derived_logistics_snapshot"),
                "source_version": str(row.get("source_version") or R2_LOGISTICS_SOURCE_VERSION),
            }
            snapshot_hash = sha256_json(payload)
            snapshot_id = str(row.get("snapshot_id") or "snapshot_" + snapshot_hash[:24])
            conn.execute("INSERT INTO logistics_snapshots VALUES (?,?,?,?,?,?,?,?,?,?,?)", (snapshot_id, payload["order_ref_hash"], payload["carrier_code"], payload["tracking_no"], payload["delivery_state"], payload["shipment_status"], payload["observed_at"], payload["data_quality"], payload["source_name"], payload["source_version"], snapshot_hash))
            events = row.get("events") or []
            for ordinal, event in enumerate(events):
                event_code = str(event.get("event_code") if isinstance(event, Mapping) else event)
                event_time = str(event.get("event_time") if isinstance(event, Mapping) else payload["observed_at"])
                event_hash = sha256_json({"snapshot_id": snapshot_id, "ordinal": ordinal, "event_code": event_code, "event_time": event_time})
                conn.execute("INSERT INTO logistics_events VALUES (?,?,?,?,?,?)", (f"event_{event_hash[:24]}", snapshot_id, ordinal, event_code, event_time, event_hash))
        conn.commit()
    finally:
        conn.close()
    return target


class R2LogisticsRepository:
    """Read-only access to versioned snapshots and event rows."""

    def __init__(self, path: str | Path):
        self.path = str(path)

    def _connect(self) -> sqlite3.Connection:
        target = Path(self.path)
        if not target.is_file():
            raise FileNotFoundError(self.path)
        conn = sqlite3.connect(f"file:{target.resolve()}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only=ON")
        return conn

    def metadata(self) -> dict[str, str]:
        conn = self._connect()
        try:
            return {str(row[0]): str(row[1]) for row in conn.execute("SELECT key,value FROM r2_metadata")}
        finally:
            conn.close()

    def read(self, *, order_ref: str, carrier_code: str, tracking_no: str) -> dict[str, Any] | None:
        conn = self._connect()
        try:
            row = conn.execute("SELECT * FROM logistics_snapshots WHERE order_ref_hash=? AND carrier_code=? AND tracking_no=? ORDER BY observed_at DESC LIMIT 1", (order_ref, carrier_code, tracking_no)).fetchone()
            if row is None:
                return None
            events = [dict(item) for item in conn.execute("SELECT ordinal,event_code,event_time,event_hash FROM logistics_events WHERE snapshot_id=? ORDER BY ordinal", (str(row["snapshot_id"]),)).fetchall()]
            data = dict(row)
            data["events"] = events
            return data
        finally:
            conn.close()


def seed_from_orders(order_db_path: str | Path, logistics_db_path: str | Path, *, scene_clock: str = "2026-01-01T00:00:00Z") -> Path:
    """Build a versioned order-derived logistics snapshot from the read-only order source.

    The source DB is opened read-only.  The target is a new, separately
    versioned snapshot DB and is never the user's ``ecommerce.db``.
    """
    source = Path(order_db_path)
    conn = sqlite3.connect(f"file:{source.resolve()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT order_id,carrier_code,tracking_no,shipment_status,delivered_at,shipped_at FROM orders WHERE carrier_code IS NOT NULL AND tracking_no IS NOT NULL AND carrier_code <> '' AND tracking_no <> '' ORDER BY order_id").fetchall()
        mapped = []
        for row in rows:
            shipment = str(row["shipment_status"] or "")
            state = "DELIVERED" if shipment.upper() in {"DELIVERED", "SIGNED"} else (shipment.upper() or "UNKNOWN")
            observed = str(row["delivered_at"] or row["shipped_at"] or scene_clock)
            mapped.append({"order_ref_hash": order_ref_hash(str(row["order_id"])), "carrier_code": str(row["carrier_code"]), "tracking_no": str(row["tracking_no"]), "delivery_state": state, "shipment_status": shipment, "observed_at": observed, "data_quality": "FRESH", "source_name": "versioned_order_derived_logistics_snapshot", "source_version": R2_LOGISTICS_SOURCE_VERSION, "events": [{"event_code": state or "UNKNOWN", "event_time": observed}]})
    finally:
        conn.close()
    return seed_rows(logistics_db_path, mapped)


__all__ = ["R2_LOGISTICS_SCHEMA_VERSION", "R2_LOGISTICS_SOURCE_VERSION", "R2LogisticsRepository", "initialize_logistics_db", "order_ref_hash", "seed_from_orders", "seed_rows"]
