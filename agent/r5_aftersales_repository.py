"""R5 isolated after-sales demo repository (ADR-0002 §3-4): read + eligibility.

The database is a self-built, versioned demo copy.  It carries an orders copy
(ownership proof via ``phone_last4``), the M2-compatible ``aftersales_cases``
table and a policy-rules table for the deterministic eligibility engine.  The
user's original ``ecommerce.db`` is never written.  ``user_id`` is derived
deterministically from the owning order's ``phone_last4`` (``demo_user_<p4>``);
case rows never override that derivation.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Mapping

from .domain.eligibility import EligibilityEngine
from .domain.facts import FactSource, OrderFact
from .domain.policy import PolicyCatalog, PolicyRule


R5_AFTERSALES_SCHEMA_VERSION = "r5.aftersales.sqlite.schema.v1"
R5_AFTERSALES_SOURCE_VERSION = "self_built_aftersales_demo.v1"

SUPPORTED_SERVICES = frozenset({"refund", "return", "exchange"})

SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS r5_metadata (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS orders_copy (
  order_id TEXT PRIMARY KEY,
  phone_last4 TEXT NOT NULL,
  product_name TEXT NOT NULL DEFAULT '',
  amount TEXT NOT NULL DEFAULT '0.00',
  order_status TEXT NOT NULL DEFAULT '',
  pay_status TEXT NOT NULL DEFAULT '',
  shipment_status TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL DEFAULT '',
  delivered_at TEXT
);
CREATE TABLE IF NOT EXISTS aftersales_cases (
  case_id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  order_id TEXT NOT NULL,
  service TEXT NOT NULL CHECK(service IN ('refund','return','exchange')),
  status TEXT NOT NULL,
  state_version INTEGER NOT NULL DEFAULT 0,
  run_id TEXT NOT NULL DEFAULT '',
  session_id TEXT NOT NULL DEFAULT '',
  task_id TEXT NOT NULL DEFAULT '',
  idempotency_key TEXT NOT NULL DEFAULT '',
  fingerprint TEXT NOT NULL DEFAULT '',
  reason TEXT NOT NULL DEFAULT '',
  amount TEXT NOT NULL DEFAULT '0.00',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_r5_cases_user_order ON aftersales_cases(user_id, order_id);
CREATE TABLE IF NOT EXISTS policy_rules (
  rule_id TEXT PRIMARY KEY,
  decision_logic TEXT NOT NULL,
  source TEXT NOT NULL,
  effective_from TEXT NOT NULL,
  effective_to TEXT,
  scope TEXT NOT NULL,
  priority INTEGER NOT NULL DEFAULT 0,
  supersedes TEXT,
  version TEXT NOT NULL
);
"""


def demo_user_id(phone_last4: str) -> str:
    return f"demo_user_{phone_last4}"


def _connect_rw(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def initialize_aftersales_db(path: str | Path, *, dataset_version: str = "r5-demo-v1", replace: bool = False) -> Path:
    target = Path(path)
    if target.exists() and not replace:
        raise FileExistsError(str(target))
    if replace and target.exists():
        target.unlink()
    conn = _connect_rw(target)
    try:
        conn.executescript(SCHEMA)
        conn.executemany(
            "INSERT INTO r5_metadata(key,value) VALUES (?,?)",
            [
                ("schema_version", R5_AFTERSALES_SCHEMA_VERSION),
                ("source_version", R5_AFTERSALES_SOURCE_VERSION),
                ("dataset_version", dataset_version),
            ],
        )
        conn.commit()
    finally:
        conn.close()
    return target


def seed_aftersales_demo(
    path: str | Path,
    *,
    orders_rows: Iterable[Mapping[str, Any]],
    case_rows: Iterable[Mapping[str, Any]],
    policy_rules: Iterable[Mapping[str, Any]] = (),
    dataset_version: str = "r5-demo-v1",
) -> Path:
    """Create the demo copy from explicit, already-canonical rows.

    ``case_rows`` require case_id/order_id/service/status/reason/amount and
    must not carry user_id — it is derived from the owning order.
    """
    target = initialize_aftersales_db(path, dataset_version=dataset_version, replace=True)
    conn = _connect_rw(target)
    try:
        for row in orders_rows:
            conn.execute(
                "INSERT INTO orders_copy VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    str(row["order_id"]), str(row["phone_last4"]), str(row.get("product_name") or ""),
                    str(row.get("amount") or "0.00"), str(row.get("order_status") or ""),
                    str(row.get("pay_status") or ""), str(row.get("shipment_status") or ""),
                    str(row.get("created_at") or ""), row.get("delivered_at"),
                ),
            )
        phone_by_order = {str(row["order_id"]): str(row["phone_last4"]) for row in orders_rows}
        for case in case_rows:
            order_id = str(case["order_id"])
            if order_id not in phone_by_order:
                raise ValueError(f"case {case['case_id']} references unknown order {order_id}")
            user_id = demo_user_id(phone_by_order[order_id])
            conn.execute(
                "INSERT INTO aftersales_cases VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    str(case["case_id"]), user_id, order_id, str(case["service"]), str(case["status"]),
                    int(case.get("state_version") or 0), str(case.get("run_id") or ""), str(case.get("session_id") or ""),
                    str(case.get("task_id") or ""), str(case.get("idempotency_key") or ""), str(case.get("fingerprint") or ""),
                    str(case.get("reason") or ""), str(case.get("amount") or "0.00"),
                    str(case.get("created_at") or "2026-03-22T00:00:00Z"), str(case.get("updated_at") or "2026-03-22T00:00:00Z"),
                ),
            )
        for rule in policy_rules:
            conn.execute(
                "INSERT INTO policy_rules VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    str(rule["rule_id"]), str(rule["decision_logic"]), str(rule.get("source") or "self_built_policy_catalog"),
                    str(rule["effective_from"]), rule.get("effective_to"), str(rule["scope"]),
                    int(rule.get("priority") or 0), rule.get("supersedes"), str(rule.get("version") or "policy.r5.v1"),
                ),
            )
        conn.commit()
    finally:
        conn.close()
    return target


ACTIVE_CASE_STATUSES = frozenset(
    {"REQUESTED", "UNDER_REVIEW", "APPROVED", "RETURN_PENDING", "RETURNED", "REFUND_PENDING", "EXCHANGE_PENDING", "HUMAN_REVIEW"}
)
TERMINAL_CASE_STATUSES = frozenset({"REFUNDED", "EXCHANGED", "REJECTED", "CANCELLED"})


def build_demo_policy_catalog(version: str = "policy.r5.v1") -> PolicyCatalog:
    """Deterministic demo catalog: allow supported services for non-terminal orders."""
    rules = [
        PolicyRule(
            rule_id=f"{version}.refund.allow", decision_logic="allow refund for cancellable or delivered orders",
            source="self_built_policy_catalog", effective_from=datetime(2026, 1, 1, tzinfo=timezone.utc),
            scope="refund", priority=10, version=version,
        ),
        PolicyRule(
            rule_id=f"{version}.return.allow", decision_logic="allow return for delivered orders",
            source="self_built_policy_catalog", effective_from=datetime(2026, 1, 1, tzinfo=timezone.utc),
            scope="return", priority=10, version=version,
        ),
        PolicyRule(
            rule_id=f"{version}.exchange.allow", decision_logic="allow exchange for delivered orders",
            source="self_built_policy_catalog", effective_from=datetime(2026, 1, 1, tzinfo=timezone.utc),
            scope="exchange", priority=10, version=version,
        ),
    ]
    return PolicyCatalog(rules, version=version)


class R5AfterSalesRepository:
    """Read-only access to the isolated after-sales demo copy."""

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
            return {str(row[0]): str(row[1]) for row in conn.execute("SELECT key,value FROM r5_metadata")}
        finally:
            conn.close()

    def get_order(self, order_id: str) -> dict[str, Any] | None:
        conn = self._connect()
        try:
            row = conn.execute("SELECT * FROM orders_copy WHERE order_id=?", (str(order_id),)).fetchone()
            return dict(row) if row is not None else None
        finally:
            conn.close()

    def cases_for(self, order_id: str) -> list[dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute("SELECT * FROM aftersales_cases WHERE order_id=? ORDER BY created_at, case_id", (str(order_id),)).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def order_fact(self, order_id: str) -> OrderFact | None:
        row = self.get_order(order_id)
        if row is None:
            return None
        created = str(row.get("created_at") or "2026-01-01T00:00:00Z")
        return OrderFact(
            fact_id=f"order_{order_id}",
            entity_id=str(order_id),
            source=FactSource.SIMULATOR,
            version="r5.orders.v1",
            user_id=demo_user_id(str(row["phone_last4"])),
            status=str(row.get("order_status") or "UNKNOWN"),
            amount=float(Decimal(str(row.get("amount") or "0"))),
            created_at=created,
        )

    def eligibility_engine(self) -> EligibilityEngine:
        conn = self._connect()
        try:
            rows = conn.execute("SELECT * FROM policy_rules").fetchall()
        finally:
            conn.close()
        if not rows:
            return EligibilityEngine(build_demo_policy_catalog())
        from datetime import datetime as _dt

        rules = [
            PolicyRule(
                rule_id=str(r["rule_id"]), decision_logic=str(r["decision_logic"]), source=str(r["source"]),
                effective_from=_dt.fromisoformat(str(r["effective_from"])),
                effective_to=_dt.fromisoformat(str(r["effective_to"])) if r["effective_to"] else None,
                scope=str(r["scope"]), priority=int(r["priority"]), supersedes=r["supersedes"], version=str(r["version"]),
            )
            for r in rows
        ]
        version = str(rows[0]["version"])
        return EligibilityEngine(PolicyCatalog(rules, version=version))


def build_aftersales_read_callable(db_path: str | Path):
    """Tool callable for ``aftersales/query@v1``.

    Distinguishes: unknown order (``ORDER_NOT_FOUND``), wrong owner
    (``AUTH_IDENTITY_MISMATCH``), no case at all, and ACTIVE vs terminal
    history.  Args schema: ``aftersales.query.v1``.
    """
    repository = R5AfterSalesRepository(db_path)

    def query_aftersales(*, order_id: str, phone_last4: str) -> dict[str, Any]:
        order = repository.get_order(order_id)
        if order is None:
            return {"success": False, "code": "ORDER_NOT_FOUND", "message": "order not found", "details": {"order_id": str(order_id)}, "data": None}
        if str(order["phone_last4"]) != str(phone_last4):
            return {"success": False, "code": "AUTH_IDENTITY_MISMATCH", "message": "order does not belong to this identity", "details": {"order_id": str(order_id)}, "data": None}
        cases = repository.cases_for(order_id)
        active = [c for c in cases if c["status"] in ACTIVE_CASE_STATUSES]
        terminal = [c for c in cases if c["status"] in TERMINAL_CASE_STATUSES]
        projection = [
            {"case_id": c["case_id"], "service": c["service"], "status": c["status"], "reason": c["reason"], "amount": c["amount"], "created_at": c["created_at"]}
            for c in cases
        ]
        return {
            "success": True,
            "code": "OK",
            "data": {
                "order_id": str(order_id),
                "case_summary": "ACTIVE_CASE" if active else ("HISTORY_ONLY" if terminal else "NO_CASE"),
                "active_cases": [{"case_id": c["case_id"], "service": c["service"], "status": c["status"]} for c in active],
                "cases": projection,
                "source": R5_AFTERSALES_SOURCE_VERSION,
            },
        }

    return query_aftersales


def build_eligibility_callable(db_path: str | Path):
    """Tool callable for ``aftersales/eligibility@v1`` (deterministic engine).

    Ownership is checked first; eligibility is decided only by the versioned
    PolicyCatalog-backed engine, never by the model.  Args schema:
    ``aftersales.eligibility.v1`` (R5ToolSpec extension).
    """
    repository = R5AfterSalesRepository(db_path)

    def check_eligibility(*, order_id: str, phone_last4: str, service: str) -> dict[str, Any]:
        order = repository.get_order(order_id)
        if order is None:
            return {"success": False, "code": "ORDER_NOT_FOUND", "message": "order not found", "details": {"order_id": str(order_id)}, "data": None}
        if str(order["phone_last4"]) != str(phone_last4):
            return {"success": False, "code": "AUTH_IDENTITY_MISMATCH", "message": "order does not belong to this identity", "details": {"order_id": str(order_id)}, "data": None}
        fact = repository.order_fact(order_id)
        engine = repository.eligibility_engine()
        eligibility = engine.check(fact, service=str(service), now=datetime.now(timezone.utc))
        return {"success": True, "code": "OK", "data": eligibility.model_dump(mode="json")}

    return check_eligibility


__all__ = [
    "ACTIVE_CASE_STATUSES",
    "R5_AFTERSALES_SCHEMA_VERSION",
    "R5AfterSalesRepository",
    "TERMINAL_CASE_STATUSES",
    "SUPPORTED_SERVICES",
    "build_aftersales_read_callable",
    "build_demo_policy_catalog",
    "build_eligibility_callable",
    "demo_user_id",
    "initialize_aftersales_db",
    "seed_aftersales_demo",
]
