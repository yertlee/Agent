"""Versioned R5 product catalog repository (ADR-0002 §1-2).

The catalog is local self-built data declared as ``self_built_product_catalog.v1``;
it is never presented as an external platform feed.  The database is created
from a versioned seed, opened read-only at query time, and never touches the
user's original ``ecommerce.db``.  ``order_sku_map`` is a versioned derived
link from legacy order product names to catalog SKUs.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Iterable, Mapping

from .domain.objects import sha256_json
from .r5_contracts import R5_CATALOG_VERSION, R5_PRODUCT_SOURCE_VERSION, product_fact_from_row


R5_PRODUCT_SCHEMA_VERSION = "r5.product.sqlite.schema.v1"


SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS r5_metadata (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS products (
  sku TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  category TEXT NOT NULL DEFAULT '',
  attributes_json TEXT NOT NULL DEFAULT '{}',
  price TEXT,
  currency TEXT NOT NULL DEFAULT 'CNY',
  stock INTEGER,
  listing_status TEXT NOT NULL DEFAULT 'ACTIVE',
  source TEXT NOT NULL,
  version TEXT NOT NULL,
  observed_at TEXT NOT NULL,
  snapshot_hash TEXT NOT NULL,
  data_quality TEXT NOT NULL CHECK(data_quality IN ('FRESH','STALE','MISSING','CONFLICT'))
);
CREATE TABLE IF NOT EXISTS order_sku_map (
  order_id TEXT NOT NULL,
  sku TEXT NOT NULL REFERENCES products(sku),
  product_name TEXT NOT NULL,
  mapping_version TEXT NOT NULL,
  PRIMARY KEY (order_id, sku)
);
"""


def _connect_rw(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def initialize_product_db(path: str | Path, *, dataset_version: str = "r5-demo-v1", replace: bool = False) -> Path:
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
                ("schema_version", R5_PRODUCT_SCHEMA_VERSION),
                ("catalog_version", R5_CATALOG_VERSION),
                ("source_version", R5_PRODUCT_SOURCE_VERSION),
                ("dataset_version", dataset_version),
            ],
        )
        conn.commit()
    finally:
        conn.close()
    return target


def _row_snapshot_hash(row: Mapping[str, Any]) -> str:
    import json as _json

    return sha256_json(
        {
            "sku": row["sku"],
            "name": row["name"],
            "category": row.get("category", ""),
            "attributes": _json.loads(row.get("attributes_json") or "{}"),
            "price": row.get("price"),
            "currency": row.get("currency", "CNY"),
            "stock": row.get("stock"),
            "listing_status": row.get("listing_status", "ACTIVE"),
            "source": row.get("source", R5_PRODUCT_SOURCE_VERSION),
            "version": row.get("version", R5_CATALOG_VERSION),
            "observed_at": row["observed_at"],
        }
    )


def seed_products(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    dataset_version: str = "r5-demo-v1",
    order_sku_rows: Iterable[Mapping[str, Any]] = (),
) -> Path:
    """Create the catalog DB from explicit seed rows.

    ``rows`` items require sku/name; price/stock default to present values and
    may be explicitly ``None`` to model missing facts.  ``order_sku_rows``
    items require order_id/sku/product_name.
    """
    import json as _json

    target = initialize_product_db(path, dataset_version=dataset_version, replace=True)
    conn = _connect_rw(target)
    try:
        for row in rows:
            record = {
                "sku": str(row["sku"]),
                "name": str(row["name"]),
                "category": str(row.get("category") or ""),
                "attributes_json": _json.dumps(dict(row.get("attributes") or {}), ensure_ascii=False, sort_keys=True),
                "price": None if row.get("price") is None else str(row["price"]),
                "currency": str(row.get("currency") or "CNY"),
                "stock": None if row.get("stock") is None else int(row["stock"]),
                "listing_status": str(row.get("listing_status") or "ACTIVE"),
                "source": R5_PRODUCT_SOURCE_VERSION,
                "version": R5_CATALOG_VERSION,
                "observed_at": str(row.get("observed_at") or "2026-01-01T00:00:00Z"),
            }
            missing = record["price"] is None or record["stock"] is None
            record["data_quality"] = "MISSING" if missing else "FRESH"
            record["snapshot_hash"] = _row_snapshot_hash(record)
            conn.execute(
                "INSERT INTO products VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    record["sku"], record["name"], record["category"], record["attributes_json"],
                    record["price"], record["currency"], record["stock"], record["listing_status"],
                    record["source"], record["version"], record["observed_at"], record["snapshot_hash"],
                    record["data_quality"],
                ),
            )
        for link in order_sku_rows:
            conn.execute(
                "INSERT INTO order_sku_map VALUES (?,?,?,?)",
                (str(link["order_id"]), str(link["sku"]), str(link["product_name"]), str(link.get("mapping_version") or "order_sku_map.v1")),
            )
        conn.commit()
    finally:
        conn.close()
    return target


class R5ProductRepository:
    """Read-only access to the versioned product catalog."""

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

    def get(self, sku: str) -> dict[str, Any] | None:
        conn = self._connect()
        try:
            row = conn.execute("SELECT * FROM products WHERE sku=?", (str(sku),)).fetchone()
            if row is None:
                return None
            return dict(row)
        finally:
            conn.close()

    def sku_for_order(self, order_id: str) -> dict[str, Any] | None:
        conn = self._connect()
        try:
            row = conn.execute("SELECT order_id,sku,product_name,mapping_version FROM order_sku_map WHERE order_id=?", (str(order_id),)).fetchone()
            return dict(row) if row is not None else None
        finally:
            conn.close()

    def fact(self, sku: str) -> dict[str, Any] | None:
        row = self.get(sku)
        if row is None:
            return None
        fact = product_fact_from_row(row)
        return fact.model_dump(mode="json")


def build_product_read_callable(db_path: str | Path):
    """Tool callable for ``product/get@v1``: exact SKU read, no guessing.

    Unknown SKU maps to canonical ``DATA_MISSING``; a row missing price or
    stock is returned with ``data_quality=MISSING`` and null fields instead of
    defaults.  Args schema: ``product.read.v1`` (sku only).
    """
    repository = R5ProductRepository(db_path)

    def product_read(*, sku: str) -> dict[str, Any]:
        row = repository.get(sku)
        if row is None:
            return {"success": False, "code": "DATA_MISSING", "message": "product not found in catalog", "details": {"sku": str(sku)}, "data": None}
        fact = product_fact_from_row(row)
        return {"success": True, "code": "OK", "data": fact.model_dump(mode="json")}

    return product_read


__all__ = [
    "R5_PRODUCT_SCHEMA_VERSION",
    "R5ProductRepository",
    "build_product_read_callable",
    "initialize_product_db",
    "seed_products",
]
