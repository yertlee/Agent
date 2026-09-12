"""Build a deterministic, self-contained R5 synthetic fixture.

The repository's historical ``r5_seed_demo_data.py`` derives its demo
databases from a local ``ecommerce.db``.  This script is the no-key path for a
fresh clone: every row below is authored synthetic data and no source
database, environment file, network service, or model provider is consulted.

The command requires an explicit output directory and refuses to overwrite
any generated target already present there.  It creates these files:

* ``ecommerce.db`` - the read-only order projection used by the R4/R5 order
  route;
* ``r5_demo_v1.db`` - the product catalog and order-to-SKU links;
* ``r5_aftersales_demo_v1.db`` - orders, case history, and policy rows;
* ``r5_logistics_demo_v1.db`` - versioned logistics snapshots; and
* ``r5_synthetic_manifest.json`` - provenance, versions, row counts, and
  file hashes.

Run from the project root, for example::

    python scripts/r5_seed_synthetic_data.py --output-dir .tmp/r5-synthetic

The output directory may be removed after a run; it is deliberately separate
from the repository's original ``ecommerce.db`` and ``data`` directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.r2_logistics_repository import (  # noqa: E402
    R2_LOGISTICS_SOURCE_VERSION,
    order_ref_hash,
    seed_rows,
)
from agent.r5_aftersales_repository import seed_aftersales_demo  # noqa: E402
from agent.r5_product_repository import seed_products  # noqa: E402


DATASET_VERSION = "r5-synthetic-v1"
ORDERS_SOURCE_VERSION = "self_built_synthetic_orders.v1"
LOGISTICS_SOURCE_NAME = "self_built_synthetic_logistics_snapshot"
MANIFEST_VERSION = "r5.synthetic.manifest.v1"
SCENE_CLOCK = "2026-03-22T00:00:00Z"

ORDERS_DB_NAME = "ecommerce.db"
PRODUCT_DB_NAME = "r5_demo_v1.db"
AFTERSALES_DB_NAME = "r5_aftersales_demo_v1.db"
LOGISTICS_DB_NAME = "r5_logistics_demo_v1.db"
MANIFEST_NAME = "r5_synthetic_manifest.json"
TARGET_NAMES = (
    ORDERS_DB_NAME,
    PRODUCT_DB_NAME,
    AFTERSALES_DB_NAME,
    LOGISTICS_DB_NAME,
    MANIFEST_NAME,
)


# These values are fixture identifiers, not copied or transformed from a
# customer database.  Keeping the public dev corpus' identifier shape makes
# the generated paths usable with the existing R5 evaluator while the rows
# themselves remain entirely self-authored.
CATALOG: tuple[dict[str, Any], ...] = (
    {
        "sku": "SKU-1001",
        "name": "演示蓝牙耳机",
        "category": "数码",
        "attributes": {"category": "数码", "fixture": True},
        "price": "199.00",
        "currency": "CNY",
        "stock": 15,
        "listing_status": "ACTIVE",
    },
    {
        "sku": "SKU-1002",
        "name": "演示旅行箱",
        "category": "箱包",
        "attributes": {"category": "箱包", "fixture": True},
        "price": "399.00",
        "currency": "CNY",
        "stock": 9,
        "listing_status": "ACTIVE",
    },
    {
        "sku": "SKU-1003",
        "name": "演示咖啡机",
        "category": "家电",
        "attributes": {"category": "家电", "fixture": True},
        "price": "249.00",
        "currency": "CNY",
        "stock": 7,
        "listing_status": "ACTIVE",
    },
    {
        "sku": "SKU-1004",
        "name": "演示保温杯",
        "category": "家居",
        "attributes": {"category": "家居", "fixture": True},
        "price": "89.00",
        "currency": "CNY",
        "stock": 20,
        "listing_status": "ACTIVE",
    },
    {
        "sku": "SKU-1005",
        "name": "演示办公键盘",
        "category": "数码",
        "attributes": {"category": "数码", "fixture": True},
        "price": "129.00",
        "currency": "CNY",
        "stock": 0,
        "listing_status": "SOLD_OUT",
    },
    {
        "sku": "SKU-1006",
        "name": "演示无价样品",
        "category": "家居",
        "attributes": {"category": "家居", "fixture": True},
        "price": None,
        "currency": "CNY",
        "stock": 3,
        "listing_status": "DRAFT",
    },
)


# ``sku`` is used only to make the explicit order_sku_map below.  The orders
# table intentionally keeps the legacy projection shape consumed by the order
# repository and therefore does not need a SKU column.
ORDERS: tuple[dict[str, Any], ...] = (
    {
        "order_id": "20260320001",
        "phone_last4": "1234",
        "sku": "SKU-1001",
        "amount": 199.00,
        "order_status": "DELIVERED",
        "pay_status": "PAID",
        "shipment_status": "DELIVERED",
        "created_at": "2026-03-20T00:00:00Z",
        "shipped_at": "2026-03-21T00:00:00Z",
        "delivered_at": "2026-03-22T00:00:00Z",
        "carrier_code": "yuantong",
        "tracking_no": "7609205232746",
        "can_apply_aftersales": 1,
    },
    {
        "order_id": "20260320008",
        "phone_last4": "5678",
        "sku": "SKU-1002",
        "amount": 399.00,
        "order_status": "SHIPPED",
        "pay_status": "PAID",
        "shipment_status": "IN_TRANSIT",
        "created_at": "2026-03-20T00:00:00Z",
        "shipped_at": "2026-03-21T00:00:00Z",
        "delivered_at": None,
        "carrier_code": "yuantong",
        "tracking_no": "7609205232753",
        "can_apply_aftersales": 1,
    },
    {
        "order_id": "20260320009",
        "phone_last4": "2468",
        "sku": "SKU-1004",
        "amount": 89.00,
        "order_status": "CANCELLED",
        "pay_status": "REFUNDED",
        "shipment_status": "",
        "created_at": "2026-03-19T00:00:00Z",
        "shipped_at": None,
        "delivered_at": None,
        "carrier_code": "",
        "tracking_no": "",
        "can_apply_aftersales": 0,
    },
    {
        "order_id": "20260320010",
        "phone_last4": "9156",
        "sku": "SKU-1003",
        "amount": 249.00,
        "order_status": "DELIVERED",
        "pay_status": "PAID",
        "shipment_status": "DELIVERED",
        "created_at": "2026-03-18T00:00:00Z",
        "shipped_at": "2026-03-19T00:00:00Z",
        "delivered_at": "2026-03-21T00:00:00Z",
        "carrier_code": "yunda",
        "tracking_no": "7609205232760",
        "can_apply_aftersales": 1,
    },
    {
        "order_id": "20260320011",
        "phone_last4": "9156",
        "sku": "SKU-1005",
        "amount": 129.00,
        "order_status": "SHIPPED",
        "pay_status": "PAID",
        "shipment_status": "IN_TRANSIT",
        "created_at": "2026-03-17T00:00:00Z",
        "shipped_at": "2026-03-18T00:00:00Z",
        "delivered_at": None,
        "carrier_code": "yunda",
        "tracking_no": "7609205232777",
        "can_apply_aftersales": 1,
    },
)


CASES: tuple[dict[str, Any], ...] = (
    {
        "case_id": "SYN-CASE-001",
        "order_id": "20260320001",
        "service": "refund",
        "status": "UNDER_REVIEW",
        "reason": "synthetic active case",
        "amount": "199.00",
        "created_at": "2026-03-22T00:00:00Z",
        "updated_at": "2026-03-22T00:00:00Z",
    },
    {
        "case_id": "SYN-CASE-002",
        "order_id": "20260320001",
        "service": "return",
        "status": "REFUNDED",
        "reason": "synthetic terminal history",
        "amount": "199.00",
        "created_at": "2026-03-22T00:01:00Z",
        "updated_at": "2026-03-22T00:02:00Z",
    },
    {
        "case_id": "SYN-CASE-003",
        "order_id": "20260320009",
        "service": "exchange",
        "status": "CANCELLED",
        "reason": "synthetic cancelled history",
        "amount": "89.00",
        "created_at": "2026-03-22T00:03:00Z",
        "updated_at": "2026-03-22T00:04:00Z",
    },
    {
        "case_id": "SYN-CASE-004",
        "order_id": "20260320010",
        "service": "return",
        "status": "RETURN_PENDING",
        "reason": "synthetic return pending",
        "amount": "249.00",
        "created_at": "2026-03-22T00:05:00Z",
        "updated_at": "2026-03-22T00:05:00Z",
    },
)


POLICY_RULES: tuple[dict[str, Any], ...] = tuple(
    {
        "rule_id": f"policy.r5.synthetic.v1.{service}.allow",
        "decision_logic": f"allow {service} for synthetic non-terminal orders",
        "source": "self_built_policy_catalog",
        "effective_from": "2026-01-01T00:00:00+00:00",
        "effective_to": None,
        "scope": service,
        "priority": 10,
        "supersedes": None,
        "version": "policy.r5.synthetic.v1",
    }
    for service in ("refund", "return", "exchange")
)


LOGISTICS_ROWS: tuple[dict[str, Any], ...] = (
    {
        "order_id": "20260320001",
        "carrier_code": "yuantong",
        "tracking_no": "7609205232746",
        "delivery_state": "DELIVERED",
        "shipment_status": "DELIVERED",
        "observed_at": "2026-03-22T00:00:00Z",
        "data_quality": "FRESH",
        "events": (
            {"event_code": "PICKED_UP", "event_time": "2026-03-21T00:00:00Z"},
            {"event_code": "DELIVERED", "event_time": "2026-03-22T00:00:00Z"},
        ),
    },
    {
        "order_id": "20260320008",
        "carrier_code": "yuantong",
        "tracking_no": "7609205232753",
        "delivery_state": "IN_TRANSIT",
        "shipment_status": "IN_TRANSIT",
        "observed_at": "2026-03-21T00:00:00Z",
        "data_quality": "FRESH",
        "events": (
            {"event_code": "PICKED_UP", "event_time": "2026-03-21T00:00:00Z"},
            {"event_code": "IN_TRANSIT", "event_time": "2026-03-21T12:00:00Z"},
        ),
    },
    {
        "order_id": "20260320010",
        "carrier_code": "yunda",
        "tracking_no": "7609205232760",
        "delivery_state": "DELIVERED",
        "shipment_status": "DELIVERED",
        "observed_at": "2026-03-21T00:00:00Z",
        "data_quality": "FRESH",
        "events": (
            {"event_code": "DELIVERED", "event_time": "2026-03-21T00:00:00Z"},
        ),
    },
    {
        "order_id": "20260320011",
        "carrier_code": "yunda",
        "tracking_no": "7609205232777",
        "delivery_state": "IN_TRANSIT",
        "shipment_status": "IN_TRANSIT",
        "observed_at": "2026-03-18T00:00:00Z",
        "data_quality": "STALE",
        "events": (
            {"event_code": "IN_TRANSIT", "event_time": "2026-03-18T00:00:00Z"},
        ),
    },
)


ORDERS_SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE synthetic_metadata (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
CREATE TABLE orders (
  order_id TEXT PRIMARY KEY,
  phone_last4 TEXT NOT NULL,
  product_name TEXT NOT NULL,
  amount REAL NOT NULL,
  order_status TEXT NOT NULL,
  pay_status TEXT NOT NULL,
  shipment_status TEXT NOT NULL,
  created_at TEXT NOT NULL,
  shipped_at TEXT,
  delivered_at TEXT,
  carrier_code TEXT NOT NULL DEFAULT '',
  tracking_no TEXT NOT NULL DEFAULT '',
  can_apply_aftersales INTEGER NOT NULL DEFAULT 0
);
"""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _catalog_by_sku() -> dict[str, Mapping[str, Any]]:
    return {str(row["sku"]): row for row in CATALOG}


def _order_rows_for_copy() -> list[dict[str, Any]]:
    catalog = _catalog_by_sku()
    rows: list[dict[str, Any]] = []
    for order in ORDERS:
        product = catalog[str(order["sku"])]
        rows.append(
            {
                "order_id": str(order["order_id"]),
                "phone_last4": str(order["phone_last4"]),
                "product_name": str(product["name"]),
                "amount": f"{float(order['amount']):.2f}",
                "order_status": str(order["order_status"]),
                "pay_status": str(order["pay_status"]),
                "shipment_status": str(order["shipment_status"]),
                "created_at": str(order["created_at"]),
                "delivered_at": order["delivered_at"],
            }
        )
    return rows


def _create_orders_db(path: Path) -> None:
    catalog = _catalog_by_sku()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(ORDERS_SCHEMA)
        conn.executemany(
            "INSERT INTO synthetic_metadata(key,value) VALUES (?,?)",
            (
                ("dataset_version", DATASET_VERSION),
                ("source_version", ORDERS_SOURCE_VERSION),
                ("source_kind", "self_built_synthetic_fixture"),
                ("privacy", "no_user_provided_data"),
                ("scene_clock", SCENE_CLOCK),
            ),
        )
        conn.executemany(
            """
            INSERT INTO orders(
              order_id, phone_last4, product_name, amount, order_status,
              pay_status, shipment_status, created_at, shipped_at,
              delivered_at, carrier_code, tracking_no, can_apply_aftersales
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                (
                    str(order["order_id"]),
                    str(order["phone_last4"]),
                    str(catalog[str(order["sku"])] ["name"]),
                    float(order["amount"]),
                    str(order["order_status"]),
                    str(order["pay_status"]),
                    str(order["shipment_status"]),
                    str(order["created_at"]),
                    order["shipped_at"],
                    order["delivered_at"],
                    str(order["carrier_code"]),
                    str(order["tracking_no"]),
                    int(order["can_apply_aftersales"]),
                )
                for order in ORDERS
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _product_rows() -> list[dict[str, Any]]:
    return [dict(row) for row in CATALOG]


def _product_links() -> list[dict[str, str]]:
    catalog = _catalog_by_sku()
    return [
        {
            "order_id": str(order["order_id"]),
            "sku": str(order["sku"]),
            "product_name": str(catalog[str(order["sku"])] ["name"]),
        }
        for order in ORDERS
    ]


def _logistics_seed_rows() -> list[dict[str, Any]]:
    return [
        {
            "order_ref_hash": order_ref_hash(str(row["order_id"])),
            "carrier_code": str(row["carrier_code"]),
            "tracking_no": str(row["tracking_no"]),
            "delivery_state": str(row["delivery_state"]),
            "shipment_status": str(row["shipment_status"]),
            "observed_at": str(row["observed_at"]),
            "data_quality": str(row["data_quality"]),
            "source_name": LOGISTICS_SOURCE_NAME,
            "source_version": R2_LOGISTICS_SOURCE_VERSION,
            "events": list(row["events"]),
        }
        for row in LOGISTICS_ROWS
    ]


def _database_file_records(output_dir: Path) -> list[dict[str, Any]]:
    records = []
    for name, counts in (
        (ORDERS_DB_NAME, {"orders": len(ORDERS), "synthetic_metadata": 5}),
        (PRODUCT_DB_NAME, {"products": len(CATALOG), "order_sku_map": len(ORDERS)}),
        (AFTERSALES_DB_NAME, {"orders_copy": len(ORDERS), "aftersales_cases": len(CASES), "policy_rules": len(POLICY_RULES)}),
        (LOGISTICS_DB_NAME, {"logistics_snapshots": len(LOGISTICS_ROWS), "logistics_events": sum(len(row["events"]) for row in LOGISTICS_ROWS)}),
    ):
        path = output_dir / name
        records.append({"path": name, "sha256": _sha256_file(path), "row_counts": counts})
    return records


def _manifest(output_dir: Path) -> dict[str, Any]:
    return {
        "manifest_version": MANIFEST_VERSION,
        "dataset_version": DATASET_VERSION,
        "scene_clock": SCENE_CLOCK,
        "source": {
            "kind": "self_built_synthetic_fixture",
            "version": ORDERS_SOURCE_VERSION,
            "declared_in": "scripts/r5_seed_synthetic_data.py",
            "statement": "All rows are authored fixture data; no source database, environment file, network service, or model provider is read.",
        },
        "privacy": {
            "contains_user_provided_data": False,
            "contains_external_platform_data": False,
            "identifiers_are_fixture_values": True,
        },
        "component_versions": {
            "product_source": "self_built_product_catalog.v1",
            "aftersales_source": "self_built_aftersales_demo.v1",
            "logistics_source": R2_LOGISTICS_SOURCE_VERSION,
        },
        "files": _database_file_records(output_dir),
    }


def seed_synthetic_data(output_dir: str | Path) -> dict[str, Any]:
    """Create all synthetic R5 databases below an explicit isolated path.

    Existing generated targets are never replaced.  The returned mapping is
    suitable for a CLI report; the manifest itself only contains relative
    paths and deterministic hashes so it is portable between clones.
    """

    target_dir = Path(output_dir).expanduser().resolve()
    if target_dir == PROJECT_ROOT:
        raise ValueError("refusing to write synthetic databases into the repository root; pass a new child directory")
    if target_dir.exists() and not target_dir.is_dir():
        raise ValueError(f"output path is not a directory: {target_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)
    existing = [name for name in TARGET_NAMES if (target_dir / name).exists()]
    if existing:
        raise FileExistsError("refusing to overwrite existing output files: " + ", ".join(existing))

    orders_path = target_dir / ORDERS_DB_NAME
    product_path = target_dir / PRODUCT_DB_NAME
    aftersales_path = target_dir / AFTERSALES_DB_NAME
    logistics_path = target_dir / LOGISTICS_DB_NAME
    manifest_path = target_dir / MANIFEST_NAME

    _create_orders_db(orders_path)
    seed_products(
        product_path,
        _product_rows(),
        dataset_version=DATASET_VERSION,
        order_sku_rows=_product_links(),
    )
    seed_aftersales_demo(
        aftersales_path,
        orders_rows=_order_rows_for_copy(),
        case_rows=[dict(row) for row in CASES],
        policy_rules=[dict(row) for row in POLICY_RULES],
        dataset_version=DATASET_VERSION,
    )
    seed_rows(logistics_path, _logistics_seed_rows(), dataset_version=DATASET_VERSION)

    manifest = _manifest(target_dir)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "output_dir": str(target_dir),
        "manifest": str(manifest_path),
        "files": {name: str(target_dir / name) for name in TARGET_NAMES},
        "dataset_version": DATASET_VERSION,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="new isolated directory for the synthetic fixture; existing target files are never overwritten",
    )
    args = parser.parse_args(argv)
    try:
        report = seed_synthetic_data(args.output_dir)
    except (FileExistsError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
