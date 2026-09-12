"""Seed the R5 isolated demo product catalog (ADR-0002 §1-2).

Reads the user's original ``ecommerce.db`` strictly read-only to derive the
order→SKU mapping from existing order product names, then writes a new
versioned catalog database at ``data/r5_demo_v1.db``.  The original database
is never modified.  Rerunning the script rebuilds the same content (fixed
seed rows, deterministic snapshot hashes).

Run from the project root:  python scripts/r5_seed_demo_data.py
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agent.r5_aftersales_repository import seed_aftersales_demo  # noqa: E402
from agent.r5_product_repository import seed_products  # noqa: E402
from agent.r2_logistics_repository import seed_from_orders  # noqa: E402

# Deterministic catalog: sku -> (name, category, stock).  Names must cover
# every product_name that appears in the legacy orders table so the derived
# order_sku_map is complete.
CATALOG = [
    ("SKU-1001", "男士连帽卫衣", "服饰", 42),
    ("SKU-1002", "蓝牙降噪耳机", "数码", 15),
    ("SKU-1003", "轻便旅行箱 24 寸", "箱包", 9),
    ("SKU-1004", "办公机械键盘", "数码", 23),
    ("SKU-1005", "夏季跑步鞋", "服饰", 31),
    ("SKU-1006", "定制刻字保温杯", "家居", 60),
    ("SKU-1007", "护肤礼盒", "美妆", 12),
    ("SKU-1008", "美式咖啡机", "家电", 7),
    ("SKU-1009", "筋膜枪", "健康", 18),
    ("SKU-1010", "婴儿湿巾 8 包装", "母婴", 88),
    ("SKU-1011", "儿童书包", "母婴", 25),
    ("SKU-1012", "家居四件套", "家居", 14),
    ("SKU-1013", "家用空气炸锅", "家电", 11),
    ("SKU-1014", "桌面收纳架", "家居", 36),
    ("SKU-1015", "生鲜礼盒", "食品", 5),
    ("SKU-1016", "电动牙刷套装", "健康", 27),
    ("SKU-1017", "平板电脑保护壳", "数码", 44),
    ("SKU-1018", "贴身内衣套装", "服饰", 52),
    ("SKU-1019", "便携显示器 15.6", "数码", 6),
    ("SKU-1020", "线下核销服务券", "服务", 100),
    # Catalog-only rows (never ordered): one sold-out, one unpriced draft.
    ("SKU-2001", "加绒保暖手套", "服饰", 0),
    ("SKU-2002", "样品展示架", "家居", 3),
]

# The unpriced draft row exercises the "missing price is never guessed" rule.
UNPRICED_SKUS = {"SKU-2002"}
DRAFT_LISTING = {"SKU-2002"}
SOLD_OUT_LISTING = {"SKU-2001"}
# Catalog-only rows were never sold, so they get an explicitly declared
# list price instead of falling back to a fake 0.00.
DECLARED_LIST_PRICES = {"SKU-2001": "29.90"}


# Canonical after-sales status mapping for legacy Chinese ticket statuses.
TICKET_STATUS_MAP = {
    "已退款": "REFUNDED",
    "已完成": "REFUNDED",
    "已关闭": "CANCELLED",
    "待审核": "UNDER_REVIEW",
    "审核通过": "APPROVED",
    "退款处理中": "REFUND_PENDING",
    "退货待寄回": "RETURN_PENDING",
}
SERVICE_MAP = {"退款": "refund", "退货": "return", "换货": "exchange"}


# Canonical order status mapping (module guide 02 §2): legacy Chinese statuses
# are normalized at seed time so downstream engines see canonical values.
ORDER_STATUS_MAP = {
    "待发货": "CREATED",
    "已发货": "SHIPPED",
    "运输中": "SHIPPED",
    "派送中": "SHIPPED",
    "已签收": "DELIVERED",
    "已完成": "DELIVERED",
    "已取消": "CANCELLED",
    "已退款": "CANCELLED",
}


def seed_aftersales(orders: list[sqlite3.Row]) -> None:
    tickets_target = PROJECT_ROOT / "data" / "r5_aftersales_demo_v1.db"
    src = sqlite3.connect(f"file:{(PROJECT_ROOT / 'ecommerce.db').resolve()}?mode=ro", uri=True)
    src.row_factory = sqlite3.Row
    try:
        tickets = src.execute("SELECT ticket_id, order_id, service_type, ticket_status, reason, created_at, updated_at FROM aftersales_tickets ORDER BY ticket_id").fetchall()
    finally:
        src.close()

    case_rows = []
    skipped = []
    for ticket in tickets:
        status = TICKET_STATUS_MAP.get(str(ticket["ticket_status"]))
        service = SERVICE_MAP.get(str(ticket["service_type"]))
        if status is None or service is None:
            skipped.append(str(ticket["ticket_id"]))
            continue
        case_rows.append(
            {
                "case_id": str(ticket["ticket_id"]),
                "order_id": str(ticket["order_id"]),
                "service": service,
                "status": status,
                "reason": str(ticket["reason"] or ""),
                "amount": next((str(o["amount"]) for o in orders if str(o["order_id"]) == str(ticket["order_id"])), "0.00"),
                "created_at": str(ticket["created_at"] or "2026-03-22T00:00:00Z"),
                "updated_at": str(ticket["updated_at"] or "2026-03-22T00:00:00Z"),
            }
        )
    if skipped:
        raise SystemExit(f"unmapped legacy ticket statuses: {skipped}")

    orders_rows = [
        {
            "order_id": str(o["order_id"]),
            "phone_last4": str(o["phone_last4"]),
            "product_name": str(o["product_name"]),
            "amount": str(o["amount"]),
            "order_status": ORDER_STATUS_MAP.get(str(o["order_status"]), str(o["order_status"])),
            "pay_status": str(o["pay_status"]),
            "shipment_status": ORDER_STATUS_MAP.get(str(o["shipment_status"]), str(o["shipment_status"])),
            "created_at": str(o["created_at"] or ""),
            "delivered_at": o["delivered_at"],
        }
        for o in orders
    ]
    seed_aftersales_demo(tickets_target, orders_rows=orders_rows, case_rows=case_rows, dataset_version="r5-demo-v1")
    print(f"seeded {len(case_rows)} after-sales cases, {len(orders_rows)} orders copy -> {tickets_target}")


def main() -> None:
    source = PROJECT_ROOT / "ecommerce.db"
    target = PROJECT_ROOT / "data" / "r5_demo_v1.db"

    src = sqlite3.connect(f"file:{source.resolve()}?mode=ro", uri=True)
    src.row_factory = sqlite3.Row
    try:
        orders = src.execute("SELECT order_id, phone_last4, product_name, amount, order_status, pay_status, shipment_status, created_at, delivered_at FROM orders ORDER BY order_id").fetchall()
    finally:
        src.close()

    name_to_sku = {name: sku for sku, name, *_rest in CATALOG}
    rows = []
    for sku, name, category, stock in CATALOG:
        rows.append(
            {
                "sku": sku,
                "name": name,
                "category": category,
                "attributes": {"category": category},
                "price": None if sku in UNPRICED_SKUS else "0.00",
                "currency": "CNY",
                "stock": stock,
                "listing_status": "DRAFT" if sku in DRAFT_LISTING else ("SOLD_OUT" if sku in SOLD_OUT_LISTING else "ACTIVE"),
            }
        )

    # Real catalog prices are seeded from the legacy order amounts where the
    # product was sold exactly once (deterministic, declared derivation).
    price_by_name = {str(row["product_name"]): str(row["amount"]) for row in orders}
    for row in rows:
        if row["sku"] in DECLARED_LIST_PRICES:
            row["price"] = DECLARED_LIST_PRICES[row["sku"]]
        elif row["price"] == "0.00" and row["name"] in price_by_name:
            row["price"] = price_by_name[row["name"]]

    links = []
    for row in orders:
        product_name = str(row["product_name"])
        sku = name_to_sku.get(product_name)
        if sku is None:
            raise SystemExit(f"order {row['order_id']} references unmapped product_name: {product_name}")
        links.append({"order_id": str(row["order_id"]), "sku": sku, "product_name": product_name})

    seed_products(target, rows, dataset_version="r5-demo-v1", order_sku_rows=links)
    print(f"seeded {len(rows)} products, {len(links)} order-sku links -> {target}")
    seed_aftersales(orders)
    logistics_target = PROJECT_ROOT / "data" / "r5_logistics_demo_v1.db"
    seed_from_orders(source, logistics_target)
    print(f"seeded versioned logistics snapshot -> {logistics_target}")


if __name__ == "__main__":
    main()
