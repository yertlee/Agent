import sqlite3
from pathlib import Path

from agent.storage.repository import SQLiteOrderRepository


def _fixture(tmp_path: Path) -> Path:
    db = tmp_path / "orders.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE orders (order_id TEXT PRIMARY KEY, phone_last4 TEXT, product_name TEXT, amount REAL, order_status TEXT, pay_status TEXT, created_at TEXT, can_apply_aftersales INTEGER)")
    conn.execute("INSERT INTO orders VALUES ('990000000001', '1234', 'fixture product', 1.0, 'PAID', 'PAID', '2026-01-01T00:00:00Z', 1)")
    conn.commit()
    conn.close()
    return db


def test_order_repository_filters_ownership_in_sql(tmp_path: Path) -> None:
    db = _fixture(tmp_path)
    with SQLiteOrderRepository(str(db)) as repo:
        order = repo.get_order_by_id("990000000001")
        assert order is not None
        assert repo.get_order_for_owner("990000000001", str(order["phone_last4"])) == order
        assert repo.get_order_for_owner("990000000001", "0000") is None


def test_repository_is_read_only(tmp_path: Path) -> None:
    db = _fixture(tmp_path)
    with SQLiteOrderRepository(str(db)) as repo:
        assert repo.get_order_by_id("does-not-exist") is None
