import sqlite3
from pathlib import Path

from agent.tools import get_order_info


def test_order_read_vertical_slice_preserves_success_and_ownership_errors(monkeypatch, tmp_path: Path) -> None:
    db = tmp_path / "orders.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE orders (order_id TEXT PRIMARY KEY, phone_last4 TEXT, product_name TEXT, amount REAL, order_status TEXT, pay_status TEXT, created_at TEXT, can_apply_aftersales INTEGER)")
    conn.execute("INSERT INTO orders VALUES ('990000000001', '1234', 'fixture product', 1.0, 'PAID', 'PAID', '2026-01-01T00:00:00Z', 1)")
    conn.commit()
    conn.close()
    monkeypatch.setattr("agent.tools.DB_PATH", str(db))
    success = get_order_info("990000000001", "1234")
    mismatch = get_order_info("990000000001", "0000")
    missing = get_order_info("990000000099", "8820")
    assert success["success"] is True
    assert success["code"] == "OK"
    assert mismatch["success"] is False
    assert mismatch["code"] == "PHONE_MISMATCH"
    assert missing["success"] is False
    assert missing["code"] == "ORDER_NOT_FOUND"
