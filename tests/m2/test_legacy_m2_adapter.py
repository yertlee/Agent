import sqlite3
from unittest.mock import patch

from agent import tools
from agent.m2_legacy_adapter import M2OrderAdapter


def _db(tmp_path):
    path = tmp_path / "orders.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE orders (order_id TEXT PRIMARY KEY, phone_last4 TEXT, product_name TEXT, amount REAL, order_status TEXT, pay_status TEXT, created_at TEXT, can_apply_aftersales INTEGER)")
    conn.execute("INSERT INTO orders VALUES ('202600000001','1234','fixture',3.0,'PAID','PAID','2026-01-01T00:00:00Z',1)")
    conn.commit()
    conn.close()
    return path


def test_enabled_get_order_info_uses_m2_registry_context_executor(monkeypatch, tmp_path):
    db = _db(tmp_path)
    monkeypatch.setattr(tools, "DB_PATH", str(db))
    adapter = M2OrderAdapter(source_db=str(db))
    with patch.object(tools, "M2OrderAdapter", wraps=lambda **kwargs: adapter) as constructor:
        monkeypatch.setenv("M2_EXECUTION_ENABLED", "1")
        result = tools.get_order_info("202600000001", "1234")
    assert result["success"] is True
    assert result["code"] == "OK"
    assert constructor.called


def test_disabled_m2_flag_keeps_legacy_response_surface(monkeypatch, tmp_path):
    db = _db(tmp_path)
    monkeypatch.setattr(tools, "DB_PATH", str(db))
    monkeypatch.setenv("M2_EXECUTION_ENABLED", "0")
    result = tools.get_order_info("202600000001", "0000")
    assert result["success"] is False
    assert result["code"] == "PHONE_MISMATCH"
