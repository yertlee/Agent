"""Repository interface + SQLite 实现（M0，plan step 2 / step 7 读方法）。

module-guide-06 §6 前置（repository / dialect / adapter）与 runbook §4.3
任务 2/7。M0 边界：
- 只交付只读订单查询所需接口（get_order_by_id / get_order_for_owner）与
  connection lifecycle / 事务入口；result/对象表留待 M1；
- ownership 作为接口参数传递（phone_last4），SQL 层过滤归属
  （get_order_for_owner）；跨用户（phone_last4 不符）一律返回 None；
- 不迁移任何写路径（aftersales create/transition 仍走 legacy tools.py）。
"""

from __future__ import annotations

import sqlite3
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from .dialect import SQLiteDialect, transaction

# orders 表只读投影字段（与 legacy tools.py _fetch_order 的 select_fields 一致）
ORDER_READ_FIELDS = (
    "order_id",
    "phone_last4",
    "product_name",
    "amount",
    "order_status",
    "pay_status",
    "created_at",
    "can_apply_aftersales",
)
ORDER_READ_OPTIONAL_FIELDS = ("carrier_code", "tracking_no")


class Repository(ABC):
    """repository interface（06 §6）：connection lifecycle、事务入口、ownership 参数。"""

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def transaction(self):
        """事务入口：with repo.transaction() as conn: ...（失败自动回滚）。"""

    @abstractmethod
    def get_order_by_id(self, order_id: str) -> Optional[Dict[str, Any]]:
        """按主键读订单（不做归属过滤；归属校验由调用方/SQL 层另验）。"""

    @abstractmethod
    def get_order_for_owner(self, order_id: str, phone_last4: str) -> Optional[Dict[str, Any]]:
        """ownership 过滤读取：order_id + phone_last4 同时命中才返回。"""


class SQLiteOrderRepository(Repository):
    """SQLite dialect 上的只读订单 repository。"""

    def __init__(self, db_path: str, dialect: Optional[SQLiteDialect] = None) -> None:
        self._dialect = dialect or SQLiteDialect()
        # 只读连接：M0 repository 只承载读路径；写路径仍属 legacy tools.py
        self._conn: sqlite3.Connection = self._dialect.connect(db_path, readonly=True)

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None  # type: ignore[assignment]

    def __enter__(self) -> "SQLiteOrderRepository":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False

    def transaction(self):
        return transaction(self._conn)

    def _table_columns(self, table: str) -> set:
        cur = self._conn.execute(f"PRAGMA table_info({table})")
        return {str(row[1]) for row in cur.fetchall()}

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        for f in ORDER_READ_FIELDS:
            data[f] = row[f]
        for f in ORDER_READ_OPTIONAL_FIELDS:
            if f in row.keys():
                data[f] = str(row[f] or "")
            else:
                data[f] = ""
        data["amount"] = float(data["amount"])
        data["can_apply_aftersales"] = int(data["can_apply_aftersales"])
        return data

    def get_order_by_id(self, order_id: str) -> Optional[Dict[str, Any]]:
        return self._fetch(order_id, phone_last4=None)

    def get_order_for_owner(self, order_id: str, phone_last4: str) -> Optional[Dict[str, Any]]:
        if not phone_last4:
            return None
        return self._fetch(order_id, phone_last4=phone_last4)

    def _fetch(self, order_id: str, phone_last4: Optional[str]) -> Optional[Dict[str, Any]]:
        columns = self._table_columns("orders")
        select_fields = [f for f in ORDER_READ_FIELDS]
        for optional_field in ORDER_READ_OPTIONAL_FIELDS:
            if optional_field in columns:
                select_fields.append(optional_field)

        sql = f"SELECT {', '.join(select_fields)} FROM orders WHERE order_id = ?"
        params: list = [order_id]
        if phone_last4 is not None:
            # ownership 在 SQL 层过滤（06 §6 repository ownership 参数传递）
            sql += " AND phone_last4 = ?"
            params.append(phone_last4)
        cur = self._conn.execute(sql, tuple(params))
        row = cur.fetchone()
        if row is None:
            return None
        return self._row_to_dict(row)
