"""Explicit seam between the legacy order tool and the M1 read slice.

The adapter is opt-in.  Disabled mode calls the existing M0 repository and
preserves legacy response shape; enabled mode runs the canonical M1 chain.
"""
from __future__ import annotations

from typing import Any

from .order_slice import DEFAULT_RUNTIME_DB, run_order_read_slice
from .storage.repository import SQLiteOrderRepository


class LegacyOrderAdapter:
    def __init__(self, *, source_db: str = "ecommerce.db", runtime_db: str = DEFAULT_RUNTIME_DB, enabled: bool = False):
        self.source_db = source_db
        self.runtime_db = runtime_db
        self.enabled = enabled

    def query(self, order_id: str, phone_last4: str) -> dict[str, Any]:
        if self.enabled:
            outcome = run_order_read_slice(source_db=self.source_db, runtime_db=self.runtime_db, order_id=order_id, phone_last4=phone_last4)
            return {"mode": "m1", **outcome}
        with SQLiteOrderRepository(self.source_db) as repository:
            order = repository.get_order_for_owner(order_id, phone_last4)
        return {"mode": "legacy", "order": order, "business_code": "ORDER_FOUND" if order else "ORDER_NOT_FOUND"}
