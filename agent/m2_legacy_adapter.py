"""Feature-flagged legacy tool seam through the M2 Registry and Executor."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from .m2_context import InvocationContext
from .m2_executor import M2Executor
from .m2_registry import Registry, ToolSpec
from .storage.repository import SQLiteOrderRepository


def _legacy_order_read(*, source_db: str, order_id: str, phone_last4: str) -> dict[str, Any]:
    with SQLiteOrderRepository(source_db) as repository:
        order = repository.get_order_for_owner(order_id, phone_last4)
        if order is None:
            exists = repository.get_order_by_id(order_id)
            code = "PHONE_MISMATCH" if exists else "ORDER_NOT_FOUND"
            return {"success": False, "code": code, "message": code, "data": None, "user_hint": ""}
        data = {k: v for k, v in order.items() if k not in {"phone_last4"}}
        return {"success": True, "code": "OK", "message": "订单查询成功", "data": data, "user_hint": ""}


class M2OrderAdapter:
    def __init__(self, *, source_db: str, config_version: str = "m2.v1"):
        if not source_db:
            raise FileNotFoundError("source database path is required")
        self.source_db = source_db
        spec = ToolSpec(tool_ref="order/get_info@v1", capability_ref="order/read@v1", owner="order-agent", args_schema="order.read.v1", result_schema="order.result.v1", risk="READ", implementation_mode="SIMULATED", side_effect="READ_ONLY", timeout_ms=5000, allowed_error_codes=("AUTH_IDENTITY_MISMATCH", "ORDER_NOT_FOUND", "CONFIG_MISSING", "INFRA_UNAVAILABLE"), callable=lambda **kwargs: _legacy_order_read(source_db=self.source_db, **kwargs))
        self.executor = M2Executor(Registry([spec]))
        self.config_version = config_version

    def query(self, order_id: str, phone_last4: str) -> dict[str, Any]:
        context = InvocationContext(session_id=f"session_{uuid4().hex}", user_id="legacy_user", run_id=f"run_{uuid4().hex}", plan_revision_id=f"plan_{uuid4().hex}", task_id=f"task_{uuid4().hex}", attempt_id=f"attempt_{uuid4().hex}", agent_ref="order-agent@v1", auth_scope="order/read@v1", idempotency_key=f"idemp_{uuid4().hex}", deadline=datetime.now(timezone.utc) + timedelta(seconds=5), config_version=self.config_version, registry_version="m2.registry.v1", dataset_version="runtime", trace_id=f"trace_{uuid4().hex}")
        result = self.executor.invoke("order/get_info@v1", context, {"order_id": order_id, "phone_last4": phone_last4}, capability_ref="order/read@v1")
        if result.ok:
            return result.data
        return {"success": False, "code": result.error.code if result.error else "TOOL_EXECUTION_FAILED", "message": result.error.message_key if result.error else "tool failure", "data": None, "user_hint": ""}


__all__ = ["M2OrderAdapter"]
