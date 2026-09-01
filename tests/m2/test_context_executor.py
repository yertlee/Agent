import asyncio
import time
from datetime import datetime, timedelta, timezone

from agent.m2_context import InvocationContext
from agent.m2_executor import M2Executor
from agent.m2_registry import Registry, ToolSpec


def ctx(**kwargs):
    values = dict(session_id="s", user_id="u", run_id="r", plan_revision_id="p", task_id="t", attempt_id="a", agent_ref="order@v1", auth_scope="order/read@v1", idempotency_key="i", deadline=datetime.now(timezone.utc) + timedelta(seconds=5), config_version="c", registry_version="r", dataset_version="d", trace_id="trace")
    values.update(kwargs)
    return InvocationContext(**values)


def registry(callable_fn):
    return Registry([ToolSpec(tool_ref="order/get@v1", capability_ref="order/read@v1", owner="order", args_schema="order.read.v1", result_schema="order.result.v1", risk="READ", implementation_mode="SIMULATED", side_effect="READ_ONLY", timeout_ms=5000, allowed_error_codes=("ORDER_NOT_FOUND",), callable=callable_fn)])


def test_executor_rejects_context_spoof_before_callable():
    calls = []
    result = M2Executor(registry(lambda **kwargs: calls.append(kwargs) or {"success": True})).invoke("order/get@v1", ctx(), {"order_id": "o", "phone_last4": "1234", "user_id": "spoof"}, capability_ref="order/read@v1")
    assert not result.ok
    assert calls == []


def test_executor_maps_error_and_counts_logical_calls():
    executor = M2Executor(registry(lambda **kwargs: {"success": False, "code": "ORDER_NOT_FOUND"}))
    first = executor.invoke("order/get@v1", ctx(), {"order_id": "o", "phone_last4": "1234"}, capability_ref="order/read@v1")
    second = executor.invoke("order/get@v1", ctx(), {"order_id": "o", "phone_last4": "1234"}, capability_ref="order/read@v1")
    assert first.error.code == "ORDER_NOT_FOUND"
    assert (first.logical_call_no, second.logical_call_no) == (1, 2)
    assert first.physical_attempt_no == 1


def test_executor_cancel_and_deadline_barriers_do_not_call():
    calls = []
    executor = M2Executor(registry(lambda **kwargs: calls.append(kwargs) or {"success": True}))
    cancelled = executor.invoke("order/get@v1", ctx(), {"order_id": "o", "phone_last4": "1234"}, capability_ref="order/read@v1", cancelled=lambda: True)
    expired = executor.invoke("order/get@v1", ctx(deadline=datetime.now(timezone.utc) - timedelta(seconds=1)), {"order_id": "o", "phone_last4": "1234"}, capability_ref="order/read@v1")
    assert cancelled.error.code == "CANCELLED"
    assert expired.error.code == "DEADLINE_EXCEEDED"
    assert calls == []


def test_executor_async_callable_and_barrier():
    async def call(**kwargs):
        await asyncio.sleep(0)
        return {"success": True, "data": kwargs}
    spec = ToolSpec(tool_ref="order/get@v1", capability_ref="order/read@v1", owner="order", args_schema="order.read.v1", result_schema="order.result.v1", risk="READ", implementation_mode="SIMULATED", side_effect="READ_ONLY", timeout_ms=5000, callable=call)
    result = asyncio.run(M2Executor(Registry([spec])).invoke_async("order/get@v1", ctx(), {"order_id": "o", "phone_last4": "1234"}, capability_ref="order/read@v1"))
    assert result.ok and result.data["data"]["order_id"] == "o"


def test_executor_requires_explicit_owner_scope_and_capability():
    calls = []
    executor = M2Executor(registry(lambda **kwargs: calls.append(kwargs) or {"success": True}))
    missing = executor.invoke("order/get@v1", ctx(), {"order_id": "o", "phone_last4": "1234"})
    wrong_owner = executor.invoke("order/get@v1", ctx(agent_ref="other@v1"), {"order_id": "o", "phone_last4": "1234"}, capability_ref="order/read@v1")
    wrong_scope = executor.invoke("order/get@v1", ctx(auth_scope="order/write@v1"), {"order_id": "o", "phone_last4": "1234"}, capability_ref="order/read@v1")
    assert missing.error.code == "AUTH_CAPABILITY_DENIED"
    assert wrong_owner.error.code == "AUTH_CAPABILITY_DENIED"
    assert wrong_scope.error.code == "AUTH_CAPABILITY_DENIED"
    assert calls == []


def test_executor_enforces_timeout_for_sync_read():
    spec = ToolSpec(tool_ref="order/get@v1", capability_ref="order/read@v1", owner="order", args_schema="order.read.v1", result_schema="order.result.v1", risk="READ", implementation_mode="SIMULATED", side_effect="READ_ONLY", timeout_ms=10, callable=lambda **kwargs: (time.sleep(0.05), {"success": True})[1])
    result = M2Executor(Registry([spec])).invoke("order/get@v1", ctx(), {"order_id": "o", "phone_last4": "1234"}, capability_ref="order/read@v1")
    assert not result.ok and result.error.code == "DEADLINE_EXCEEDED"
