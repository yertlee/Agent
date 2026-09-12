from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from agent.domain.objects import sha256_json
from agent.r4_a2a_contracts import A2AMessageEnvelopeV1
from agent.r4_a2a_runtime import R4A2ARuntime
from agent.r2_logistics_repository import seed_from_orders


ROOT = Path(__file__).parents[2]
ORDER_DB = ROOT / "ecommerce.db"


def _order_row() -> tuple[str, str, str, str]:
    with sqlite3.connect(ORDER_DB) as conn:
        row = conn.execute(
            "SELECT order_id, phone_last4, carrier_code, tracking_no "
            "FROM orders WHERE carrier_code <> '' AND tracking_no <> '' "
            "ORDER BY order_id LIMIT 1"
        ).fetchone()
    if row is None:
        pytest.skip("the existing read-only ecommerce.db has no order with tracking")
    return tuple(str(value) for value in row)


def _runtime(tmp_path: Path, **kwargs) -> R4A2ARuntime:
    return R4A2ARuntime(db_path=ORDER_DB, ledger_path=tmp_path / "r4-ledger.sqlite", **kwargs)


def test_envelope_is_frozen_extra_forbid_and_hash_bound():
    now = datetime.now(timezone.utc)
    envelope = A2AMessageEnvelopeV1.request(
        message_id="m1",
        correlation_id="c1",
        run_id="r1",
        plan_revision_id="p1",
        task_id="t1",
        attempt_id="a1",
        trace_id="tr1",
        sender_ref="supervisor@v1",
        receiver_ref="order-agent@v1",
        capability_ref="order/read@v1",
        created_at=now,
        deadline=now + timedelta(seconds=5),
        idempotency_key="i1",
        payload={"order_id": "o1", "phone_last4": "1234"},
    )
    assert envelope.model_config["frozen"] is True
    assert envelope.payload_hash == sha256_json(envelope.payload)
    with pytest.raises(ValidationError):
        A2AMessageEnvelopeV1.model_validate({**envelope.model_dump(mode="python"), "extra": True})


def test_existing_order_sqlite_read_uses_isolated_ledger_and_agent_path(tmp_path: Path):
    order_id, phone_last4, _carrier, _tracking = _order_row()
    runtime = _runtime(tmp_path)
    try:
        request = runtime.build_request(
            run_id="r-order",
            plan_revision_id="p-order",
            task_id="order",
            capability_ref="order/read@v1",
            payload={"order_id": order_id, "phone_last4": phone_last4},
        )
        result = runtime.dispatch(request)
        assert result.status == "SUCCEEDED"
        assert result.specialist_result is not None and result.specialist_result.ok
        assert result.canonical_result is not None
        assert result.canonical_result.payload["source_version"] == "ecommerce.sqlite.read.v1"
        assert runtime.registry.refs() == (
            "logistics/query@v1",
            "order/get_info@v1",
            "policy/search@v1",
        )
        tables = {str(row[0]) for row in runtime.ledger.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert "a2a_messages" in tables
        assert "orders" not in tables
    finally:
        runtime.close()


def test_duplicate_same_hash_returns_cached_result_without_second_physical_call(tmp_path: Path):
    order_id, phone_last4, _carrier, _tracking = _order_row()
    runtime = _runtime(tmp_path)
    try:
        request = runtime.build_request(run_id="r-dup", plan_revision_id="p-dup", task_id="order", capability_ref="order/read@v1", payload={"order_id": order_id, "phone_last4": phone_last4}, idempotency_key="same-key")
        first = runtime.dispatch(request)
        second = runtime.dispatch(request)
        assert first.status == second.status == "SUCCEEDED"
        assert second.duplicate is True
        assert runtime.physical_call_counts["order/read@v1"] == 1
        assert len(runtime.ledger.attempts(request.message_id)) == 1
    finally:
        runtime.close()


def test_same_idempotency_key_different_hash_is_rejected(tmp_path: Path):
    order_id, phone_last4, _carrier, _tracking = _order_row()
    runtime = _runtime(tmp_path)
    try:
        first = runtime.build_request(run_id="r-conflict", plan_revision_id="p-conflict", task_id="order", capability_ref="order/read@v1", payload={"order_id": order_id, "phone_last4": phone_last4}, idempotency_key="collision")
        assert runtime.dispatch(first).status == "SUCCEEDED"
        second = first.model_copy(update={"message_id": "different", "payload": {"order_id": "different", "phone_last4": phone_last4}, "payload_hash": sha256_json({"order_id": "different", "phone_last4": phone_last4})})
        result = runtime.dispatch(second)
        assert result.status == "FAILED"
        assert result.error_code == "A2A_IDEMPOTENCY_CONFLICT"
        assert runtime.physical_call_counts["order/read@v1"] == 1
    finally:
        runtime.close()


def test_timeout_is_retried_once_and_lost_first_attempt_does_not_call_tool(tmp_path: Path):
    order_id, phone_last4, _carrier, _tracking = _order_row()
    runtime = _runtime(tmp_path, failure_script={"order": {"kind": "timeout", "count": 1}})
    try:
        request = runtime.build_request(run_id="r-retry", plan_revision_id="p-retry", task_id="order", capability_ref="order/read@v1", payload={"order_id": order_id, "phone_last4": phone_last4})
        result = runtime.dispatch(request)
        assert result.status == "SUCCEEDED"
        assert result.attempt_count == 2
        assert result.physical_call_count == 1
        assert runtime.physical_call_counts["order/read@v1"] == 1
        assert any(event.event_type == "A2A_BOUNDED_RETRY" for event in runtime.trace("r-retry"))
    finally:
        runtime.close()


def test_version_and_semantic_faults_fail_closed_without_write_or_retry(tmp_path: Path):
    order_id, phone_last4, _carrier, _tracking = _order_row()
    runtime = _runtime(tmp_path, failure_script={"order": "semantic_wrong"})
    try:
        valid = runtime.build_request(run_id="r-fault", plan_revision_id="p-fault", task_id="order", capability_ref="order/read@v1", payload={"order_id": order_id, "phone_last4": phone_last4})
        version_bad = valid.model_copy(update={"schema_version": "r4.a2a.message.v0"})
        assert runtime.dispatch(version_bad).error_code == "A2A_SCHEMA_VERSION_MISMATCH"
        wrong_receiver = valid.model_copy(update={"receiver_ref": "logistics-agent@v1"})
        assert runtime.dispatch(wrong_receiver).error_code == "A2A_RECEIVER_MISMATCH"
        semantic = runtime.dispatch(valid)
        assert semantic.status == "FAILED"
        assert semantic.error_code == "A2A_SEMANTIC_WRONG"
        assert semantic.attempt_count == 1
        assert runtime.physical_call_counts["order/read@v1"] == 1
        assert "aftersales/create@v1" not in runtime.registry.refs()
    finally:
        runtime.close()


def test_out_of_order_pending_then_blocked_without_guessing(tmp_path: Path):
    order_id, phone_last4, carrier, tracking = _order_row()
    runtime = _runtime(tmp_path, failure_script={"order": "semantic_wrong"})
    try:
        order = runtime.build_request(run_id="r-ordering", plan_revision_id="p-ordering", task_id="order", capability_ref="order/read@v1", payload={"order_id": order_id, "phone_last4": phone_last4})
        logistics = runtime.build_request(run_id="r-ordering", plan_revision_id="p-ordering", task_id="logistics", capability_ref="logistics/read@v1", payload={"carrier_code": carrier, "tracking_no": tracking, "phone_last4": phone_last4}, dependency_message_ids=(order.message_id,))
        assert runtime.dispatch(logistics).status == "PENDING"
        assert runtime.dispatch(order).status == "FAILED"
        blocked = runtime.dispatch(logistics)
        assert blocked.status == "BLOCKED"
        assert blocked.error_code == "A2A_DEPENDENCY_BLOCKED"
        assert runtime.physical_call_counts["logistics/read@v1"] == 0
    finally:
        runtime.close()


def test_late_after_freeze_only_audits_and_does_not_mutate_state(tmp_path: Path):
    order_id, phone_last4, _carrier, _tracking = _order_row()
    runtime = _runtime(tmp_path)
    try:
        request = runtime.build_request(run_id="r-late", plan_revision_id="p-late", task_id="order", capability_ref="order/read@v1", payload={"order_id": order_id, "phone_last4": phone_last4})
        assert runtime.dispatch(request).status == "SUCCEEDED"
        runtime.freeze("r-late")
        result = runtime.dispatch(request)
        assert result.status == "LATE"
        assert result.late is True
        assert runtime.physical_call_counts["order/read@v1"] == 1
        assert runtime.ledger.late_events("r-late")
        assert not any(event.event_type == "A2A_LATE_AUDIT" for event in runtime.trace("r-late"))
        assert runtime.ledger.late_events("r-late")[0]["reason"] == "RUN_TERMINAL"
    finally:
        runtime.close()


def test_policy_read_adapter_is_canonical_and_read_only(tmp_path: Path):
    version_tuple = ["r3.manifest.v1", "r3.test.corpus.v1"] + [f"v{i}" for i in range(2, 14)] + ["a" * 64, "r3.5.runtime.v1", "b" * 64, "c" * 64, "d" * 64, "e" * 64]
    source_version = sha256_json(version_tuple)
    authority = {
        "source": "r3.5.project-authored-kb",
        "source_version": source_version,
        "version_tuple": version_tuple,
        "strategy_checksum": "d" * 64,
        "evidence": [{"evidence_id": "ev-1", "source_id": "source-1", "version": "v1", "chunk_id": "chunk-1", "text_hash": "f" * 64, "locator": "chunk-1"}],
    }

    def reader(query: str):
        return {
            "query": query,
            "status": "ANSWERED",
            "source": "r3.5.project-authored-kb",
            "source_version": source_version,
            "source_version_tuple": version_tuple,
            "evidence": [{"evidence_id": "ev-1", "source_id": "source-1", "version": "v1", "chunk_id": "chunk-1", "text_hash": "f" * 64, "locator": "chunk-1"}],
            "claims": [{"claim_id": "claim-1", "text": "verified", "evidence_ids": ["ev-1"]}],
        }

    runtime = _runtime(tmp_path, policy_reader=reader, policy_authority=authority)
    try:
        result = runtime.run(topology="policy_only", policy_query="read-only policy")
        assert result.status == "SUCCEEDED"
        assert result.dispatches["policy"].specialist_result is not None
        assert result.dispatches["policy"].specialist_result.payload["source_version"] == source_version
        assert all(spec.side_effect == "READ_ONLY" for spec in (runtime.registry.get(ref) for ref in runtime.registry.refs()))
    finally:
        runtime.close()


def test_logistics_adapter_reads_separate_snapshot_only(tmp_path: Path):
    order_id, phone_last4, carrier, tracking = _order_row()
    logistics_db = tmp_path / "logistics.sqlite"
    seed_from_orders(ORDER_DB, logistics_db)
    runtime = _runtime(tmp_path, logistics_db_path=logistics_db)
    try:
        result = runtime.run(topology="order->logistics", order_id=order_id, phone_last4=phone_last4)
        assert result.status == "SUCCEEDED"
        payload = result.dispatches["logistics"].specialist_result.payload
        assert payload["source_kind"] == "local_derived_snapshot"
        assert payload["source_version"] == "versioned_order_derived_logistics_snapshot.v1"
        assert payload["carrier_code"] == carrier
        assert payload["tracking_no"] == tracking
    finally:
        runtime.close()
