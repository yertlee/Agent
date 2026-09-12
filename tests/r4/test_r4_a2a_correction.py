from __future__ import annotations

import hashlib
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from agent.domain.objects import ResultStatus, sha256_json
from agent.r4_a2a_contracts import A2AContractError, A2AResultReferenceV1
from agent.r4_a2a_runtime import R4A2ARuntime


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


def _runtime(tmp_path: Path, **kwargs: object) -> R4A2ARuntime:
    return R4A2ARuntime(db_path=ORDER_DB, ledger_path=tmp_path / "r4-correction.sqlite", **kwargs)


def _policy_version() -> list[str]:
    return [
        "r3.manifest.v1",
        "r3.test.corpus.v1",
        *[f"v{i}" for i in range(2, 14)],
        "a" * 64,
        "r3.5.runtime.v1",
        "b" * 64,
        "c" * 64,
        "d" * 64,
        "e" * 64,
    ]


def _policy_value(query: str) -> dict[str, object]:
    version_tuple = _policy_version()
    return {
        "query": query,
        "query_hash": sha256_json(query),
        "status": "ANSWERED",
        "source": "r3.5.project-authored-kb",
        "source_version": sha256_json(version_tuple),
        "source_version_tuple": version_tuple,
        "evidence": [
            {
                "evidence_id": "evidence-1",
                "source_id": "project-source-1",
                "version": "v1",
                "chunk_id": "chunk-1",
                "text_hash": "f" * 64,
                "locator": "chunk-1",
            }
        ],
        "claims": [{"claim_id": "claim-1", "text": "verified", "evidence_ids": ["evidence-1"]}],
    }


def _policy_authority() -> dict[str, object]:
    version_tuple = _policy_version()
    return {
        "source": "r3.5.project-authored-kb",
        "source_version": sha256_json(version_tuple),
        "version_tuple": version_tuple,
        "strategy_checksum": "d" * 64,
        "evidence": [
            {
                "evidence_id": "evidence-1",
                "source_id": "project-source-1",
                "version": "v1",
                "chunk_id": "chunk-1",
                "text_hash": "f" * 64,
                "locator": "chunk-1",
            }
        ],
    }


def test_pending_same_key_different_hash_is_rejected_without_physical_call(tmp_path: Path):
    order_id, phone_last4, carrier, tracking = _order_row()
    runtime = _runtime(tmp_path)
    try:
        upstream = runtime.build_request(
            run_id="r-pending-fingerprint",
            plan_revision_id="p-pending-fingerprint",
            task_id="order",
            capability_ref="order/read@v1",
            payload={"order_id": order_id, "phone_last4": phone_last4},
        )
        dependent = runtime.build_request(
            run_id="r-pending-fingerprint",
            plan_revision_id="p-pending-fingerprint",
            task_id="logistics",
            capability_ref="logistics/read@v1",
            payload={"carrier_code": carrier, "tracking_no": tracking, "phone_last4": phone_last4},
            dependency_message_ids=(upstream.message_id,),
            idempotency_key="pending-same-key",
        )
        assert runtime.dispatch(dependent).status == "PENDING"
        tampered = dependent.model_copy(
            update={
                "message_id": "pending-tampered-id",
                "payload": {"carrier_code": carrier, "tracking_no": "different", "phone_last4": phone_last4},
                "payload_hash": sha256_json({"carrier_code": carrier, "tracking_no": "different", "phone_last4": phone_last4}),
            }
        )
        rejected = runtime.dispatch(tampered)
        assert rejected.status == "FAILED"
        assert rejected.error_code == "A2A_IDEMPOTENCY_CONFLICT"
        assert runtime.physical_call_counts["logistics/read@v1"] == 0
        assert runtime.ledger.attempts(dependent.message_id) == []
    finally:
        runtime.close()


def test_dependency_binding_rejects_cross_run_plan_and_correlation(tmp_path: Path):
    order_id, phone_last4, carrier, tracking = _order_row()
    runtime = _runtime(tmp_path)
    try:
        upstream = runtime.build_request(
            run_id="r-dep-source",
            plan_revision_id="p-dep-source",
            task_id="order",
            capability_ref="order/read@v1",
            payload={"order_id": order_id, "phone_last4": phone_last4},
            correlation_id="corr-source",
        )
        cross_run = runtime.build_request(
            run_id="r-dep-other-run",
            plan_revision_id="p-dep-other-run",
            task_id="logistics",
            capability_ref="logistics/read@v1",
            payload={"carrier_code": carrier, "tracking_no": tracking, "phone_last4": phone_last4},
            correlation_id="corr-source",
            dependency_message_ids=(upstream.message_id,),
        )
        assert runtime.dispatch(cross_run).error_code == "A2A_DEPENDENCY_MISMATCH"

        same_run = runtime.build_request(
            run_id="r-dep-correlation",
            plan_revision_id="p-dep-correlation",
            task_id="order",
            capability_ref="order/read@v1",
            payload={"order_id": order_id, "phone_last4": phone_last4},
            correlation_id="corr-one",
        )
        cross_correlation = runtime.build_request(
            run_id="r-dep-correlation",
            plan_revision_id="p-dep-correlation",
            task_id="logistics",
            capability_ref="logistics/read@v1",
            payload={"carrier_code": carrier, "tracking_no": tracking},
            correlation_id="corr-two",
            dependency_message_ids=(same_run.message_id,),
        )
        assert runtime.dispatch(cross_correlation).error_code == "A2A_DEPENDENCY_MISMATCH"

        same_plan = runtime.build_request(
            run_id="r-dep-plan",
            plan_revision_id="p-dep-plan",
            task_id="order",
            capability_ref="order/read@v1",
            payload={"order_id": order_id, "phone_last4": phone_last4},
            correlation_id="corr-plan",
        )
        same_plan_downstream = runtime.build_request(
            run_id="r-dep-plan",
            plan_revision_id="p-dep-plan",
            task_id="logistics",
            capability_ref="logistics/read@v1",
            payload={"carrier_code": carrier, "tracking_no": tracking},
            correlation_id="corr-plan",
            dependency_message_ids=(same_plan.message_id,),
        )
        # A trusted run has one current plan.  This mutation models a stale
        # dependency row after an adversarial cross-plan handoff; the
        # dispatcher must reject it before any adapter call.
        runtime.ledger.conn.execute(
            "UPDATE a2a_messages SET plan_revision_id=? WHERE message_id=?",
            ("p-forged-dependency", same_plan.message_id),
        )
        assert runtime.dispatch(same_plan_downstream).error_code == "A2A_DEPENDENCY_MISMATCH"
        assert runtime.physical_call_counts["logistics/read@v1"] == 0
    finally:
        runtime.close()


def test_same_run_new_plan_is_rejected_until_trusted_supersede(tmp_path: Path):
    order_id, phone_last4, _carrier, _tracking = _order_row()
    runtime = _runtime(tmp_path)
    try:
        old = runtime.build_request(
            run_id="r-plan-owner",
            plan_revision_id="p-old",
            task_id="order",
            capability_ref="order/read@v1",
            payload={"order_id": order_id, "phone_last4": phone_last4},
        )
        with pytest.raises(A2AContractError, match="A2A_RUN_PLAN_BINDING_MISMATCH"):
            runtime.start_run(run_id="r-plan-owner", plan_revision_id="p-new")
        new_without_supersede = old.model_copy(update={"message_id": "new-plan-message", "plan_revision_id": "p-new"})
        rejected = runtime.dispatch(new_without_supersede)
        assert rejected.status == "FAILED"
        assert rejected.error_code == "A2A_PLAN_NOT_CURRENT"
        assert runtime.physical_call_counts["order/read@v1"] == 0
    finally:
        runtime.close()


def test_trusted_supersede_updates_plan_and_old_revision_is_late(tmp_path: Path):
    order_id, phone_last4, _carrier, _tracking = _order_row()
    runtime = _runtime(tmp_path)
    try:
        old = runtime.build_request(
            run_id="r-supersede",
            plan_revision_id="p-old",
            task_id="order",
            capability_ref="order/read@v1",
            payload={"order_id": order_id, "phone_last4": phone_last4},
        )
        runtime.supersede_revision("r-supersede", "p-old", "p-new")
        old_result = runtime.dispatch(old)
        assert old_result.status == "LATE"
        assert runtime.ledger.late_events("r-supersede")[0]["reason"] == "REVISION_SUPERSEDED"
        new = runtime.build_request(
            run_id="r-supersede",
            plan_revision_id="p-new",
            task_id="order-new",
            capability_ref="order/read@v1",
            payload={"order_id": order_id, "phone_last4": phone_last4},
        )
        assert runtime.dispatch(new).status == "SUCCEEDED"
    finally:
        runtime.close()


def test_restart_duplicate_rehydrates_verified_response_specialist_and_canonical(tmp_path: Path):
    order_id, phone_last4, _carrier, _tracking = _order_row()
    ledger_path = tmp_path / "restart.sqlite"
    first_runtime = R4A2ARuntime(db_path=ORDER_DB, ledger_path=ledger_path)
    request = first_runtime.build_request(
        run_id="r-restart",
        plan_revision_id="p-restart",
        task_id="order",
        capability_ref="order/read@v1",
        payload={"order_id": order_id, "phone_last4": phone_last4},
        idempotency_key="restart-key",
    )
    first = first_runtime.dispatch(request)
    assert first.status == "SUCCEEDED"
    first_runtime.close()

    second_runtime = R4A2ARuntime(db_path=ORDER_DB, ledger_path=ledger_path)
    try:
        duplicate = second_runtime.dispatch(request)
        assert duplicate.status == "SUCCEEDED"
        assert duplicate.duplicate is True
        assert duplicate.duplicate_of == request.message_id
        assert duplicate.response is not None
        assert duplicate.response.parent_message_id == request.message_id
        assert duplicate.specialist_result is not None and duplicate.specialist_result.ok
        assert duplicate.canonical_result is not None
        assert duplicate.canonical_result.status is ResultStatus.SUCCEEDED
        assert duplicate.canonical_result.usage.physical_attempts == 1
        assert duplicate.physical_call_count == 1
        assert second_runtime.physical_call_counts["order/read@v1"] == 0
    finally:
        second_runtime.close()


@pytest.mark.parametrize("tampered_column", ["response_json", "canonical_json"])
def test_restart_tampered_persisted_artifact_fails_closed(tmp_path: Path, tampered_column: str):
    order_id, phone_last4, _carrier, _tracking = _order_row()
    ledger_path = tmp_path / f"tampered-{tampered_column}.sqlite"
    runtime = R4A2ARuntime(db_path=ORDER_DB, ledger_path=ledger_path)
    request = runtime.build_request(
        run_id=f"r-tampered-{tampered_column}",
        plan_revision_id="p-tampered",
        task_id="order",
        capability_ref="order/read@v1",
        payload={"order_id": order_id, "phone_last4": phone_last4},
    )
    assert runtime.dispatch(request).status == "SUCCEEDED"
    runtime.close()

    with sqlite3.connect(ledger_path) as conn:
        row = conn.execute(f"SELECT {tampered_column} FROM a2a_messages WHERE message_id=?", (request.message_id,)).fetchone()
        value = json.loads(str(row[0]))
        value["tampered"] = True
        conn.execute(f"UPDATE a2a_messages SET {tampered_column}=? WHERE message_id=?", (json.dumps(value), request.message_id))

    restarted = R4A2ARuntime(db_path=ORDER_DB, ledger_path=ledger_path)
    try:
        rejected = restarted.dispatch(request)
        assert rejected.status == "FAILED"
        assert rejected.duplicate is True
        assert rejected.error_code in {"A2A_PERSISTED_RESPONSE_CHECKSUM_MISMATCH", "A2A_PERSISTED_RESULT_CHECKSUM_MISMATCH"}
        assert rejected.canonical_result is None
    finally:
        restarted.close()


def test_same_key_same_hash_new_message_id_replays_original_correlation(tmp_path: Path):
    order_id, phone_last4, _carrier, _tracking = _order_row()
    runtime = _runtime(tmp_path)
    try:
        payload = {"order_id": order_id, "phone_last4": phone_last4}
        original = runtime.build_request(
            run_id="r-correlation-replay",
            plan_revision_id="p-correlation-replay",
            task_id="order",
            capability_ref="order/read@v1",
            payload=payload,
            correlation_id="corr-original",
            idempotency_key="correlation-replay-key",
        )
        assert runtime.dispatch(original).status == "SUCCEEDED"
        replay = runtime.build_request(
            run_id="r-correlation-replay",
            plan_revision_id="p-correlation-replay",
            task_id="order",
            capability_ref="order/read@v1",
            payload=payload,
            message_id="new-message-id",
            correlation_id="corr-original",
            idempotency_key="correlation-replay-key",
        )
        duplicate = runtime.dispatch(replay)
        assert duplicate.status == "SUCCEEDED"
        assert duplicate.duplicate_of == original.message_id
        assert duplicate.request.message_id == "new-message-id"
        assert duplicate.response is not None
        assert duplicate.response.correlation_id == original.correlation_id
        assert duplicate.response.parent_message_id == original.message_id
        with pytest.raises(A2AContractError, match="A2A_IDEMPOTENCY_CONFLICT"):
            runtime.build_request(
                run_id="r-correlation-replay",
                plan_revision_id="p-correlation-replay",
                task_id="order",
                capability_ref="order/read@v1",
                payload=payload,
                message_id="different-correlation-id",
                correlation_id="corr-other",
                idempotency_key="correlation-replay-key",
            )
    finally:
        runtime.close()


def test_result_ref_is_fail_closed_when_r4_a_has_no_result_store_reference_path(tmp_path: Path):
    order_id, phone_last4, _carrier, _tracking = _order_row()
    runtime = _runtime(tmp_path)
    try:
        request = runtime.build_request(
            run_id="r-result-ref",
            plan_revision_id="p-result-ref",
            task_id="order",
            capability_ref="order/read@v1",
            payload={"order_id": order_id, "phone_last4": phone_last4},
        )
        reference = A2AResultReferenceV1(
            result_id="forged-result",
            task_id=request.task_id,
            contract="order.result.v1",
            payload_hash="0" * 64,
            source_version="forged-source",
        )
        forged = request.result(
            request=request,
            sender_ref="order-agent@v1",
            receiver_ref="supervisor@v1",
            attempt_id="forged-attempt",
            result_ref=reference,
        )
        with pytest.raises(A2AContractError, match="A2A_RESULT_REFERENCE_DISABLED"):
            runtime.verifier.verify_result_envelope(request, forged)
    finally:
        runtime.close()


def test_physical_retry_counts_are_taken_from_attempt_ledger(tmp_path: Path):
    order_id, phone_last4, _carrier, _tracking = _order_row()
    runtime = _runtime(tmp_path, failure_script={"order": {"kind": "physical_timeout", "count": 1}})
    try:
        request = runtime.build_request(
            run_id="r-physical-retry",
            plan_revision_id="p-physical-retry",
            task_id="order",
            capability_ref="order/read@v1",
            payload={"order_id": order_id, "phone_last4": phone_last4},
        )
        result = runtime.dispatch(request)
        assert result.status == "SUCCEEDED"
        assert result.attempt_count == 2
        assert result.physical_call_count == 2
        assert runtime.physical_call_counts["order/read@v1"] == 2
        assert result.canonical_result is not None
        assert result.canonical_result.usage.physical_attempts == 2
        assert [int(row["physical_call"]) for row in runtime.ledger.attempts(request.message_id)] == [1, 1]
    finally:
        runtime.close()


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("failed", "A2A_POLICY_FAILED_STATUS"),
        ("empty_evidence", "A2A_POLICY_ANSWERED_WITHOUT_EVIDENCE"),
        ("wrong_claim", "A2A_POLICY_CLAIM_BINDING_INVALID"),
        ("source", "A2A_POLICY_SOURCE_UNAUTHORIZED"),
        ("version", "A2A_POLICY_VERSION_UNAUTHORIZED"),
    ],
)
def test_policy_status_evidence_claim_source_and_version_fail_closed(tmp_path: Path, mutation: str, expected: str):
    def reader(query: str) -> dict[str, object]:
        value = _policy_value(query)
        if mutation == "failed":
            value["status"] = "FAILED"
        elif mutation == "empty_evidence":
            value["evidence"] = []
        elif mutation == "wrong_claim":
            value["claims"] = [{"claim_id": "claim-1", "text": "forged", "evidence_ids": ["missing"]}]
        elif mutation == "source":
            value["source"] = "unauthorized-source"
        elif mutation == "version":
            value["source_version"] = "unauthorized-version"
        return value

    runtime = _runtime(tmp_path, policy_reader=reader, policy_authority=_policy_authority())
    try:
        result = runtime.run(topology="policy_only", policy_query=f"policy {mutation}")
        dispatch = result.dispatches["policy"]
        assert result.status == "FAILED"
        assert dispatch.error_code == expected
        assert dispatch.canonical_result is not None
        assert dispatch.canonical_result.status is ResultStatus.FAILED
    finally:
        runtime.close()


def test_scene_clock_ledger_lock_and_freeze_late_audit_are_deterministic(tmp_path: Path):
    order_id, phone_last4, _carrier, _tracking = _order_row()
    fixed = datetime.now(timezone.utc) + timedelta(days=1)
    runtime = _runtime(tmp_path, scene_clock=fixed)
    try:
        request = runtime.build_request(
            run_id="r-freeze-lock",
            plan_revision_id="p-freeze-lock",
            task_id="order",
            capability_ref="order/read@v1",
            payload={"order_id": order_id, "phone_last4": phone_last4},
        )
        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(pool.map(runtime.dispatch, [request, request]))
        assert all(outcome.status == "SUCCEEDED" for outcome in outcomes)
        assert sum(outcome.duplicate for outcome in outcomes) == 1
        assert runtime.physical_call_counts["order/read@v1"] == 1
        assert all(event.scene_clock == fixed for event in runtime.trace(request.run_id))
        assert all(str(row["scene_clock"]) == fixed.isoformat() for row in runtime.ledger.attempts(request.message_id))

        frozen = runtime.freeze(request.run_id)
        late = runtime.dispatch(request)
        assert late.status == "LATE"
        frozen_again = runtime.freeze(request.run_id)
        assert frozen_again["trace_checksum"] == frozen["trace_checksum"]
        assert frozen_again["trace"] == frozen["trace"]
        assert runtime.ledger.freeze_projection(request.run_id) == (frozen["trace_checksum"], len(frozen["trace"]))
        assert len(runtime.ledger.late_events(request.run_id)) == 1
        assert not any(event.event_type == "A2A_LATE_AUDIT" for event in runtime.trace(request.run_id))
    finally:
        runtime.close()


def test_ecommerce_db_hash_and_read_only_capability_set_are_unchanged(tmp_path: Path):
    before = hashlib.sha256(ORDER_DB.read_bytes()).hexdigest()
    order_id, phone_last4, _carrier, _tracking = _order_row()
    runtime = _runtime(tmp_path)
    try:
        result = runtime.run(topology="order_only", order_id=order_id, phone_last4=phone_last4)
        assert result.status == "SUCCEEDED"
        assert set(runtime.adapters) == {"order/read@v1", "logistics/read@v1", "policy/read@v1"}
        assert all(runtime.registry.get(ref).side_effect == "READ_ONLY" for ref in runtime.registry.refs())
        assert "aftersales/create@v1" not in runtime.registry.refs()
        assert all(count == 0 for ref, count in runtime.physical_call_counts.items() if ref != "order/read@v1")
    finally:
        runtime.close()
    assert hashlib.sha256(ORDER_DB.read_bytes()).hexdigest() == before


def test_pending_dependency_and_parent_bindings_cannot_change_before_execution(tmp_path: Path):
    order_id, phone_last4, carrier, tracking = _order_row()
    runtime = _runtime(tmp_path)
    try:
        upstream_one = runtime.build_request(
            run_id="r-binding-pending",
            plan_revision_id="p-binding-pending",
            task_id="order-one",
            capability_ref="order/read@v1",
            payload={"order_id": order_id, "phone_last4": phone_last4},
        )
        upstream_two = runtime.build_request(
            run_id="r-binding-pending",
            plan_revision_id="p-binding-pending",
            task_id="order-two",
            capability_ref="order/read@v1",
            payload={"order_id": order_id, "phone_last4": phone_last4},
        )
        dependent = runtime.build_request(
            run_id="r-binding-pending",
            plan_revision_id="p-binding-pending",
            task_id="logistics",
            capability_ref="logistics/read@v1",
            payload={"carrier_code": carrier, "tracking_no": tracking},
            parent_message_id=upstream_one.message_id,
            dependency_message_ids=(upstream_one.message_id, upstream_two.message_id),
            idempotency_key="binding-pending-key",
        )
        assert runtime.dispatch(dependent).status == "PENDING"

        reordered = dependent.model_copy(update={"message_id": "reordered", "dependency_message_ids": (upstream_two.message_id, upstream_one.message_id)})
        assert runtime.dispatch(reordered).status == "PENDING"
        removed = dependent.model_copy(update={"message_id": "removed", "dependency_message_ids": (upstream_one.message_id,)})
        assert runtime.dispatch(removed).error_code == "A2A_IDEMPOTENCY_CONFLICT"
        replaced = dependent.model_copy(update={"message_id": "replaced", "dependency_message_ids": ("forged-dependency",)})
        assert runtime.dispatch(replaced).error_code == "A2A_IDEMPOTENCY_CONFLICT"
        parent_changed = dependent.model_copy(update={"message_id": "parent-changed", "parent_message_id": upstream_two.message_id})
        assert runtime.dispatch(parent_changed).error_code == "A2A_IDEMPOTENCY_CONFLICT"
        assert runtime.physical_call_counts["logistics/read@v1"] == 0
    finally:
        runtime.close()


@pytest.mark.parametrize("field", ["correlation_id", "parent_message_id"])
def test_persisted_response_binding_is_verified_even_after_checksum_recompute(tmp_path: Path, field: str):
    order_id, phone_last4, _carrier, _tracking = _order_row()
    ledger_path = tmp_path / f"response-binding-{field}.sqlite"
    runtime = R4A2ARuntime(db_path=ORDER_DB, ledger_path=ledger_path)
    request = runtime.build_request(
        run_id=f"r-response-binding-{field}",
        plan_revision_id="p-response-binding",
        task_id="order",
        capability_ref="order/read@v1",
        payload={"order_id": order_id, "phone_last4": phone_last4},
    )
    assert runtime.dispatch(request).status == "SUCCEEDED"
    runtime.close()
    with sqlite3.connect(ledger_path) as conn:
        response = json.loads(str(conn.execute("SELECT response_json FROM a2a_messages WHERE message_id=?", (request.message_id,)).fetchone()[0]))
        response[field] = "forged-binding"
        conn.execute(
            "UPDATE a2a_messages SET response_json=?,response_checksum=? WHERE message_id=?",
            (json.dumps(response), sha256_json(response), request.message_id),
        )
    restarted = R4A2ARuntime(db_path=ORDER_DB, ledger_path=ledger_path)
    try:
        rejected = restarted.dispatch(request)
        assert rejected.status == "FAILED"
        assert rejected.error_code == "A2A_CORRELATION_MISMATCH"
        assert rejected.canonical_result is None
    finally:
        restarted.close()


def test_persisted_request_safe_projection_is_verified_against_registered_identity(tmp_path: Path):
    order_id, phone_last4, _carrier, _tracking = _order_row()
    ledger_path = tmp_path / "request-projection.sqlite"
    runtime = R4A2ARuntime(db_path=ORDER_DB, ledger_path=ledger_path)
    request = runtime.build_request(
        run_id="r-request-projection",
        plan_revision_id="p-request-projection",
        task_id="order",
        capability_ref="order/read@v1",
        payload={"order_id": order_id, "phone_last4": phone_last4},
    )
    assert runtime.dispatch(request).status == "SUCCEEDED"
    runtime.close()
    with sqlite3.connect(ledger_path) as conn:
        value = json.loads(str(conn.execute("SELECT request_json FROM a2a_messages WHERE message_id=?", (request.message_id,)).fetchone()[0]))
        value["payload"]["order_id"] = "forged-order"
        value["payload_hash"] = sha256_json(value["payload"])
        conn.execute(
            "UPDATE a2a_messages SET request_json=?,request_checksum=? WHERE message_id=?",
            (json.dumps(value), sha256_json(value), request.message_id),
        )
    restarted = R4A2ARuntime(db_path=ORDER_DB, ledger_path=ledger_path)
    try:
        rejected = restarted.dispatch(request)
        assert rejected.status == "FAILED"
        assert rejected.error_code == "A2A_PERSISTED_REQUEST_PAYLOAD_MISMATCH"
    finally:
        restarted.close()


@pytest.mark.parametrize("artifact", ["specialist_json", "canonical_json"])
def test_persisted_specialist_and_canonical_semantics_are_verified_after_rehash(tmp_path: Path, artifact: str):
    order_id, phone_last4, _carrier, _tracking = _order_row()
    ledger_path = tmp_path / f"artifact-semantic-{artifact}.sqlite"
    runtime = R4A2ARuntime(db_path=ORDER_DB, ledger_path=ledger_path)
    request = runtime.build_request(
        run_id=f"r-artifact-{artifact}",
        plan_revision_id="p-artifact",
        task_id="order",
        capability_ref="order/read@v1",
        payload={"order_id": order_id, "phone_last4": phone_last4},
    )
    assert runtime.dispatch(request).status == "SUCCEEDED"
    runtime.close()
    with sqlite3.connect(ledger_path) as conn:
        value = json.loads(str(conn.execute(f"SELECT {artifact} FROM a2a_messages WHERE message_id=?", (request.message_id,)).fetchone()[0]))
        value["payload"]["order_id"] = "forged-order"
        if artifact == "specialist_json":
            value["payload_hash"] = sha256_json(value["payload"])
            checksum_column = "specialist_checksum"
        else:
            value["payload_hash"] = sha256_json(value["payload"])
            checksum_column = "canonical_checksum"
        conn.execute(
            f"UPDATE a2a_messages SET {artifact}=?,{checksum_column}=? WHERE message_id=?",
            (json.dumps(value), sha256_json(value), request.message_id),
        )
    restarted = R4A2ARuntime(db_path=ORDER_DB, ledger_path=ledger_path)
    try:
        rejected = restarted.dispatch(request)
        assert rejected.status == "FAILED"
        assert rejected.canonical_result is None
        assert rejected.error_code in {
            "A2A_RESULT_PAYLOAD_HASH_MISMATCH",
            "A2A_SEMANTIC_WRONG",
            "A2A_PERSISTED_RESULT_SPECIALIST_MISMATCH",
            "A2A_PERSISTED_RESULT_SEMANTIC_MISMATCH",
        }
    finally:
        restarted.close()


def test_duplicate_rehydrates_after_caller_mutates_previous_result(tmp_path: Path):
    order_id, phone_last4, _carrier, _tracking = _order_row()
    runtime = _runtime(tmp_path)
    try:
        request = runtime.build_request(
            run_id="r-caller-mutation",
            plan_revision_id="p-caller-mutation",
            task_id="order",
            capability_ref="order/read@v1",
            payload={"order_id": order_id, "phone_last4": phone_last4},
        )
        first = runtime.dispatch(request)
        assert first.status == "SUCCEEDED"
        assert first.specialist_result is not None and isinstance(first.specialist_result.payload, dict)
        first.specialist_result.payload["order_id"] = "caller-forged"
        assert first.canonical_result is not None and isinstance(first.canonical_result.payload, dict)
        first.canonical_result.payload["order_id"] = "caller-forged"
        duplicate = runtime.dispatch(request)
        assert duplicate.status == "SUCCEEDED"
        assert duplicate.specialist_result is not None
        assert duplicate.specialist_result.payload["order_id"] == order_id
        assert duplicate.canonical_result is not None
        assert duplicate.canonical_result.payload["order_id"] == order_id
    finally:
        runtime.close()


def test_superseded_revision_survives_runtime_restart_as_late_audit(tmp_path: Path):
    order_id, phone_last4, _carrier, _tracking = _order_row()
    ledger_path = tmp_path / "persistent-supersede.sqlite"
    first_runtime = R4A2ARuntime(db_path=ORDER_DB, ledger_path=ledger_path)
    request = first_runtime.build_request(
        run_id="r-persistent-supersede",
        plan_revision_id="p-old",
        task_id="order",
        capability_ref="order/read@v1",
        payload={"order_id": order_id, "phone_last4": phone_last4},
    )
    first_runtime.supersede_revision("r-persistent-supersede", "p-old", "p-new")
    first_runtime.close()
    second_runtime = R4A2ARuntime(db_path=ORDER_DB, ledger_path=ledger_path)
    try:
        late = second_runtime.dispatch(request)
        assert late.status == "LATE"
        assert second_runtime.ledger.late_events("r-persistent-supersede")[0]["reason"] == "REVISION_SUPERSEDED"
        assert second_runtime.ledger.row(request.message_id)["status"] == "LATE"
    finally:
        second_runtime.close()


def test_shared_ledger_freeze_returns_busy_for_other_runtime_inflight(tmp_path: Path):
    import threading

    order_id, phone_last4, _carrier, _tracking = _order_row()
    ledger_path = tmp_path / "shared-freeze.sqlite"
    runtime_one = R4A2ARuntime(db_path=ORDER_DB, ledger_path=ledger_path)
    runtime_two = R4A2ARuntime(db_path=ORDER_DB, ledger_path=ledger_path)
    started = threading.Event()
    release = threading.Event()
    original_adapter = runtime_one.adapters["order/read@v1"]

    class BlockingAdapter:
        agent_ref = original_adapter.agent_ref
        capability_ref = original_adapter.capability_ref

        def invoke(self, payload, *, context):
            started.set()
            assert release.wait(timeout=10)
            return original_adapter.invoke(payload, context=context)

    runtime_one.adapters["order/read@v1"] = BlockingAdapter()
    request = runtime_one.build_request(
        run_id="r-shared-freeze",
        plan_revision_id="p-shared-freeze",
        task_id="order",
        capability_ref="order/read@v1",
        payload={"order_id": order_id, "phone_last4": phone_last4},
    )
    outcome: dict[str, object] = {}

    def dispatch() -> None:
        outcome["value"] = runtime_one.dispatch(request)

    thread = threading.Thread(target=dispatch)
    thread.start()
    try:
        assert started.wait(timeout=10)
        with pytest.raises(A2AContractError, match="A2A_RUN_BUSY"):
            runtime_two.freeze("r-shared-freeze")
        release.set()
        thread.join(timeout=10)
        assert thread.is_alive() is False
        assert outcome["value"].status == "SUCCEEDED"
        frozen = runtime_two.freeze("r-shared-freeze")
        frozen_again = runtime_two.freeze("r-shared-freeze")
        assert frozen_again["trace_checksum"] == frozen["trace_checksum"]
    finally:
        release.set()
        thread.join(timeout=10)
        runtime_one.close()
        runtime_two.close()


def test_real_r35_manifest_authority_binds_evidence_and_rejects_forgery(tmp_path: Path):
    from agent.r3_5_rag import R35Retriever

    retriever = R35Retriever.from_manifest(ROOT / "data/r3_corpus_v1/manifest-r3_5.json")
    reader_adapter_runtime = R4A2ARuntime(db_path=ORDER_DB, ledger_path=tmp_path / "real-authority.sqlite", policy_retriever=retriever, policy_as_of=None, policy_mode="bm25")
    try:
        query = "退货"
        result = reader_adapter_runtime.run(topology="policy_only", policy_query=query)
        assert result.status == "SUCCEEDED"
        dispatch = result.dispatches["policy"]
        assert dispatch.specialist_result is not None and dispatch.specialist_result.ok
        payload = dispatch.specialist_result.payload
        assert payload["source"] == "r3.5.project-authored-kb"
        assert payload["strategy_checksum"] == reader_adapter_runtime.authority_snapshot.strategy_checksum
        assert payload["authority_snapshot_hash"] == reader_adapter_runtime.authority_snapshot.snapshot_hash
        assert any(event.payload.get("authority_snapshot_hash") == reader_adapter_runtime.authority_snapshot.snapshot_hash for event in reader_adapter_runtime.trace(result.run_id))
        snapshot = reader_adapter_runtime.authority_snapshot
        authority = {
            "source": snapshot.source,
            "source_version": snapshot.source_version,
            "version_tuple": list(snapshot.version_tuple),
            "strategy_checksum": snapshot.strategy_checksum,
            "evidence": [dict(item) for item in snapshot.evidence],
        }
        valid = dict(payload)
    finally:
        reader_adapter_runtime.close()

    for field in ("source", "source_version", "chunk_id", "text_hash"):
        def reader(_query: str, field: str = field) -> dict[str, object]:
            value = json.loads(json.dumps(valid))
            if field == "source":
                value["source"] = "forged-corpus"
            elif field == "source_version":
                value["source_version"] = "0" * 64
            else:
                value["evidence"][0][field] = "forged" if field == "chunk_id" else "0" * 64
            return value

        ledger_path = tmp_path / f"forged-authority-{field}.sqlite"
        runtime = R4A2ARuntime(db_path=ORDER_DB, ledger_path=ledger_path, policy_reader=reader, policy_authority=authority)
        try:
            rejected = runtime.run(topology="policy_only", policy_query="退货")
            assert rejected.status == "FAILED"
            assert rejected.dispatches["policy"].error_code in {
                "A2A_POLICY_SOURCE_UNAUTHORIZED",
                "A2A_POLICY_VERSION_UNAUTHORIZED",
                "A2A_POLICY_EVIDENCE_UNAUTHORIZED",
            }
        finally:
            runtime.close()


@pytest.mark.parametrize(
    "field",
    [
        "task_id",
        "correlation_id",
        "sender_ref",
        "receiver_ref",
        "schema_version",
        "deadline",
        "attempt_id",
        "trace_id",
        "idempotency_key",
        "created_at",
    ],
)
def test_persisted_request_envelope_identity_cannot_be_rewritten_with_new_checksum(tmp_path: Path, field: str):
    order_id, phone_last4, _carrier, _tracking = _order_row()
    ledger_path = tmp_path / f"request-envelope-{field}.sqlite"
    runtime = R4A2ARuntime(db_path=ORDER_DB, ledger_path=ledger_path)
    request = runtime.build_request(
        run_id=f"r-request-envelope-{field}",
        plan_revision_id="p-request-envelope",
        task_id="order",
        capability_ref="order/read@v1",
        payload={"order_id": order_id, "phone_last4": phone_last4},
    )
    assert runtime.dispatch(request).status == "SUCCEEDED"
    runtime.close()
    with sqlite3.connect(ledger_path) as conn:
        value = json.loads(str(conn.execute("SELECT request_json FROM a2a_messages WHERE message_id=?", (request.message_id,)).fetchone()[0]))
        if field in {"deadline", "created_at"}:
            original = datetime.fromisoformat(str(value[field]).replace("Z", "+00:00"))
            value[field] = (original + timedelta(seconds=1)).isoformat().replace("+00:00", "Z")
        elif field == "schema_version":
            value[field] = "r4.a2a.message.v0"
        elif field == "sender_ref":
            value[field] = "forged-sender@v1"
        elif field == "receiver_ref":
            value[field] = "logistics-agent@v1"
        else:
            value[field] = f"forged-{field}"
        conn.execute(
            "UPDATE a2a_messages SET request_json=?,request_checksum=? WHERE message_id=?",
            (json.dumps(value), sha256_json(value), request.message_id),
        )
    restarted = R4A2ARuntime(db_path=ORDER_DB, ledger_path=ledger_path)
    try:
        rejected = restarted.dispatch(request)
        assert rejected.status == "FAILED"
        assert rejected.canonical_result is None
        assert rejected.duplicate is True
        assert rejected.error_code.startswith("A2A_PERSISTED_REQUEST_")
    finally:
        restarted.close()


@pytest.mark.parametrize("field", ["message_id", "deadline", "attempt_id", "trace_id", "idempotency_key", "created_at"])
def test_persisted_response_envelope_identity_cannot_be_rewritten_with_new_checksum(tmp_path: Path, field: str):
    order_id, phone_last4, _carrier, _tracking = _order_row()
    ledger_path = tmp_path / f"response-envelope-{field}.sqlite"
    runtime = R4A2ARuntime(db_path=ORDER_DB, ledger_path=ledger_path)
    request = runtime.build_request(
        run_id=f"r-response-envelope-{field}",
        plan_revision_id="p-response-envelope",
        task_id="order",
        capability_ref="order/read@v1",
        payload={"order_id": order_id, "phone_last4": phone_last4},
    )
    assert runtime.dispatch(request).status == "SUCCEEDED"
    runtime.close()
    with sqlite3.connect(ledger_path) as conn:
        value = json.loads(str(conn.execute("SELECT response_json FROM a2a_messages WHERE message_id=?", (request.message_id,)).fetchone()[0]))
        if field in {"deadline", "created_at"}:
            original = datetime.fromisoformat(str(value[field]).replace("Z", "+00:00"))
            value[field] = (original + timedelta(seconds=1)).isoformat().replace("+00:00", "Z")
        elif field == "message_id":
            value[field] = "forged-response-id"
        else:
            value[field] = f"forged-{field}"
        conn.execute(
            "UPDATE a2a_messages SET response_json=?,response_checksum=? WHERE message_id=?",
            (json.dumps(value), sha256_json(value), request.message_id),
        )
    restarted = R4A2ARuntime(db_path=ORDER_DB, ledger_path=ledger_path)
    try:
        rejected = restarted.dispatch(request)
        assert rejected.status == "FAILED"
        assert rejected.canonical_result is None
        assert rejected.duplicate is True
        assert rejected.error_code.startswith("A2A_")
    finally:
        restarted.close()


@pytest.mark.parametrize("field", ["task_id", "correlation_id"])
def test_persisted_request_and_response_identity_cannot_be_rewritten_together(tmp_path: Path, field: str):
    order_id, phone_last4, _carrier, _tracking = _order_row()
    ledger_path = tmp_path / f"request-response-cross-{field}.sqlite"
    runtime = R4A2ARuntime(db_path=ORDER_DB, ledger_path=ledger_path)
    request = runtime.build_request(
        run_id=f"r-request-response-cross-{field}",
        plan_revision_id="p-request-response-cross",
        task_id="order",
        capability_ref="order/read@v1",
        payload={"order_id": order_id, "phone_last4": phone_last4},
    )
    assert runtime.dispatch(request).status == "SUCCEEDED"
    runtime.close()
    with sqlite3.connect(ledger_path) as conn:
        request_value = json.loads(str(conn.execute("SELECT request_json FROM a2a_messages WHERE message_id=?", (request.message_id,)).fetchone()[0]))
        response_value = json.loads(str(conn.execute("SELECT response_json FROM a2a_messages WHERE message_id=?", (request.message_id,)).fetchone()[0]))
        forged = f"forged-{field}"
        request_value[field] = forged
        response_value[field] = forged
        conn.execute(
            "UPDATE a2a_messages SET request_json=?,request_checksum=?,response_json=?,response_checksum=? WHERE message_id=?",
            (json.dumps(request_value), sha256_json(request_value), json.dumps(response_value), sha256_json(response_value), request.message_id),
        )
    restarted = R4A2ARuntime(db_path=ORDER_DB, ledger_path=ledger_path)
    try:
        rejected = restarted.dispatch(request)
        assert rejected.status == "FAILED"
        assert rejected.canonical_result is None
    finally:
        restarted.close()


def test_freeze_blocks_registered_and_pending_until_terminal(tmp_path: Path):
    order_id, phone_last4, carrier, tracking = _order_row()
    runtime = _runtime(tmp_path)
    try:
        registered = runtime.build_request(
            run_id="r-freeze-registered",
            plan_revision_id="p-freeze-registered",
            task_id="order",
            capability_ref="order/read@v1",
            payload={"order_id": order_id, "phone_last4": phone_last4},
        )
        with pytest.raises(A2AContractError, match="A2A_RUN_BUSY"):
            runtime.freeze(registered.run_id)
        assert runtime.dispatch(registered).status == "SUCCEEDED"
        assert runtime.freeze(registered.run_id)["status"] == "FROZEN"

        upstream = runtime.build_request(
            run_id="r-freeze-pending",
            plan_revision_id="p-freeze-pending",
            task_id="order",
            capability_ref="order/read@v1",
            payload={"order_id": order_id, "phone_last4": phone_last4},
        )
        pending = runtime.build_request(
            run_id="r-freeze-pending",
            plan_revision_id="p-freeze-pending",
            task_id="dependent-order",
            capability_ref="order/read@v1",
            payload={"order_id": order_id, "phone_last4": phone_last4},
            dependency_message_ids=(upstream.message_id,),
        )
        assert runtime.dispatch(pending).status == "PENDING"
        with pytest.raises(A2AContractError, match="A2A_RUN_BUSY"):
            runtime.freeze("r-freeze-pending")
        assert runtime.dispatch(upstream).status == "SUCCEEDED"
        assert runtime.dispatch(pending).status == "SUCCEEDED"
        assert runtime.freeze("r-freeze-pending")["status"] == "FROZEN"
    finally:
        runtime.close()


def test_supersede_during_inflight_return_is_late_and_new_plan_is_clean(tmp_path: Path):
    import threading

    order_id, phone_last4, _carrier, _tracking = _order_row()
    ledger_path = tmp_path / "supersede-inflight.sqlite"
    runtime_one = R4A2ARuntime(db_path=ORDER_DB, ledger_path=ledger_path)
    runtime_two = R4A2ARuntime(db_path=ORDER_DB, ledger_path=ledger_path)
    started = threading.Event()
    release = threading.Event()
    original_adapter = runtime_one.adapters["order/read@v1"]

    class BlockingAdapter:
        agent_ref = original_adapter.agent_ref
        capability_ref = original_adapter.capability_ref

        def invoke(self, payload, *, context):
            started.set()
            assert release.wait(timeout=10)
            return original_adapter.invoke(payload, context=context)

    runtime_one.adapters["order/read@v1"] = BlockingAdapter()
    old = runtime_one.build_request(
        run_id="r-supersede-inflight",
        plan_revision_id="p-old",
        task_id="old-order",
        capability_ref="order/read@v1",
        payload={"order_id": order_id, "phone_last4": phone_last4},
        idempotency_key="supersede-old",
    )
    outcome: dict[str, object] = {}

    def dispatch_old() -> None:
        outcome["old"] = runtime_one.dispatch(old)

    thread = threading.Thread(target=dispatch_old)
    thread.start()
    try:
        assert started.wait(timeout=10)
        runtime_two.supersede_revision("r-supersede-inflight", "p-old", "p-new")
        new = runtime_two.build_request(
            run_id="r-supersede-inflight",
            plan_revision_id="p-new",
            task_id="new-order",
            capability_ref="order/read@v1",
            payload={"order_id": order_id, "phone_last4": phone_last4},
            idempotency_key="supersede-new",
        )
        assert runtime_two.dispatch(new).status == "SUCCEEDED"
        release.set()
        thread.join(timeout=10)
        assert thread.is_alive() is False
        assert outcome["old"].status == "LATE"
        assert outcome["old"].canonical_result is None
        row = runtime_two.ledger.row(old.message_id)
        assert row["status"] == "LATE"
        assert row["canonical_json"] is None
        assert row["specialist_json"] is None
        assert all(str(attempt["status"]) == "LATE" for attempt in runtime_two.ledger.attempts(old.message_id))
        assert runtime_two.ledger.late_events("r-supersede-inflight")[0]["reason"] == "REVISION_SUPERSEDED"
        assert runtime_two.ledger.row(new.message_id)["status"] == "SUCCEEDED"
    finally:
        release.set()
        thread.join(timeout=10)
        runtime_one.close()
        runtime_two.close()


def test_cancel_during_inflight_return_is_late_and_has_no_canonical_result(tmp_path: Path):
    import threading

    order_id, phone_last4, _carrier, _tracking = _order_row()
    ledger_path = tmp_path / "cancel-inflight.sqlite"
    runtime_one = R4A2ARuntime(db_path=ORDER_DB, ledger_path=ledger_path)
    runtime_two = R4A2ARuntime(db_path=ORDER_DB, ledger_path=ledger_path)
    started = threading.Event()
    release = threading.Event()
    original_adapter = runtime_one.adapters["order/read@v1"]

    class BlockingAdapter:
        agent_ref = original_adapter.agent_ref
        capability_ref = original_adapter.capability_ref

        def invoke(self, payload, *, context):
            started.set()
            assert release.wait(timeout=10)
            return original_adapter.invoke(payload, context=context)

    runtime_one.adapters["order/read@v1"] = BlockingAdapter()
    request = runtime_one.build_request(
        run_id="r-cancel-inflight",
        plan_revision_id="p-cancel-inflight",
        task_id="order",
        capability_ref="order/read@v1",
        payload={"order_id": order_id, "phone_last4": phone_last4},
        idempotency_key="cancel-inflight",
    )
    outcome: dict[str, object] = {}

    def dispatch() -> None:
        outcome["value"] = runtime_one.dispatch(request)

    thread = threading.Thread(target=dispatch)
    thread.start()
    try:
        assert started.wait(timeout=10)
        runtime_two.cancel("r-cancel-inflight")
        release.set()
        thread.join(timeout=10)
        assert thread.is_alive() is False
        assert outcome["value"].status == "LATE"
        assert outcome["value"].canonical_result is None
        row = runtime_two.ledger.row(request.message_id)
        assert row["status"] == "LATE"
        assert row["canonical_json"] is None
        assert runtime_two.ledger.late_events("r-cancel-inflight")[0]["reason"] == "RUN_CANCELLED"
    finally:
        release.set()
        thread.join(timeout=10)
        runtime_one.close()
        runtime_two.close()


def test_terminal_finalize_and_freeze_cannot_observe_partial_bundle(tmp_path: Path):
    import threading

    order_id, phone_last4, _carrier, _tracking = _order_row()
    ledger_path = tmp_path / "terminal-finalize-freeze.sqlite"
    runtime_one = R4A2ARuntime(db_path=ORDER_DB, ledger_path=ledger_path)
    runtime_two = R4A2ARuntime(db_path=ORDER_DB, ledger_path=ledger_path)
    entered_final_trace = threading.Event()
    release_final_trace = threading.Event()
    freeze_started = threading.Event()
    freeze_finished = threading.Event()
    original_append = runtime_one.ledger._append_trace_locked
    freeze_outcome: dict[str, object] = {}

    def blocking_append(**kwargs):
        if kwargs.get("event_type") == "A2A_RESULT_VERIFIED":
            entered_final_trace.set()
            assert release_final_trace.wait(timeout=10)
        return original_append(**kwargs)

    runtime_one.ledger._append_trace_locked = blocking_append
    request = runtime_one.build_request(
        run_id="r-terminal-finalize-freeze",
        plan_revision_id="p-terminal-finalize-freeze",
        task_id="order",
        capability_ref="order/read@v1",
        payload={"order_id": order_id, "phone_last4": phone_last4},
    )

    def dispatch() -> None:
        freeze_outcome["dispatch"] = runtime_one.dispatch(request)

    def freeze() -> None:
        freeze_started.set()
        try:
            freeze_outcome["freeze"] = runtime_two.freeze(request.run_id)
        except Exception as exc:  # pragma: no cover - assertion reports it below
            freeze_outcome["freeze_error"] = exc
        finally:
            freeze_finished.set()

    dispatch_thread = threading.Thread(target=dispatch)
    dispatch_thread.start()
    try:
        assert entered_final_trace.wait(timeout=10)
        freeze_thread = threading.Thread(target=freeze)
        freeze_thread.start()
        assert freeze_started.wait(timeout=10)
        assert not freeze_finished.wait(timeout=0.2)
        release_final_trace.set()
        dispatch_thread.join(timeout=10)
        freeze_thread.join(timeout=10)
        assert dispatch_thread.is_alive() is False
        assert freeze_thread.is_alive() is False
        assert "freeze_error" not in freeze_outcome
        assert freeze_outcome["dispatch"].status == "SUCCEEDED"
        bundle = freeze_outcome["freeze"]
        assert any(event["event_type"] == "A2A_RESULT_VERIFIED" for event in bundle["trace"])
        assert runtime_two.ledger.row(request.message_id)["canonical_json"] is not None
    finally:
        release_final_trace.set()
        dispatch_thread.join(timeout=10)
        runtime_one.close()
        runtime_two.close()


def test_duplicate_rejects_success_artifacts_rewritten_to_earlier_failed_attempt(tmp_path: Path):
    order_id, phone_last4, _carrier, _tracking = _order_row()
    ledger_path = tmp_path / "latest-attempt-tamper.sqlite"
    runtime = R4A2ARuntime(
        db_path=ORDER_DB,
        ledger_path=ledger_path,
        failure_script={"order": {"kind": "PHYSICAL_FAILURE", "count": 1}},
    )
    request = runtime.build_request(
        run_id="r-latest-attempt-tamper",
        plan_revision_id="p-latest-attempt-tamper",
        task_id="order",
        capability_ref="order/read@v1",
        payload={"order_id": order_id, "phone_last4": phone_last4},
    )
    assert runtime.dispatch(request).status == "SUCCEEDED"
    attempts = runtime.ledger.attempts(request.message_id)
    assert len(attempts) == 2
    first_attempt_id = str(attempts[0]["attempt_id"])
    runtime.close()
    with sqlite3.connect(ledger_path) as conn:
        response = json.loads(str(conn.execute("SELECT response_json FROM a2a_messages WHERE message_id=?", (request.message_id,)).fetchone()[0]))
        canonical = json.loads(str(conn.execute("SELECT canonical_json FROM a2a_messages WHERE message_id=?", (request.message_id,)).fetchone()[0]))
        response["attempt_id"] = first_attempt_id
        response["message_id"] = f"{request.message_id}:result:{first_attempt_id}"
        canonical["attempt_id"] = first_attempt_id
        conn.execute(
            "UPDATE a2a_messages SET response_json=?,response_checksum=?,response_message_id=?,response_attempt_id=?,canonical_json=?,canonical_checksum=? WHERE message_id=?",
            (json.dumps(response), sha256_json(response), response["message_id"], first_attempt_id, json.dumps(canonical), sha256_json(canonical), request.message_id),
        )
    restarted = R4A2ARuntime(db_path=ORDER_DB, ledger_path=ledger_path)
    try:
        rejected = restarted.dispatch(request)
        assert rejected.status == "FAILED"
        assert rejected.duplicate is True
        assert rejected.canonical_result is None
        assert rejected.error_code in {"A2A_PERSISTED_LATEST_ATTEMPT_MISMATCH", "A2A_PERSISTED_LATEST_ATTEMPT_STATUS_MISMATCH"}
    finally:
        restarted.close()


@pytest.mark.parametrize(
    ("failure_script", "expected_status"),
    [
        ({"order": {"kind": "PHYSICAL_FAILURE", "count": 1}}, "SUCCEEDED"),
        ({"order": {"kind": "PHYSICAL_FAILURE", "count": 2}}, "FAILED"),
    ],
)
def test_duplicate_requires_latest_ended_attempt_status_error_and_usage(tmp_path: Path, failure_script: dict[str, object], expected_status: str):
    order_id, phone_last4, _carrier, _tracking = _order_row()
    runtime = _runtime(tmp_path, failure_script=failure_script)
    try:
        request = runtime.build_request(
            run_id=f"r-latest-{expected_status.lower()}",
            plan_revision_id="p-latest",
            task_id="order",
            capability_ref="order/read@v1",
            payload={"order_id": order_id, "phone_last4": phone_last4},
        )
        first = runtime.dispatch(request)
        assert first.status == expected_status
        attempts = runtime.ledger.attempts(request.message_id)
        assert len(attempts) == 2
        latest = attempts[-1]
        assert latest["status"] == expected_status
        assert latest["ended_at"] is not None
        assert first.canonical_result is not None
        assert first.canonical_result.attempt_id == latest["attempt_id"]
        assert first.canonical_result.usage.physical_attempts == sum(int(row["physical_call"]) for row in attempts)
        if expected_status == "FAILED":
            assert latest["error_code"] == first.canonical_result.error_ref == first.response.error.code
        else:
            assert latest["error_code"] is None
        duplicate = runtime.dispatch(request)
        assert duplicate.status == expected_status
        assert duplicate.canonical_result is not None
        assert duplicate.canonical_result.attempt_id == latest["attempt_id"]
    finally:
        runtime.close()


def test_terminal_run_rejects_new_build_and_registration_without_new_rows(tmp_path: Path):
    order_id, phone_last4, _carrier, _tracking = _order_row()
    runtime = _runtime(tmp_path)
    try:
        runtime.start_run(run_id="r-terminal-freeze", plan_revision_id="p-terminal-freeze")
        runtime.freeze("r-terminal-freeze")
        before = runtime.ledger.conn.execute("SELECT COUNT(*) FROM a2a_messages WHERE run_id=?", ("r-terminal-freeze",)).fetchone()[0]
        with pytest.raises(A2AContractError, match="A2A_RUN_TERMINAL"):
            runtime.build_request(run_id="r-terminal-freeze", plan_revision_id="p-terminal-freeze", task_id="order", capability_ref="order/read@v1", payload={"order_id": order_id, "phone_last4": phone_last4})
        assert runtime.ledger.conn.execute("SELECT COUNT(*) FROM a2a_messages WHERE run_id=?", ("r-terminal-freeze",)).fetchone()[0] == before

        runtime.start_run(run_id="r-terminal-cancel", plan_revision_id="p-terminal-cancel")
        runtime.cancel("r-terminal-cancel")
        before = runtime.ledger.conn.execute("SELECT COUNT(*) FROM a2a_messages WHERE run_id=?", ("r-terminal-cancel",)).fetchone()[0]
        with pytest.raises(A2AContractError, match="A2A_RUN_TERMINAL"):
            runtime.build_request(run_id="r-terminal-cancel", plan_revision_id="p-terminal-cancel", task_id="order", capability_ref="order/read@v1", payload={"order_id": order_id, "phone_last4": phone_last4})
        assert runtime.ledger.conn.execute("SELECT COUNT(*) FROM a2a_messages WHERE run_id=?", ("r-terminal-cancel",)).fetchone()[0] == before

        runtime.start_run(run_id="r-terminal-supersede", plan_revision_id="p-terminal-supersede")
        runtime.supersede_revision("r-terminal-supersede", "p-terminal-supersede")
        before = runtime.ledger.conn.execute("SELECT COUNT(*) FROM a2a_messages WHERE run_id=?", ("r-terminal-supersede",)).fetchone()[0]
        with pytest.raises(A2AContractError, match="A2A_REVISION_SUPERSEDED"):
            runtime.build_request(run_id="r-terminal-supersede", plan_revision_id="p-terminal-supersede", task_id="order", capability_ref="order/read@v1", payload={"order_id": order_id, "phone_last4": phone_last4})
        assert runtime.ledger.conn.execute("SELECT COUNT(*) FROM a2a_messages WHERE run_id=?", ("r-terminal-supersede",)).fetchone()[0] == before
    finally:
        runtime.close()


def test_registered_message_blocks_freeze_across_two_runtime_connections(tmp_path: Path):
    order_id, phone_last4, _carrier, _tracking = _order_row()
    ledger_path = tmp_path / "registered-freeze-race.sqlite"
    runtime_one = R4A2ARuntime(db_path=ORDER_DB, ledger_path=ledger_path)
    runtime_two = R4A2ARuntime(db_path=ORDER_DB, ledger_path=ledger_path)
    try:
        request = runtime_one.build_request(
            run_id="r-registered-freeze-race",
            plan_revision_id="p-registered-freeze-race",
            task_id="order",
            capability_ref="order/read@v1",
            payload={"order_id": order_id, "phone_last4": phone_last4},
        )
        with pytest.raises(A2AContractError, match="A2A_RUN_BUSY"):
            runtime_two.freeze(request.run_id)
        assert runtime_one.dispatch(request).status == "SUCCEEDED"
        frozen = runtime_two.freeze(request.run_id)
        assert any(event["event_type"] == "A2A_RESULT_VERIFIED" for event in frozen["trace"])
    finally:
        runtime_one.close()
        runtime_two.close()


def test_repeated_freeze_fails_closed_on_injected_nonterminal_row_and_tampered_trace(tmp_path: Path):
    order_id, phone_last4, _carrier, _tracking = _order_row()
    runtime = _runtime(tmp_path)
    try:
        request = runtime.build_request(
            run_id="r-freeze-integrity",
            plan_revision_id="p-freeze-integrity",
            task_id="order",
            capability_ref="order/read@v1",
            payload={"order_id": order_id, "phone_last4": phone_last4},
        )
        assert runtime.dispatch(request).status == "SUCCEEDED"
        frozen = runtime.freeze(request.run_id)
        runtime.ledger.conn.execute("UPDATE a2a_messages SET status='PENDING' WHERE message_id=?", (request.message_id,))
        with pytest.raises(A2AContractError, match="A2A_RUN_BUSY"):
            runtime.freeze(request.run_id)
        runtime.ledger.conn.execute("UPDATE a2a_messages SET status='SUCCEEDED' WHERE message_id=?", (request.message_id,))
        runtime.ledger.conn.execute("UPDATE a2a_trace SET payload_json=? WHERE run_id=? AND event_type='A2A_RESULT_VERIFIED'", ("{}", request.run_id))
        with pytest.raises(A2AContractError, match="A2A_FREEZE_CHECKSUM_MISMATCH"):
            runtime.freeze(request.run_id)
        final_trace = next(event for event in frozen["trace"] if event["event_type"] == "A2A_RESULT_VERIFIED")
        runtime.ledger.conn.execute("UPDATE a2a_trace SET payload_json=? WHERE run_id=? AND event_type='A2A_RESULT_VERIFIED'", (json.dumps(final_trace["payload"]), request.run_id))
    finally:
        runtime.close()


def test_empty_run_freeze_zero_event_count_is_stable_across_replay(tmp_path: Path):
    runtime = _runtime(tmp_path)
    try:
        run_id = "r-empty-freeze"
        runtime.start_run(run_id=run_id, plan_revision_id="p-empty-freeze")
        first = runtime.freeze(run_id)
        second = runtime.freeze(run_id)
        assert first["trace"] == second["trace"] == []
        assert first["trace_checksum"] == second["trace_checksum"]
        assert runtime.ledger.freeze_projection(run_id) == (first["trace_checksum"], 0)
        stored = runtime.ledger.conn.execute("SELECT freeze_event_count FROM a2a_runs WHERE run_id=?", (run_id,)).fetchone()[0]
        assert stored is not None
        assert int(stored) == 0
    finally:
        runtime.close()


@pytest.mark.parametrize("field", ["trace_id", "run_id", "message_id", "attempt_id", "event_type"])
def test_duplicate_rejects_terminal_trace_provenance_tampering(tmp_path: Path, field: str):
    order_id, phone_last4, _carrier, _tracking = _order_row()
    ledger_path = tmp_path / f"terminal-trace-provenance-{field}.sqlite"
    runtime = R4A2ARuntime(db_path=ORDER_DB, ledger_path=ledger_path)
    request = runtime.build_request(
        run_id=f"r-terminal-trace-provenance-{field}",
        plan_revision_id="p-terminal-trace-provenance",
        task_id="order",
        capability_ref="order/read@v1",
        payload={"order_id": order_id, "phone_last4": phone_last4},
    )
    assert runtime.dispatch(request).status == "SUCCEEDED"
    runtime.close()
    replacements = {
        "trace_id": "forged-trace",
        "run_id": "forged-run",
        "message_id": "forged-message",
        "attempt_id": "forged-attempt",
        "event_type": "A2A_MESSAGE_FAILED",
    }
    with sqlite3.connect(ledger_path) as conn:
        conn.execute(
            f"UPDATE a2a_trace SET {field}=? WHERE run_id=? AND message_id=? AND event_type='A2A_RESULT_VERIFIED'",
            (replacements[field], request.run_id, request.message_id),
        )
    restarted = R4A2ARuntime(db_path=ORDER_DB, ledger_path=ledger_path)
    try:
        rejected = restarted.dispatch(request)
        assert rejected.status == "FAILED"
        assert rejected.duplicate is True
        assert rejected.canonical_result is None
        assert rejected.error_code.startswith("A2A_PERSISTED_TERMINAL_TRACE_")
    finally:
        restarted.close()


@pytest.mark.parametrize("initial_state", ["REGISTERED", "PENDING"])
def test_supersede_converges_old_nonterminal_messages_and_unblocks_new_plan(tmp_path: Path, initial_state: str):
    order_id, phone_last4, carrier, tracking = _order_row()
    runtime = _runtime(tmp_path)
    try:
        upstream = runtime.build_request(
            run_id=f"r-supersede-converge-{initial_state.lower()}",
            plan_revision_id="p-old-converge",
            task_id="order",
            capability_ref="order/read@v1",
            payload={"order_id": order_id, "phone_last4": phone_last4},
        )
        old_messages = [upstream]
        if initial_state == "PENDING":
            dependent = runtime.build_request(
                run_id=upstream.run_id,
                plan_revision_id=upstream.plan_revision_id,
                task_id="logistics",
                capability_ref="logistics/read@v1",
                payload={"carrier_code": carrier, "tracking_no": tracking, "phone_last4": phone_last4},
                dependency_message_ids=(upstream.message_id,),
                idempotency_key=f"pending-{initial_state.lower()}",
            )
            assert runtime.dispatch(dependent).status == "PENDING"
            old_messages.append(dependent)

        runtime.supersede_revision(upstream.run_id, "p-old-converge", "p-new-converge")
        for old in old_messages:
            row = runtime.ledger.row(old.message_id)
            assert row["status"] == "LATE"
            assert row["response_json"] is None
            assert row["specialist_json"] is None
            assert row["canonical_json"] is None
            assert runtime.ledger.attempts(old.message_id) == []
            assert runtime.ledger.late_events(upstream.run_id)[-1]["reason"] == "REVISION_SUPERSEDED"
            assert runtime.dispatch(old).status == "LATE"
            assert runtime.dispatch(old).status == "LATE"
        assert runtime.physical_call_counts["order/read@v1"] == 0
        assert runtime.physical_call_counts["logistics/read@v1"] == 0

        new = runtime.build_request(
            run_id=upstream.run_id,
            plan_revision_id="p-new-converge",
            task_id="new-order",
            capability_ref="order/read@v1",
            payload={"order_id": order_id, "phone_last4": phone_last4},
        )
        assert runtime.dispatch(new).status == "SUCCEEDED"
        assert runtime.freeze(upstream.run_id)["status"] == "FROZEN"
    finally:
        runtime.close()


@pytest.mark.parametrize("initial_state", ["REGISTERED", "PENDING"])
def test_cancel_converges_old_nonterminal_messages_without_execution(tmp_path: Path, initial_state: str):
    order_id, phone_last4, carrier, tracking = _order_row()
    runtime = _runtime(tmp_path)
    try:
        upstream = runtime.build_request(
            run_id=f"r-cancel-converge-{initial_state.lower()}",
            plan_revision_id="p-cancel-converge",
            task_id="order",
            capability_ref="order/read@v1",
            payload={"order_id": order_id, "phone_last4": phone_last4},
        )
        old_messages = [upstream]
        if initial_state == "PENDING":
            dependent = runtime.build_request(
                run_id=upstream.run_id,
                plan_revision_id=upstream.plan_revision_id,
                task_id="logistics",
                capability_ref="logistics/read@v1",
                payload={"carrier_code": carrier, "tracking_no": tracking, "phone_last4": phone_last4},
                dependency_message_ids=(upstream.message_id,),
                idempotency_key=f"cancel-pending-{initial_state.lower()}",
            )
            assert runtime.dispatch(dependent).status == "PENDING"
            old_messages.append(dependent)

        runtime.cancel(upstream.run_id)
        for old in old_messages:
            row = runtime.ledger.row(old.message_id)
            assert row["status"] == "LATE"
            assert row["response_json"] is None
            assert row["specialist_json"] is None
            assert row["canonical_json"] is None
            assert runtime.ledger.attempts(old.message_id) == []
            assert runtime.dispatch(old).status == "LATE"
            assert runtime.dispatch(old).status == "LATE"
        assert all(event["reason"] == "RUN_CANCELLED" for event in runtime.ledger.late_events(upstream.run_id))
        assert runtime.physical_call_counts["order/read@v1"] == 0
        assert runtime.physical_call_counts["logistics/read@v1"] == 0
    finally:
        runtime.close()


@pytest.mark.parametrize("action", ["supersede", "cancel"])
def test_two_runtime_lifecycle_change_race_never_leaves_old_message_live(tmp_path: Path, action: str):
    import threading

    order_id, phone_last4, _carrier, _tracking = _order_row()
    ledger_path = tmp_path / f"lifecycle-race-{action}.sqlite"
    runtime_one = R4A2ARuntime(db_path=ORDER_DB, ledger_path=ledger_path)
    runtime_two = R4A2ARuntime(db_path=ORDER_DB, ledger_path=ledger_path)
    request = runtime_one.build_request(
        run_id=f"r-lifecycle-race-{action}",
        plan_revision_id="p-race-old",
        task_id="order",
        capability_ref="order/read@v1",
        payload={"order_id": order_id, "phone_last4": phone_last4},
    )
    barrier = threading.Barrier(2)
    outcomes: dict[str, object] = {}

    def dispatch() -> None:
        try:
            barrier.wait(timeout=10)
            outcomes["dispatch"] = runtime_one.dispatch(request)
        except Exception as exc:  # pragma: no cover - assertion reports it below
            outcomes["dispatch_error"] = exc

    def lifecycle_change() -> None:
        try:
            barrier.wait(timeout=10)
            if action == "supersede":
                runtime_two.supersede_revision(request.run_id, "p-race-old", "p-race-new")
            else:
                runtime_two.cancel(request.run_id)
            outcomes["changed"] = True
        except Exception as exc:  # pragma: no cover - assertion reports it below
            outcomes["change_error"] = exc

    dispatch_thread = threading.Thread(target=dispatch)
    change_thread = threading.Thread(target=lifecycle_change)
    dispatch_thread.start()
    change_thread.start()
    dispatch_thread.join(timeout=15)
    change_thread.join(timeout=15)
    try:
        assert dispatch_thread.is_alive() is False
        assert change_thread.is_alive() is False
        assert "dispatch_error" not in outcomes
        assert "change_error" not in outcomes
        row = runtime_two.ledger.row(request.message_id)
        assert row["status"] in {"LATE", "SUCCEEDED"}
        if row["status"] == "LATE":
            assert row["canonical_json"] is None
        else:
            assert row["canonical_json"] is not None
        assert all(str(attempt["status"]) in {"LATE", "SUCCEEDED", "FAILED"} for attempt in runtime_two.ledger.attempts(request.message_id))
        assert len(runtime_two.ledger.late_events(request.run_id)) <= 1
        assert runtime_one.physical_call_counts["order/read@v1"] <= 1
        assert runtime_two.dispatch(request).status == "LATE"
    finally:
        runtime_one.close()
        runtime_two.close()


TRACE_IDENTITY_FIELDS = [
    "event_id",
    "seq_no",
    "actor",
    "scene_clock",
    "trace_id",
    "event_type",
    "message_id",
    "attempt_id",
    "run_id",
    "payload_hash",
]


def _tamper_terminal_trace(ledger_path: Path, request, field: str) -> None:
    values = {
        "event_id": "forged-event-id",
        "seq_no": 999,
        "actor": "forged-actor",
        "scene_clock": "2026-09-09T12:34:56+00:00",
        "trace_id": "forged-trace-id",
        "event_type": "A2A_MESSAGE_FAILED",
        "message_id": "forged-message-id",
        "attempt_id": "forged-attempt-id",
        "run_id": "forged-run-id",
        "payload_hash": "0" * 64,
    }
    with sqlite3.connect(ledger_path) as conn:
        conn.execute(
            "UPDATE a2a_trace SET " + field + "=? WHERE run_id=? AND message_id=? AND event_type='A2A_RESULT_VERIFIED'",
            (values[field], request.run_id, request.message_id),
        )


@pytest.mark.parametrize("field", TRACE_IDENTITY_FIELDS)
def test_duplicate_rejects_any_terminal_trace_identity_tamper(tmp_path: Path, field: str):
    order_id, phone_last4, _carrier, _tracking = _order_row()
    ledger_path = tmp_path / f"duplicate-trace-identity-{field}.sqlite"
    runtime = R4A2ARuntime(db_path=ORDER_DB, ledger_path=ledger_path)
    request = runtime.build_request(
        run_id=f"r-duplicate-trace-identity-{field}",
        plan_revision_id="p-duplicate-trace-identity",
        task_id="order",
        capability_ref="order/read@v1",
        payload={"order_id": order_id, "phone_last4": phone_last4},
    )
    assert runtime.dispatch(request).status == "SUCCEEDED"
    runtime.close()
    _tamper_terminal_trace(ledger_path, request, field)
    restarted = R4A2ARuntime(db_path=ORDER_DB, ledger_path=ledger_path)
    try:
        rejected = restarted.dispatch(request)
        assert rejected.status == "FAILED"
        assert rejected.duplicate is True
        assert rejected.canonical_result is None
        assert rejected.error_code.startswith("A2A_PERSISTED_TERMINAL_TRACE_")
    finally:
        restarted.close()


@pytest.mark.parametrize("field", TRACE_IDENTITY_FIELDS)
def test_freeze_rejects_any_terminal_trace_identity_tamper(tmp_path: Path, field: str):
    order_id, phone_last4, _carrier, _tracking = _order_row()
    ledger_path = tmp_path / f"freeze-trace-identity-{field}.sqlite"
    runtime = R4A2ARuntime(db_path=ORDER_DB, ledger_path=ledger_path)
    try:
        request = runtime.build_request(
            run_id=f"r-freeze-trace-identity-{field}",
            plan_revision_id="p-freeze-trace-identity",
            task_id="order",
            capability_ref="order/read@v1",
            payload={"order_id": order_id, "phone_last4": phone_last4},
        )
        assert runtime.dispatch(request).status == "SUCCEEDED"
        runtime.freeze(request.run_id)
        _tamper_terminal_trace(ledger_path, request, field)
        with pytest.raises(A2AContractError, match="A2A_FREEZE_CHECKSUM_MISMATCH"):
            runtime.freeze(request.run_id)
    finally:
        runtime.close()


@pytest.mark.parametrize("mutation", ["delete", "insert", "reorder"])
def test_trace_sequence_integrity_rejects_delete_insert_and_reorder(tmp_path: Path, mutation: str):
    order_id, phone_last4, _carrier, _tracking = _order_row()
    ledger_path = tmp_path / f"trace-sequence-{mutation}.sqlite"
    runtime = R4A2ARuntime(db_path=ORDER_DB, ledger_path=ledger_path)
    request = runtime.build_request(
        run_id=f"r-trace-sequence-{mutation}",
        plan_revision_id="p-trace-sequence",
        task_id="order",
        capability_ref="order/read@v1",
        payload={"order_id": order_id, "phone_last4": phone_last4},
    )
    assert runtime.dispatch(request).status == "SUCCEEDED"
    runtime.close()
    with sqlite3.connect(ledger_path) as conn:
        rows = conn.execute(
            "SELECT run_id,seq_no,event_id,trace_id,event_type,message_id,attempt_id,actor,scene_clock,payload_json,payload_hash FROM a2a_trace WHERE run_id=? ORDER BY seq_no",
            (request.run_id,),
        ).fetchall()
        assert len(rows) >= 2
        if mutation == "delete":
            conn.execute("DELETE FROM a2a_trace WHERE run_id=? AND seq_no=1", (request.run_id,))
        elif mutation == "insert":
            copied = rows[-1]
            conn.execute(
                "INSERT INTO a2a_trace(run_id,seq_no,event_id,trace_id,event_type,message_id,attempt_id,actor,scene_clock,payload_json,payload_hash) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (copied[0], 999, "forged-inserted-event", *copied[3:]),
            )
        else:
            conn.execute("UPDATE a2a_trace SET seq_no=-1 WHERE run_id=? AND seq_no=1", (request.run_id,))
            conn.execute("UPDATE a2a_trace SET seq_no=1 WHERE run_id=? AND seq_no=2", (request.run_id,))
            conn.execute("UPDATE a2a_trace SET seq_no=2 WHERE run_id=? AND seq_no=-1", (request.run_id,))
    restarted = R4A2ARuntime(db_path=ORDER_DB, ledger_path=ledger_path)
    try:
        rejected = restarted.dispatch(request)
        assert rejected.status == "FAILED"
        assert rejected.duplicate is True
        assert rejected.canonical_result is None
        assert rejected.error_code.startswith("A2A_PERSISTED_TERMINAL_TRACE_")
    finally:
        restarted.close()


@pytest.mark.parametrize("action", ["supersede", "cancel"])
def test_dispatched_lifecycle_change_allows_one_physical_call_without_retry(tmp_path: Path, action: str):
    import threading

    order_id, phone_last4, _carrier, _tracking = _order_row()
    ledger_path = tmp_path / f"dispatched-boundary-{action}.sqlite"
    runtime_one = R4A2ARuntime(
        db_path=ORDER_DB,
        ledger_path=ledger_path,
        failure_script={"order": {"kind": "PHYSICAL_FAILURE", "count": 1}},
    )
    runtime_two = R4A2ARuntime(db_path=ORDER_DB, ledger_path=ledger_path)
    started = threading.Event()
    release = threading.Event()
    original_adapter = runtime_one.adapters["order/read@v1"]

    class BlockingFailureAdapter:
        agent_ref = original_adapter.agent_ref
        capability_ref = original_adapter.capability_ref

        def invoke(self, payload, *, context):
            started.set()
            assert release.wait(timeout=10)
            return original_adapter.invoke(payload, context=context)

    runtime_one.adapters["order/read@v1"] = BlockingFailureAdapter()
    request = runtime_one.build_request(
        run_id=f"r-dispatched-boundary-{action}",
        plan_revision_id="p-dispatched-old",
        task_id="order",
        capability_ref="order/read@v1",
        payload={"order_id": order_id, "phone_last4": phone_last4},
    )
    outcome: dict[str, object] = {}

    def dispatch() -> None:
        try:
            outcome["result"] = runtime_one.dispatch(request)
        except Exception as exc:  # pragma: no cover - assertion reports it below
            outcome["error"] = exc

    thread = threading.Thread(target=dispatch)
    thread.start()
    try:
        assert started.wait(timeout=10)
        if action == "supersede":
            runtime_two.supersede_revision(request.run_id, "p-dispatched-old", "p-dispatched-new")
        else:
            runtime_two.cancel(request.run_id)
        release.set()
        thread.join(timeout=15)
        assert thread.is_alive() is False
        assert "error" not in outcome
        result = outcome["result"]
        assert result.status == "LATE"
        assert result.canonical_result is None
        assert runtime_one.physical_call_counts["order/read@v1"] == 1
        assert runtime_one.ledger.row(request.message_id)["status"] == "LATE"
        assert all(str(attempt["status"]) == "LATE" for attempt in runtime_one.ledger.attempts(request.message_id))
        assert not any(event.event_type == "A2A_BOUNDED_RETRY" for event in runtime_one.trace(request.run_id))
        assert len(runtime_two.ledger.late_events(request.run_id)) == 1
    finally:
        release.set()
        thread.join(timeout=15)
        runtime_one.close()
        runtime_two.close()


def test_live_trace_empty_projection_and_cross_runtime_append_are_stable(tmp_path: Path):
    order_id, phone_last4, carrier, tracking = _order_row()
    ledger_path = tmp_path / "live-trace-chain.sqlite"
    runtime_one = R4A2ARuntime(db_path=ORDER_DB, ledger_path=ledger_path)
    runtime_two = R4A2ARuntime(db_path=ORDER_DB, ledger_path=ledger_path)
    try:
        run_id = "r-live-trace-chain"
        runtime_one.start_run(run_id=run_id, plan_revision_id="p-live-trace-chain")
        empty = runtime_one.ledger.conn.execute(
            "SELECT live_event_count,live_head,live_checksum FROM a2a_runs WHERE run_id=?",
            (run_id,),
        ).fetchone()
        assert (empty[0], empty[1], empty[2]) == (0, "", sha256_json([]))

        order = runtime_one.build_request(
            run_id=run_id,
            plan_revision_id="p-live-trace-chain",
            task_id="order",
            capability_ref="order/read@v1",
            payload={"order_id": order_id, "phone_last4": phone_last4},
        )
        assert runtime_one.dispatch(order).status == "SUCCEEDED"
        second_order = runtime_two.build_request(
            run_id=run_id,
            plan_revision_id="p-live-trace-chain",
            task_id="order-two",
            capability_ref="order/read@v1",
            payload={"order_id": order_id, "phone_last4": phone_last4},
            idempotency_key="live-chain-second-order",
        )
        assert runtime_two.dispatch(second_order).status == "SUCCEEDED"
        assert runtime_two.dispatch(order).status == "SUCCEEDED"
        trace = runtime_one.trace(run_id)
        live = runtime_two.ledger.conn.execute(
            "SELECT live_event_count,live_head,live_checksum FROM a2a_runs WHERE run_id=?",
            (run_id,),
        ).fetchone()
        assert int(live[0]) == len(trace)
        assert live[1] == trace[-1].event_id
        assert live[2] == sha256_json([event.model_dump(mode="json") for event in trace])
        first_freeze = runtime_one.freeze(run_id)
        second_freeze = runtime_two.freeze(run_id)
        assert first_freeze["trace_checksum"] == second_freeze["trace_checksum"]
        assert first_freeze["trace"] == second_freeze["trace"]
    finally:
        runtime_one.close()
        runtime_two.close()


@pytest.mark.parametrize("outcome", ["SUCCEEDED", "FAILED"])
@pytest.mark.parametrize("mutation", ["delete_terminal", "move_terminal_run"])
def test_unfrozen_terminal_trace_delete_or_run_move_blocks_freeze(tmp_path: Path, outcome: str, mutation: str):
    order_id, phone_last4, _carrier, _tracking = _order_row()
    ledger_path = tmp_path / f"unfrozen-terminal-{outcome.lower()}-{mutation}.sqlite"
    failure_script = {"order": {"kind": "PHYSICAL_FAILURE", "count": 2}} if outcome == "FAILED" else None
    runtime = R4A2ARuntime(db_path=ORDER_DB, ledger_path=ledger_path, failure_script=failure_script)
    try:
        request = runtime.build_request(
            run_id=f"r-unfrozen-terminal-{outcome.lower()}-{mutation}",
            plan_revision_id="p-unfrozen-terminal",
            task_id="order",
            capability_ref="order/read@v1",
            payload={"order_id": order_id, "phone_last4": phone_last4},
        )
        assert runtime.dispatch(request).status == outcome
        event_type = "A2A_RESULT_VERIFIED" if outcome == "SUCCEEDED" else "A2A_MESSAGE_FAILED"
        if mutation == "delete_terminal":
            runtime.ledger.conn.execute(
                "DELETE FROM a2a_trace WHERE run_id=? AND message_id=? AND event_type=?",
                (request.run_id, request.message_id, event_type),
            )
        else:
            runtime.ledger.conn.execute(
                "UPDATE a2a_trace SET run_id=? WHERE run_id=? AND message_id=? AND event_type=?",
                ("forged-terminal-run", request.run_id, request.message_id, event_type),
            )
        with pytest.raises(A2AContractError, match="A2A_FREEZE_CHECKSUM_MISMATCH"):
            runtime.freeze(request.run_id)
    finally:
        runtime.close()


@pytest.mark.parametrize("field", ["live_event_count", "live_head", "live_checksum"])
def test_duplicate_and_freeze_reject_live_projection_tamper(tmp_path: Path, field: str):
    order_id, phone_last4, _carrier, _tracking = _order_row()
    ledger_path = tmp_path / f"live-projection-tamper-{field}.sqlite"
    runtime = R4A2ARuntime(db_path=ORDER_DB, ledger_path=ledger_path)
    request = runtime.build_request(
        run_id=f"r-live-projection-tamper-{field}",
        plan_revision_id="p-live-projection-tamper",
        task_id="order",
        capability_ref="order/read@v1",
        payload={"order_id": order_id, "phone_last4": phone_last4},
    )
    assert runtime.dispatch(request).status == "SUCCEEDED"
    runtime.close()
    with sqlite3.connect(ledger_path) as conn:
        value = {"live_event_count": 999999, "live_head": "forged-head", "live_checksum": "0" * 64}[field]
        conn.execute(f"UPDATE a2a_runs SET {field}=? WHERE run_id=?", (value, request.run_id))
    restarted = R4A2ARuntime(db_path=ORDER_DB, ledger_path=ledger_path)
    try:
        rejected = restarted.dispatch(request)
        assert rejected.status == "FAILED"
        assert rejected.error_code.startswith("A2A_PERSISTED_TERMINAL_TRACE_LIVE_CHAIN_")
        with pytest.raises(A2AContractError, match="A2A_FREEZE_CHECKSUM_MISMATCH"):
            restarted.freeze(request.run_id)
    finally:
        restarted.close()


@pytest.mark.parametrize("outcome", ["SUCCEEDED", "FAILED"])
@pytest.mark.parametrize("operation", ["duplicate", "freeze"])
def test_terminal_trace_cardinality_is_exactly_one(tmp_path: Path, outcome: str, operation: str):
    order_id, phone_last4, _carrier, _tracking = _order_row()
    ledger_path = tmp_path / f"terminal-cardinality-{outcome.lower()}-{operation}.sqlite"
    failure_script = {"order": {"kind": "PHYSICAL_FAILURE", "count": 2}} if outcome == "FAILED" else None
    runtime = R4A2ARuntime(db_path=ORDER_DB, ledger_path=ledger_path, failure_script=failure_script)
    try:
        request = runtime.build_request(
            run_id=f"r-terminal-cardinality-{outcome.lower()}-{operation}",
            plan_revision_id="p-terminal-cardinality",
            task_id="order",
            capability_ref="order/read@v1",
            payload={"order_id": order_id, "phone_last4": phone_last4},
        )
        result = runtime.dispatch(request)
        assert result.status == outcome
        assert result.canonical_result is not None
        attempts = runtime.ledger.attempts(request.message_id)
        terminal_type = "A2A_RESULT_VERIFIED" if outcome == "SUCCEEDED" else "A2A_MESSAGE_FAILED"
        terminal_payload = {
            "attempt_no": int(attempts[-1]["attempt_no"]),
            "result_id": result.canonical_result.result_id,
            "payload_hash": result.specialist_result.payload_hash if outcome == "SUCCEEDED" and result.specialist_result is not None else None,
            "error_code": result.error_code if outcome == "FAILED" else None,
        }
        terminal_payload = {key: value for key, value in terminal_payload.items() if value is not None}
        runtime.ledger.append_trace(
            run_id=request.run_id,
            trace_id=request.trace_id,
            event_type=terminal_type,
            scene_clock=runtime.now(),
            payload=terminal_payload,
            message_id=request.message_id,
            attempt_id=result.canonical_result.attempt_id,
        )
        if operation == "duplicate":
            rejected = runtime.dispatch(request)
            assert rejected.status == "FAILED"
            assert rejected.duplicate is True
            assert rejected.error_code == "A2A_PERSISTED_TERMINAL_TRACE_MISSING"
        else:
            with pytest.raises(A2AContractError, match="A2A_TERMINAL_TRACE_INTEGRITY_MISMATCH"):
                runtime.freeze(request.run_id)
    finally:
        runtime.close()
