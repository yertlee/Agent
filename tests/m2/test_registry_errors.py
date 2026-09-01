from datetime import datetime, timedelta, timezone

import pytest

from agent.m2_context import InvocationContext
from agent.m2_errors import ErrorCatalog, ErrorEnvelope
from agent.m2_registry import Registry, ToolSpec, build_m2_registry


def test_registry_has_canonical_refs_and_manifest():
    registry = build_m2_registry()
    assert len(registry.refs()) == 6
    assert all("/" in ref and "@" in ref for ref in registry.refs())
    assert all("callable" not in item for item in registry.as_manifest())
    for spec in registry._specs.values():
        examples = {
            "order.read.v1": {"order_id": "o", "phone_last4": "1234"},
            "aftersales.query.v1": {"order_id": "o", "phone_last4": "1234"},
            "aftersales.create.v1": {"order_id": "o", "phone_last4": "1234", "service_type": "refund", "reason": "damaged"},
            "logistics.query.v1": {"carrier_code": "carrier", "tracking_no": "tracking"},
            "handoff.v1": {"summary": "summary", "reason": "reason"},
            "policy.query.v1": {"query": "refund"},
        }
        spec.validate_candidate_args(examples[spec.args_schema])
        with pytest.raises(ValueError):
            spec.validate_candidate_args({})


def test_registry_duplicate_and_unknown_error_fail_closed():
    spec = ToolSpec(tool_ref="x/read@v1", capability_ref="x/read@v1", owner="x", args_schema="x.v1", result_schema="x.v1", risk="READ", implementation_mode="SIMULATED", side_effect="READ_ONLY", timeout_ms=10, allowed_error_codes=("ORDER_NOT_FOUND",), callable=lambda: {"success": True})
    with pytest.raises(ValueError):
        ToolSpec(tool_ref="x/write@v1", capability_ref="x/write@v1", owner="x", args_schema="x.v1", result_schema="x.v1", risk="READ", implementation_mode="SIMULATED", side_effect="READ_ONLY", timeout_ms=10, allowed_error_codes=("NOT_CANONICAL",), callable=lambda: {})
    with pytest.raises(ValueError):
        Registry([spec, spec])


def test_error_envelope_maps_legacy_without_accepting_unknown():
    envelope = ErrorCatalog.envelope("PHONE_MISMATCH")
    assert envelope.code == "AUTH_IDENTITY_MISMATCH"
    with pytest.raises(ValueError):
        ErrorEnvelope(code="NOPE", layer="business", action="reject", message_key="x", retryable=False)


def test_invocation_context_is_frozen_and_has_trusted_fields():
    ctx = InvocationContext(session_id="s", user_id="u", run_id="r", plan_revision_id="p", task_id="t", attempt_id="a", agent_ref="order@v1", auth_scope="u:order", idempotency_key="i", deadline=datetime.now(timezone.utc) + timedelta(seconds=1), config_version="c", registry_version="r", dataset_version="d", trace_id="trace")
    with pytest.raises((TypeError, ValueError)):
        ctx.user_id = "spoof"
