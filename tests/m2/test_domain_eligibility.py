from datetime import datetime, timedelta, timezone

from agent.domain.eligibility import EligibilityEngine
from agent.domain.facts import DataQuality
from agent.domain.policy import PolicyCatalog, PolicyRule

from .helpers import order_fact


def catalog():
    return PolicyCatalog([PolicyRule(rule_id="refund_v1", decision_logic="allow eligible paid orders", source="policy_catalog", effective_from=datetime(2025, 1, 1, tzinfo=timezone.utc), scope="refund", priority=1, version="policy.v1")], version="policy.v1")


def test_order_fact_and_allow_eligibility_are_hashed_and_versioned():
    order = order_fact()
    eligibility = EligibilityEngine(catalog()).check(order, service="refund", now=datetime(2026, 1, 1, tzinfo=timezone.utc))
    assert order.snapshot_hash and len(order.snapshot_hash) == 64
    assert eligibility.decision == "ALLOW"
    assert eligibility.policy_version == "policy.v1"


def test_bad_quality_and_unknown_service_fail_safe():
    engine = EligibilityEngine(catalog())
    stale = engine.check(order_fact(quality=DataQuality.STALE), service="refund")
    unknown = engine.check(order_fact(), service="gift")
    assert stale.decision == "MANUAL" and stale.rule_id == "DATA_STALE"
    assert unknown.decision == "DENY"


def test_conflicting_active_policy_is_manual_not_silently_sorted():
    catalog = PolicyCatalog([
        PolicyRule(rule_id="refund_allow", decision_logic="allow", source="a", effective_from=datetime(2025, 1, 1, tzinfo=timezone.utc), scope="refund", priority=5, version="policy.v2"),
        PolicyRule(rule_id="refund_deny", decision_logic="deny", source="b", effective_from=datetime(2025, 1, 1, tzinfo=timezone.utc), scope="refund", priority=5, version="policy.v2"),
    ], version="policy.v2")
    fact = EligibilityEngine(catalog).check(order_fact(), service="refund", now=datetime(2026, 1, 1, tzinfo=timezone.utc))
    assert fact.decision == "MANUAL" and fact.rule_id == "POLICY_CONFLICT"
