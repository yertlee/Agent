"""Deterministic Eligibility Engine for M2."""
from __future__ import annotations

from datetime import datetime, timezone

from .facts import DataQuality, EligibilityFact, OrderFact
from .policy import PolicyCatalog, PolicyConflictError
from .objects import sha256_json


class EligibilityEngine:
    def __init__(self, catalog: PolicyCatalog):
        self.catalog = catalog

    def check(self, order: OrderFact, *, service: str, now: datetime | None = None) -> EligibilityFact:
        now = now or datetime.now(timezone.utc)
        if order.data_quality == DataQuality.CONFLICT:
            decision, rule_id, reason = "MANUAL", "DATA_CONFLICT", "order fact conflict requires human review"
        elif order.data_quality in {DataQuality.MISSING, DataQuality.STALE}:
            decision, rule_id, reason = "MANUAL", f"DATA_{order.data_quality.value}", "order fact quality is insufficient"
        elif service not in {"refund", "return", "exchange"}:
            decision, rule_id, reason = "DENY", "SERVICE_UNSUPPORTED", "service is outside catalog scope"
        elif str(order.status).upper() in {"CANCELLED", "REFUNDED", "EXCHANGED"}:
            decision, rule_id, reason = "DENY", "ORDER_TERMINAL", "order is already terminal"
        else:
            try:
                rule = self.catalog.resolve(service=service, when=now)
                decision, rule_id, reason = "ALLOW", rule.rule_id, rule.decision_logic
            except PolicyConflictError:
                decision, rule_id, reason = "MANUAL", "POLICY_CONFLICT", "multiple active policy rules conflict"
            except LookupError:
                decision, rule_id, reason = "MANUAL", "POLICY_MISSING", "no active policy rule"
        return EligibilityFact(
            fact_id=f"eligibility_{sha256_json({'order': order.entity_id, 'service': service, 'catalog': self.catalog.version})[:20]}",
            entity_id=f"{order.entity_id}:{service}",
            source="policy_catalog",
            version=self.catalog.version,
            data_quality=order.data_quality,
            service=service,
            decision=decision,
            rule_id=rule_id,
            reason=reason,
            policy_version=self.catalog.version,
            observed_at=now,
        )


__all__ = ["EligibilityEngine"]
