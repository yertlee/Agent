"""M3 review recovery facade over the M2 AfterSalesService transition API."""
from __future__ import annotations

from typing import Any

from agent.domain.aftersales import AfterSalesService
from agent.m2_errors import ErrorCatalog


class ReviewRecovery:
    decisions = {"APPROVE", "REJECT", "EXPIRE", "CANCEL"}

    def __init__(self, repository):
        self.repository = repository

    def decide(self, *, ticket_id: str, actor: str, decision: str, run_id: str, task_id: str, session_id: str, reason: str, expected_plan_revision_id: str | None = None, expected_state_version: int | None = None) -> dict[str, Any]:
        if actor != "reviewer" or decision not in self.decisions:
            raise ValueError(ErrorCatalog.envelope("STATE_TRANSITION_REJECTED", details={"actor": actor, "decision": decision}).code)
        ticket = self.repository.conn.execute("SELECT run_id,task_id FROM review_tickets WHERE ticket_id=?", (ticket_id,)).fetchone()
        if not ticket or str(ticket[0]) != run_id or str(ticket[1]) != task_id:
            raise ValueError(ErrorCatalog.envelope("CONTRACT_INVALID_STATE", details={"ticket_id": ticket_id}).code)
        ownership = self.repository.conn.execute("SELECT t.plan_revision_id,p.status FROM tasks t JOIN plan_revisions p ON p.plan_revision_id=t.plan_revision_id WHERE t.task_id=? AND t.run_id=?", (task_id, run_id)).fetchone()
        if not ownership or str(ownership[1]) != "ACTIVE" or (expected_plan_revision_id is not None and str(ownership[0]) != expected_plan_revision_id):
            raise ValueError(ErrorCatalog.envelope("CONTRACT_INVALID_STATE", details={"ticket_id": ticket_id, "reason": "inactive_or_mismatched_plan"}).code)
        case = self.repository.conn.execute("SELECT session_id,state_version FROM aftersales_cases WHERE run_id=? AND task_id=? AND session_id=? AND status='HUMAN_REVIEW' ORDER BY created_at DESC LIMIT 1", (run_id, task_id, session_id)).fetchone()
        if not case:
            raise ValueError(ErrorCatalog.envelope("CONTRACT_INVALID_STATE", details={"ticket_id": ticket_id}).code)
        if expected_state_version is not None and int(case[1]) != int(expected_state_version):
            raise ValueError(ErrorCatalog.envelope("CONTRACT_INVALID_STATE", details={"ticket_id": ticket_id, "reason": "state_version_conflict"}).code)
        result = AfterSalesService(self.repository).decide_review_ticket(ticket_id=ticket_id, actor=actor, decision=decision, reason=reason)
        result["decision"] = decision
        result["session_id"] = session_id
        return result


__all__ = ["ReviewRecovery"]
