"""M2 after-sales case transaction and single actor transition API (02 §3-5)."""
from __future__ import annotations

from decimal import Decimal
from typing import Any, Optional
from uuid import uuid4

from agent.domain.objects import Result, ResultStatus, sha256_json
from agent.m2_errors import ErrorCatalog
from agent.trace.events import build_event
from .confirm_tokens import ConfirmTokenManager, TokenError
from .eligibility import EligibilityEngine
from .facts import EligibilityFact, OrderFact


ACTIVE_STATUSES = {"REQUESTED", "UNDER_REVIEW", "APPROVED", "RETURN_PENDING", "RETURNED", "REFUND_PENDING", "EXCHANGE_PENDING", "HUMAN_REVIEW"}
TERMINAL_STATUSES = {"REFUNDED", "EXCHANGED", "REJECTED", "CANCELLED"}
TRANSITIONS = {
    ("REQUESTED", "UNDER_REVIEW"): {"aftersales_service"},
    ("REQUESTED", "APPROVED"): {"aftersales_service"},
    ("UNDER_REVIEW", "APPROVED"): {"reviewer"},
    ("UNDER_REVIEW", "REJECTED"): {"reviewer"},
    ("APPROVED", "RETURN_PENDING"): {"aftersales_service"},
    ("RETURN_PENDING", "RETURNED"): {"warehouse", "logistics"},
    ("APPROVED", "REFUND_PENDING"): {"aftersales_service"},
    ("REFUND_PENDING", "REFUNDED"): {"funds_system"},
    ("APPROVED", "EXCHANGE_PENDING"): {"aftersales_service"},
    ("EXCHANGE_PENDING", "EXCHANGED"): {"warehouse", "fulfillment"},
}


class AfterSalesError(RuntimeError):
    def __init__(self, envelope):
        self.envelope = envelope
        super().__init__(envelope.code)


class AfterSalesService:
    def __init__(self, repository):
        self.repository = repository
        self.tokens = ConfirmTokenManager(repository)

    def create_case(
        self,
        *,
        session_id: str,
        user_id: str,
        run_id: str,
        task_id: str,
        attempt_id: str,
        order: OrderFact,
        eligibility: EligibilityFact,
        service: str,
        reason: str,
        amount: Decimal,
        raw_token: str,
        idempotency_key: str,
        review_required: bool = False,
    ) -> dict[str, Any]:
        if order.user_id != user_id:
            raise AfterSalesError(ErrorCatalog.envelope("AUTH_RESOURCE_FORBIDDEN"))
        if eligibility.decision != "ALLOW":
            code = "ELIGIBILITY_MANUAL" if eligibility.decision == "MANUAL" else "ELIGIBILITY_DENIED"
            raise AfterSalesError(ErrorCatalog.envelope(code, details={"rule_id": eligibility.rule_id}))
        fingerprint = sha256_json({"user_id": user_id, "order_id": order.entity_id, "service": service, "reason": reason, "amount": str(amount)})
        tool_ref = "aftersales/create@v1"
        try:
            self.repository.conn.execute("BEGIN IMMEDIATE")
            replay = self.repository.conn.execute("SELECT result_id,idempotency_key FROM tool_submit_log WHERE user_id=? AND tool_ref=? AND request_fingerprint=?", (user_id, tool_ref, fingerprint)).fetchone()
            if replay:
                result_row = self.repository.conn.execute("SELECT payload_json FROM results WHERE result_id=?", (replay[0],)).fetchone()
                action = "IDEMPOTENCY_REPLAY" if str(replay[1]) == idempotency_key else "IDEMPOTENCY_CONFLICT"
                self.repository.log_audit_locked(run_id=run_id, actor="runtime", action=action, reason="same business request replay", trace_id=None)
                self.repository.conn.commit()
                return {"replayed": True, "result_id": replay[0], "case_id": __import__("json").loads(result_row[0])["case_id"] if result_row and result_row[0] else None}
            key_collision = self.repository.conn.execute("SELECT request_fingerprint FROM tool_submit_log WHERE idempotency_key=?", (idempotency_key,)).fetchone()
            if key_collision and str(key_collision[0]) != fingerprint:
                raise AfterSalesError(ErrorCatalog.envelope("IDEMPOTENCY_CONFLICT"))
            active = self.repository.conn.execute("SELECT case_id FROM aftersales_cases WHERE user_id=? AND order_id=? AND status IN ({})".format(",".join("?" for _ in ACTIVE_STATUSES)), (user_id, order.entity_id, *sorted(ACTIVE_STATUSES))).fetchone()
            if active:
                raise AfterSalesError(ErrorCatalog.envelope("ACTIVE_CASE_EXISTS", details={"case_id": active[0]}))
            if not self.repository.conn.execute("SELECT 1 FROM eligibility_facts WHERE fact_id=?", (eligibility.fact_id,)).fetchone():
                self.repository.conn.execute("INSERT INTO eligibility_facts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", (eligibility.fact_id, run_id, task_id, eligibility.entity_id, eligibility.service, eligibility.decision, eligibility.rule_id, eligibility.reason, eligibility.policy_version, eligibility.snapshot_hash, __import__("agent.domain.objects", fromlist=["canonical_json"]).canonical_json(eligibility), __import__("agent.storage.repositories", fromlist=["_now"])._now(), __import__("agent.storage.repositories", fromlist=["_now"])._now()))
            self.tokens._consume_locked(raw_token, session_id=session_id, user_id=user_id, run_id=run_id, task_id=task_id, order_id=order.entity_id, service=service, amount=amount, payload_hash=order.snapshot_hash, topic_version="v1")
            case_id = f"case_{uuid4().hex}"
            initial_status = "HUMAN_REVIEW" if review_required else "REQUESTED"
            self.repository.conn.execute("INSERT INTO aftersales_cases VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (case_id, user_id, order.entity_id, service, initial_status, 0, run_id, session_id, task_id, idempotency_key, fingerprint, reason, str(amount), __import__("agent.storage.repositories", fromlist=["_now"])._now(), __import__("agent.storage.repositories", fromlist=["_now"])._now()))
            if review_required:
                self.repository.conn.execute(
                    "INSERT INTO review_tickets VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (f"review_{uuid4().hex}", run_id, task_id, idempotency_key, "HIGH_RISK", reason, sha256_json({"case_id": case_id, "fingerprint": fingerprint}), "OPEN", __import__("agent.storage.repositories", fromlist=["_now"])._now(), None, None, None, __import__("agent.storage.repositories", fromlist=["_now"])._now(), __import__("agent.storage.repositories", fromlist=["_now"])._now()),
                )
            parent = self.repository.conn.execute("SELECT event_id FROM trace_outbox WHERE run_id=? ORDER BY seq_no DESC LIMIT 1", (run_id,)).fetchone()
            parent_id = str(parent[0]) if parent else None
            self._event_locked(run_id=run_id, session_id=session_id, task_id=task_id, event_type="TOOL_CALLED", payload={"tool_ref": tool_ref, "capability_ref": "aftersales/write@v1", "safe_args_hash": sha256_json({"order_id": order.entity_id, "service": service, "reason_hash": sha256_json(reason)}), "implementation_mode": "SIMULATED"}, parent_event_id=parent_id)
            result_id = f"result_{uuid4().hex}"
            result = Result(result_id=result_id, run_id=run_id, plan_revision_id=str(self.repository.conn.execute("SELECT plan_revision_id FROM tasks WHERE task_id=?", (task_id,)).fetchone()[0]), task_id=task_id, attempt_id=attempt_id, status=ResultStatus.SUCCEEDED, output_contract="aftersales.result.v1", payload={"case_id": case_id, "status": initial_status}, business_code="OK")
            parent = self.repository.conn.execute("SELECT event_id FROM trace_outbox WHERE run_id=? ORDER BY seq_no DESC LIMIT 1", (run_id,)).fetchone()
            self._event_locked(run_id=run_id, session_id=session_id, task_id=task_id, event_type="TOOL_RETURNED", payload={"result_id": result_id, "ok": True, "error_code": None, "output_hash": result.payload_hash}, parent_event_id=str(parent[0]) if parent else None)
            self.repository._append_result_locked(result)
            self.repository.conn.execute("UPDATE task_attempts SET status='SUCCEEDED',ended_at=?,updated_at=? WHERE attempt_id=? AND status IN ('CREATED','RUNNING')", (__import__("agent.storage.repositories", fromlist=["_now"])._now(), __import__("agent.storage.repositories", fromlist=["_now"])._now(), attempt_id))
            parent = self.repository.conn.execute("SELECT event_id FROM trace_outbox WHERE run_id=? ORDER BY seq_no DESC LIMIT 1", (run_id,)).fetchone()
            result_event = self._event_locked(run_id=run_id, session_id=session_id, plan_revision_id=result.plan_revision_id, task_id=task_id, attempt_id=attempt_id, event_type="RESULT_WRITTEN", payload={"result_id": result_id, "status": "SUCCEEDED", "payload_hash": result.payload_hash}, parent_event_id=str(parent[0]) if parent else None)
            self.repository.conn.execute("INSERT INTO tool_submit_log VALUES (?,?,?,?,?,?,?,?,?,?,?)", (f"submit_{uuid4().hex}", user_id, run_id, task_id, tool_ref, fingerprint, idempotency_key, result_id, "REVIEW_REQUIRED" if review_required else "COMMITTED", __import__("agent.storage.repositories", fromlist=["_now"])._now(), __import__("agent.storage.repositories", fromlist=["_now"])._now()))
            self.repository.log_audit_locked(run_id=run_id, actor="aftersales_service", action="CASE_CREATED", reason="eligible confirmed request", trace_id=result_event.trace_id, after_hash=sha256_json({"case_id": case_id, "status": "REQUESTED"}))
            self.repository.conn.commit()
            return {"replayed": False, "case_id": case_id, "result": result, "result_id": result_id}
        except Exception:
            self.repository.conn.rollback()
            raise

    def decide_review_ticket(self, *, ticket_id: str, actor: str, decision: str, reason: str) -> dict[str, Any]:
        """The only API that closes a HIGH_RISK review and advances its case."""
        if actor != "reviewer" or decision not in {"APPROVE", "REJECT"}:
            raise AfterSalesError(ErrorCatalog.envelope("STATE_TRANSITION_REJECTED", details={"actor": actor, "decision": decision}))
        try:
            self.repository.conn.execute("BEGIN IMMEDIATE")
            ticket = self.repository.conn.execute("SELECT ticket_id,run_id,task_id,status FROM review_tickets WHERE ticket_id=?", (ticket_id,)).fetchone()
            if not ticket or ticket[3] != "OPEN":
                raise AfterSalesError(ErrorCatalog.envelope("CONTRACT_INVALID_STATE", details={"ticket_id": ticket_id}))
            target = "APPROVED" if decision == "APPROVE" else "REJECTED"
            self.repository.conn.execute("UPDATE review_tickets SET status=?,decision=?,decision_by=?,decision_at=?,updated_at=? WHERE ticket_id=? AND status='OPEN'", ("APPROVED" if decision == "APPROVE" else "REJECTED", decision, actor, __import__("agent.storage.repositories", fromlist=["_now"])._now(), __import__("agent.storage.repositories", fromlist=["_now"])._now(), ticket_id))
            case = self.repository.conn.execute("SELECT case_id,status,state_version,session_id FROM aftersales_cases WHERE task_id=? AND run_id=? AND status='HUMAN_REVIEW' ORDER BY created_at DESC LIMIT 1", (ticket[2], ticket[1])).fetchone()
            if not case:
                raise AfterSalesError(ErrorCatalog.envelope("CONTRACT_INVALID_STATE", details={"ticket_id": ticket_id}))
            new_version = int(case[2]) + 1
            self.repository.conn.execute("UPDATE aftersales_cases SET status=?,state_version=?,updated_at=? WHERE case_id=? AND status='HUMAN_REVIEW'", (target, new_version, __import__("agent.storage.repositories", fromlist=["_now"])._now(), case[0]))
            event = self._event_locked(run_id=ticket[1], session_id=case[3], task_id=ticket[2], event_type="TASK_STATE_CHANGED", payload={"task_id": ticket[2], "actor": actor, "from": "HUMAN_REVIEW", "to": target, "reason": reason, "ticket_id": ticket_id, "state_version": new_version}, parent_event_id=self._last_event(ticket[1]))
            self.repository.log_audit_locked(run_id=ticket[1], actor=actor, action="REVIEW_DECISION", reason=reason, trace_id=event.trace_id)
            self.repository.conn.commit()
            return {"ticket_id": ticket_id, "case_id": case[0], "status": target, "state_version": new_version}
        except Exception:
            self.repository.conn.rollback()
            raise

    def transition_after_sales(self, *, case_id: str, actor: str, expected_status: str, target_status: str, reason: str, idempotency_key: str) -> dict[str, Any]:
        try:
            self.repository.conn.execute("BEGIN IMMEDIATE")
            row = self.repository.conn.execute("SELECT case_id,user_id,order_id,service,status,state_version,run_id,session_id,task_id,idempotency_key FROM aftersales_cases WHERE case_id=?", (case_id,)).fetchone()
            if row and str(row[9]) == idempotency_key and str(row[4]) == target_status:
                self.repository.conn.rollback()
                return {"case_id": case_id, "status": row[4], "state_version": row[5], "snapshot_hash": sha256_json({"case_id": case_id, "status": row[4], "version": row[5]})}
            if not row or str(row[4]) != expected_status:
                raise AfterSalesError(ErrorCatalog.envelope("CONTRACT_INVALID_STATE", details={"case_id": case_id}))
            if target_status not in ACTIVE_STATUSES | TERMINAL_STATUSES or (expected_status, target_status) not in TRANSITIONS and not (target_status == "CANCELLED" and expected_status in ACTIVE_STATUSES and actor in {"aftersales_service", "reviewer"}) and not (target_status == "HUMAN_REVIEW" and expected_status in ACTIVE_STATUSES and actor == "aftersales_service"):
                raise AfterSalesError(ErrorCatalog.envelope("STATE_TRANSITION_REJECTED", details={"from": expected_status, "to": target_status, "actor": actor}))
            if actor not in TRANSITIONS.get((expected_status, target_status), {actor}):
                raise AfterSalesError(ErrorCatalog.envelope("STATE_TRANSITION_REJECTED", details={"from": expected_status, "to": target_status, "actor": actor}))
            new_version = int(row[5]) + 1
            snapshot_hash = sha256_json({"case_id": case_id, "from": expected_status, "to": target_status, "version": new_version})
            self.repository.conn.execute("UPDATE aftersales_cases SET status=?,state_version=?,idempotency_key=?,updated_at=? WHERE case_id=? AND status=? AND state_version=?", (target_status, new_version, idempotency_key, __import__("agent.storage.repositories", fromlist=["_now"])._now(), case_id, expected_status, row[5]))
            transition_event = self._event_locked(run_id=row[6], session_id=row[7], task_id=row[8], event_type="TASK_STATE_CHANGED", payload={"task_id": row[8], "actor": actor, "from": expected_status, "to": target_status, "reason": reason, "state_version": new_version, "snapshot_hash": snapshot_hash}, parent_event_id=self._last_event(row[6]))
            self.repository.log_audit_locked(run_id=row[6], actor=actor, action="AFTERSALES_TRANSITION", reason=reason, trace_id=transition_event.trace_id, after_hash=snapshot_hash)
            self.repository.conn.commit()
            return {"case_id": case_id, "status": target_status, "state_version": new_version, "snapshot_hash": snapshot_hash}
        except Exception:
            self.repository.conn.rollback()
            raise

    def _last_event(self, run_id: str) -> Optional[str]:
        row = self.repository.conn.execute("SELECT event_id FROM trace_outbox WHERE run_id=? ORDER BY seq_no DESC LIMIT 1", (run_id,)).fetchone()
        return str(row[0]) if row else None

    def _event_locked(self, *, run_id: str, session_id: str, event_type: str, payload: dict[str, Any], parent_event_id: Optional[str], plan_revision_id: Optional[str] = None, task_id: Optional[str] = None, attempt_id: Optional[str] = None):
        row = self.repository.conn.execute("SELECT next_seq_no FROM runs WHERE run_id=?", (run_id,)).fetchone()
        event = build_event(run_id=run_id, session_id=session_id, event_type=event_type, seq_no=int(row[0]), plan_revision_id=plan_revision_id, task_id=task_id, attempt_id=attempt_id, parent_event_id=parent_event_id, payload=payload)
        self.repository._append_m2_event_locked(event)
        return event


__all__ = ["ACTIVE_STATUSES", "AfterSalesError", "AfterSalesService", "TERMINAL_STATUSES", "TRANSITIONS"]
