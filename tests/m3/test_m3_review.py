from decimal import Decimal
import pytest

from agent.domain.aftersales import AfterSalesService
from agent.domain.confirm_tokens import ConfirmTokenManager
from agent.m3_review import ReviewRecovery

from tests.m2.test_tokens_transitions import _eligibility
from tests.m2.helpers import order_fact, setup_repo


def test_review_recovery_supports_all_terminal_decisions(tmp_path):
    for index, decision in enumerate(("APPROVE", "REJECT", "EXPIRE", "CANCEL")):
        repo, task = setup_repo(tmp_path / str(index))
        try:
            order = order_fact()
            _, raw = ConfirmTokenManager(repo).issue(session_id="s1", user_id="u1", run_id="r1", task_id=task.task_id, order_id="o1", service="refund", amount=Decimal("10.00"), payload_hash=order.snapshot_hash, topic_version="v1")
            AfterSalesService(repo).create_case(session_id="s1", user_id="u1", run_id="r1", task_id=task.task_id, attempt_id="a1", order=order, eligibility=_eligibility(order), service="refund", reason="review", amount=Decimal("10.00"), raw_token=raw, idempotency_key=f"review-{index}", review_required=True)
            ticket = repo.conn.execute("select ticket_id from review_tickets").fetchone()[0]
            result = ReviewRecovery(repo).decide(ticket_id=ticket, actor="reviewer", decision=decision, run_id="r1", task_id=task.task_id, session_id="s1", reason="reviewed")
            assert result["decision"] == decision
            assert repo.conn.execute("select status from review_tickets").fetchone()[0] == {"APPROVE": "APPROVED", "REJECT": "REJECTED", "EXPIRE": "EXPIRED", "CANCEL": "CANCELLED"}[decision]
        finally:
            repo.close()


def test_review_recovery_validates_revision_version_ownership_and_is_single_use(tmp_path):
    repo, task = setup_repo(tmp_path / "ownership")
    try:
        order = order_fact()
        _, raw = ConfirmTokenManager(repo).issue(session_id="s1", user_id="u1", run_id="r1", task_id=task.task_id, order_id="o1", service="refund", amount=Decimal("10.00"), payload_hash=order.snapshot_hash, topic_version="v1")
        AfterSalesService(repo).create_case(session_id="s1", user_id="u1", run_id="r1", task_id=task.task_id, attempt_id="a1", order=order, eligibility=_eligibility(order), service="refund", reason="review", amount=Decimal("10.00"), raw_token=raw, idempotency_key="review-ownership", review_required=True)
        ticket = repo.conn.execute("select ticket_id from review_tickets").fetchone()[0]
        recovery = ReviewRecovery(repo)
        for kwargs in (
            {"session_id": "other"},
            {"expected_plan_revision_id": "wrong"},
            {"expected_state_version": 7},
        ):
            with pytest.raises(ValueError, match="CONTRACT_INVALID_STATE"):
                recovery.decide(ticket_id=ticket, actor="reviewer", decision="APPROVE", run_id="r1", task_id=task.task_id, session_id=kwargs.pop("session_id", "s1"), reason="checked", **kwargs)
        result = recovery.decide(ticket_id=ticket, actor="reviewer", decision="APPROVE", run_id="r1", task_id=task.task_id, session_id="s1", reason="checked", expected_plan_revision_id="p1", expected_state_version=0)
        assert result["decision"] == "APPROVE"
        with pytest.raises(ValueError, match="CONTRACT_INVALID_STATE"):
            recovery.decide(ticket_id=ticket, actor="reviewer", decision="APPROVE", run_id="r1", task_id=task.task_id, session_id="s1", reason="duplicate", expected_plan_revision_id="p1", expected_state_version=0)
        assert repo.conn.execute("select count(*) from m2_audit_log where run_id='r1' and actor='reviewer'").fetchone()[0] == 1
        assert repo.conn.execute("select count(*) from trace_outbox where run_id='r1' and envelope_json like '%REVIEW_DECISION%'").fetchone()[0] == 1
    finally:
        repo.close()
