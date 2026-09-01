from decimal import Decimal

import pytest

from agent.domain.aftersales import AfterSalesError, AfterSalesService
from agent.domain.confirm_tokens import ConfirmTokenManager, TokenError
from agent.domain.eligibility import EligibilityEngine
from agent.domain.policy import PolicyCatalog, PolicyRule
from agent.domain.objects import sha256_json

from .helpers import order_fact, setup_repo


def _eligibility(order):
    catalog = PolicyCatalog([PolicyRule(rule_id="refund_v1", decision_logic="allow", source="policy", effective_from=__import__("datetime").datetime(2025, 1, 1, tzinfo=__import__("datetime").timezone.utc), scope="refund", version="v1")], version="v1")
    return EligibilityEngine(catalog).check(order, service="refund")


def test_token_is_hash_only_binding_and_one_time(tmp_path):
    repo, task = setup_repo(tmp_path)
    try:
        payload_hash = sha256_json({"order": "o1"})
        manager = ConfirmTokenManager(repo)
        token, raw = manager.issue(session_id="s1", user_id="u1", run_id="r1", task_id=task.task_id, order_id="o1", service="refund", amount=Decimal("10.00"), payload_hash=payload_hash, topic_version="v1")
        assert raw not in str(repo.conn.execute("select token_hash from confirm_tokens").fetchone()[0])
        consumed = manager.consume(raw, session_id="s1", user_id="u1", run_id="r1", task_id=task.task_id, order_id="o1", service="refund", amount=Decimal("10.00"), payload_hash=payload_hash, topic_version="v1")
        assert consumed.status == "CONSUMED"
        with pytest.raises(TokenError) as exc:
            manager.consume(raw, session_id="s1", user_id="u1", run_id="r1", task_id=task.task_id, order_id="o1", service="refund", amount=Decimal("10.00"), payload_hash=payload_hash, topic_version="v1")
        assert exc.value.envelope.code == "CONTRACT_CONFIRM_EXPIRED"
        assert repo.conn.execute("select count(*) from trace_outbox where run_id='r1' and envelope_json like '%TOKEN_EVENT%'").fetchone()[0] == 2
    finally:
        repo.close()


def test_aftersales_create_and_transition_use_atomic_result_and_outbox(tmp_path):
    repo, task = setup_repo(tmp_path)
    try:
        order = order_fact()
        eligibility = _eligibility(order)
        token, raw = ConfirmTokenManager(repo).issue(session_id="s1", user_id="u1", run_id="r1", task_id=task.task_id, order_id="o1", service="refund", amount=Decimal("10.00"), payload_hash=order.snapshot_hash, topic_version="v1")
        result = AfterSalesService(repo).create_case(session_id="s1", user_id="u1", run_id="r1", task_id=task.task_id, attempt_id="a1", order=order, eligibility=eligibility, service="refund", reason="damaged", amount=Decimal("10.00"), raw_token=raw, idempotency_key="id1")
        assert result["replayed"] is False
        assert repo.conn.execute("select status from aftersales_cases").fetchone()[0] == "REQUESTED"
        assert repo.conn.execute("select status from task_attempts where attempt_id='a1'").fetchone()[0] == "SUCCEEDED"
        assert raw not in "\n".join(str(r[0]) for r in repo.conn.execute("select envelope_json from trace_outbox"))
        assert repo.conn.execute("select trace_id from m2_audit_log where action='CASE_CREATED'").fetchone()[0]
        transitioned = AfterSalesService(repo).transition_after_sales(case_id=result["case_id"], actor="aftersales_service", expected_status="REQUESTED", target_status="APPROVED", reason="eligible", idempotency_key="id2")
        assert transitioned["status"] == "APPROVED"
        transition_replay = AfterSalesService(repo).transition_after_sales(case_id=result["case_id"], actor="aftersales_service", expected_status="REQUESTED", target_status="APPROVED", reason="eligible", idempotency_key="id2")
        assert transition_replay["status"] == "APPROVED" and transition_replay["state_version"] == transitioned["state_version"]
        with pytest.raises(AfterSalesError) as exc:
            AfterSalesService(repo).transition_after_sales(case_id=result["case_id"], actor="user", expected_status="APPROVED", target_status="REFUND_PENDING", reason="bad actor", idempotency_key="id3")
        assert exc.value.envelope.code == "STATE_TRANSITION_REJECTED"
    finally:
        repo.close()


def test_same_fingerprint_replays_and_active_unique_rejects(tmp_path):
    repo, task = setup_repo(tmp_path)
    try:
        order = order_fact()
        eligibility = _eligibility(order)
        manager = ConfirmTokenManager(repo)
        _, raw = manager.issue(session_id="s1", user_id="u1", run_id="r1", task_id=task.task_id, order_id="o1", service="refund", amount=Decimal("10.00"), payload_hash=order.snapshot_hash, topic_version="v1")
        service = AfterSalesService(repo)
        first = service.create_case(session_id="s1", user_id="u1", run_id="r1", task_id="t1", attempt_id="a1", order=order, eligibility=eligibility, service="refund", reason="damaged", amount=Decimal("10.00"), raw_token=raw, idempotency_key="id1")
        replay = service.create_case(session_id="s1", user_id="u1", run_id="r1", task_id="t1", attempt_id="a1", order=order, eligibility=eligibility, service="refund", reason="damaged", amount=Decimal("10.00"), raw_token="not-used", idempotency_key="id2")
        assert replay["replayed"] is True and replay["result_id"] == first["result_id"]
        assert repo.conn.execute("select action from m2_audit_log order by created_at desc limit 1").fetchone()[0] == "IDEMPOTENCY_CONFLICT"
        _, raw2 = manager.issue(session_id="s1", user_id="u1", run_id="r1", task_id=task.task_id, order_id="o1", service="refund", amount=Decimal("10.00"), payload_hash=order.snapshot_hash, topic_version="v1")
        with pytest.raises(AfterSalesError) as exc:
            service.create_case(session_id="s1", user_id="u1", run_id="r1", task_id="t1", attempt_id="a1", order=order.model_copy(update={"fact_id": "fact_2"}), eligibility=eligibility, service="refund", reason="different", amount=Decimal("10.00"), raw_token=raw2, idempotency_key="id3")
        assert exc.value.envelope.code == "ACTIVE_CASE_EXISTS"
    finally:
        repo.close()


def test_high_risk_write_enters_review_ticket_and_reviewer_decides(tmp_path):
    repo, task = setup_repo(tmp_path)
    try:
        order = order_fact()
        eligibility = _eligibility(order)
        _, raw = ConfirmTokenManager(repo).issue(session_id="s1", user_id="u1", run_id="r1", task_id=task.task_id, order_id="o1", service="refund", amount=Decimal("10.00"), payload_hash=order.snapshot_hash, topic_version="v1")
        created = AfterSalesService(repo).create_case(session_id="s1", user_id="u1", run_id="r1", task_id=task.task_id, attempt_id="a1", order=order, eligibility=eligibility, service="refund", reason="high risk", amount=Decimal("10.00"), raw_token=raw, idempotency_key="review-1", review_required=True)
        assert repo.conn.execute("select status from aftersales_cases").fetchone()[0] == "HUMAN_REVIEW"
        ticket = repo.conn.execute("select ticket_id,status from review_tickets").fetchone()
        assert ticket[1] == "OPEN"
        decision = AfterSalesService(repo).decide_review_ticket(ticket_id=ticket[0], actor="reviewer", decision="APPROVE", reason="reviewed")
        assert decision["status"] == "APPROVED"
        assert repo.conn.execute("select status from review_tickets").fetchone()[0] == "APPROVED"
    finally:
        repo.close()
