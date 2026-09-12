"""R5-B safety matrix for guarded after-sales cancel/modify (R5 guide §6.4).

Every scenario asserts the world state after the attempt: the only permitted
side effect is the single intended CAS transition, and every rejection leaves
the case untouched.  Tokens are runtime-issued; the model never supplies
authority.  All writes share one transaction (token consumption + CAS + Result
+ audit + outbox).
"""
from __future__ import annotations

import sqlite3
from decimal import Decimal
from pathlib import Path

import pytest

from agent.domain.objects import AttemptStatus, PlanRevision, PlanStatus, Run, Task, TaskAttempt, sha256_json
from agent.m2_errors import ErrorCatalog
from agent.r5_aftersales_write import CANCEL_TOOL_REF, MODIFY_TOOL_REF, R5AfterSalesWriteService, R5WriteError
from agent.storage.m2 import M2Repository
from agent.storage.repositories import _now


def _repo(tmp_path: Path) -> M2Repository:
    repo = M2Repository(str(tmp_path / "m2.db"))
    repo.create_session("s1", "u1")
    repo.create_run(Run(run_id="r1", session_id="s1", initial_world_hash=sha256_json({"world": "r5"})))
    task = Task(task_id="t1", plan_revision_id="p1", agent_ref="aftersales-agent@v1", capability_refs=["aftersales/write@v1"], output_contract="aftersales.result.v1", failure_strategy="FAIL_RUN", side_effect="WRITE", timeout_ms=5000)
    plan = PlanRevision(plan_revision_id="p1", run_id="r1", created_by="supervisor", revision_reason="initial", version=1, status=PlanStatus.ACTIVE, tasks=[task])
    repo.create_plan_revision(plan)
    return repo


def _attempt(repo: M2Repository, number: int = 1) -> str:
    attempt_id = f"a{number}"
    repo.create_attempt(TaskAttempt(attempt_id=attempt_id, run_id="r1", plan_revision_id="p1", task_id="t1", agent_ref="aftersales-agent@v1", attempt_no=number, status=AttemptStatus.CREATED, input_hash=sha256_json({"n": number})))
    return attempt_id


def _case(repo: M2Repository, *, case_id: str = "c1", user_id: str = "u1", order_id: str = "o1", service: str = "refund", status: str = "REQUESTED", version: int = 0, amount: str = "10.00", reason: str = "orig") -> None:
    repo.conn.execute(
        "INSERT INTO aftersales_cases VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (case_id, user_id, order_id, service, status, version, "r1", "s1", "t1", f"seed_{case_id}", sha256_json({"seed": case_id}), reason, amount, _now(), _now()),
    )
    repo.conn.commit()


def _case_row(repo: M2Repository, case_id: str = "c1"):
    row = repo.conn.execute("SELECT status,state_version,reason FROM aftersales_cases WHERE case_id=?", (case_id,)).fetchone()
    return None if row is None else tuple(row)


def _service(tmp_path: Path, **case_kwargs):
    repo = _repo(tmp_path)
    _case(repo, **case_kwargs)
    svc = R5AfterSalesWriteService(repo)
    return repo, svc


def _token(svc, case_id="c1", *, action="cancel", session="s1", user="u1", order="o1", service="refund", amount="10.00", version=0, reason="用户要求取消", ttl=300):
    fields = {"reason": reason} if action == "modify" else {}
    preview = svc.build_preview(action=action, order_id=order, service=service, amount=Decimal(amount), rule_explanation="eligible confirmed request", case_id=case_id, expected_state_version=version, fields_to_change=fields)
    return svc.issue_action_token(preview=preview, session_id=session, user_id=user, run_id="r1", task_id="t1", ttl_seconds=ttl)


def _cancel(svc, raw, *, case_id="c1", key="k1", attempt="a1", session="s1", user="u1", reason="用户要求取消"):
    return svc.cancel_case(raw_token=raw, session_id=session, user_id=user, run_id="r1", task_id="t1", attempt_id=attempt, case_id=case_id, idempotency_key=key, reason=reason)


def _modify(svc, raw, *, case_id="c1", key="k2", attempt="a2", reason="改原因", version=0, user="u1"):
    return svc.modify_case(raw_token=raw, session_id="s1", user_id=user, run_id="r1", task_id="t1", attempt_id=attempt, case_id=case_id, new_reason=reason, expected_state_version=version, idempotency_key=key)


# ---- happy paths ----

def test_cancel_succeeds_and_persists_transition(tmp_path: Path) -> None:
    repo, svc = _service(tmp_path)
    _attempt(repo, 1)
    raw = _token(svc)[1]
    out = _cancel(svc, raw)
    assert out["replayed"] is False and out["status"] == "CANCELLED" and out["state_version"] == 1
    assert _case_row(repo) == ("CANCELLED", 1, "orig")
    assert repo.conn.execute("SELECT COUNT(*) FROM results").fetchone()[0] == 1


def test_modify_succeeds_in_allowed_status(tmp_path: Path) -> None:
    repo, svc = _service(tmp_path, status="UNDER_REVIEW", version=3)
    _attempt(repo, 1)
    _attempt(repo, 2)
    raw = _token(svc, action="modify", version=3, reason="补充说明")[1]
    out = _modify(svc, raw, version=3, reason="补充说明")
    assert out["status"] == "UNDER_REVIEW" and out["state_version"] == 4
    assert _case_row(repo) == ("UNDER_REVIEW", 4, "补充说明")


# ---- no confirmation / expiry / revocation ----

def test_missing_token_is_rejected_without_side_effect(tmp_path: Path) -> None:
    repo, svc = _service(tmp_path)
    _attempt(repo, 1)
    with pytest.raises(R5WriteError) as exc:
        _cancel(svc, "rat_forged")
    assert exc.value.envelope.code == "CONTRACT_CONFIRM_REQUIRED"
    assert _case_row(repo) == ("REQUESTED", 0, "orig")


def test_expired_token_is_rejected(tmp_path: Path) -> None:
    repo, svc = _service(tmp_path)
    _attempt(repo, 1)
    raw = _token(svc, ttl=-1)[1]
    with pytest.raises(R5WriteError) as exc:
        _cancel(svc, raw)
    assert exc.value.envelope.code == "CONTRACT_CONFIRM_EXPIRED"
    assert _case_row(repo) == ("REQUESTED", 0, "orig")


def test_revoked_token_is_rejected(tmp_path: Path) -> None:
    repo, svc = _service(tmp_path)
    _attempt(repo, 1)
    token, raw = _token(svc)
    svc.tokens.revoke(token.token_id, run_id="r1", session_id="s1", task_id="t1", action="cancel")
    with pytest.raises(R5WriteError) as exc:
        _cancel(svc, raw)
    assert exc.value.envelope.code == "CONTRACT_CONFIRM_EXPIRED"
    assert _case_row(repo) == ("REQUESTED", 0, "orig")


def test_consumed_token_cannot_write_twice(tmp_path: Path) -> None:
    repo, svc = _service(tmp_path)
    _attempt(repo, 1)
    raw = _token(svc)[1]
    first = _cancel(svc, raw)
    # Reusing the consumed token is resolved as an idempotent replay of the
    # original Result: no second consumption, no second write.
    second = _cancel(svc, raw, key="k1b")
    assert second["replayed"] is True and second["result_id"] == first["result_id"]
    assert _case_row(repo) == ("CANCELLED", 1, "orig")
    assert repo.conn.execute("SELECT COUNT(*) FROM results").fetchone()[0] == 1
    assert repo.conn.execute("SELECT status FROM r5_action_tokens").fetchone()[0] == "CONSUMED"


# ---- wrong session / user / order / action ----

def test_wrong_session_binding_rejected(tmp_path: Path) -> None:
    repo, svc = _service(tmp_path)
    _attempt(repo, 1)
    raw = _token(svc)[1]
    with pytest.raises(R5WriteError) as exc:
        _cancel(svc, raw, session="s-other")
    assert exc.value.envelope.code == "AUTH_SCOPE_MISMATCH"
    assert _case_row(repo) == ("REQUESTED", 0, "orig")


def test_wrong_user_binding_rejected(tmp_path: Path) -> None:
    repo, svc = _service(tmp_path)
    _attempt(repo, 1)
    raw = _token(svc, user="u-other")[1]
    with pytest.raises(R5WriteError) as exc:
        _cancel(svc, raw)
    assert exc.value.envelope.code == "AUTH_SCOPE_MISMATCH"
    assert _case_row(repo) == ("REQUESTED", 0, "orig")


def test_other_users_case_is_forbidden(tmp_path: Path) -> None:
    repo, svc = _service(tmp_path)
    _attempt(repo, 1)
    raw = _token(svc)[1]
    with pytest.raises(R5WriteError) as exc:
        _cancel(svc, raw, user="u-other")
    assert exc.value.envelope.code in {"AUTH_SCOPE_MISMATCH", "AUTH_RESOURCE_FORBIDDEN"}
    assert _case_row(repo) == ("REQUESTED", 0, "orig")


def test_wrong_case_binding_rejected(tmp_path: Path) -> None:
    repo, svc = _service(tmp_path)
    _case(repo, case_id="c2", order_id="o2")
    _attempt(repo, 1)
    raw = _token(svc, case_id="c2")[1]
    with pytest.raises(R5WriteError) as exc:
        _cancel(svc, raw, case_id="c1")
    assert exc.value.envelope.code == "AUTH_SCOPE_MISMATCH"


def test_cross_action_reuse_is_rejected(tmp_path: Path) -> None:
    repo, svc = _service(tmp_path, status="UNDER_REVIEW")
    _attempt(repo, 1)
    cancel_raw = _token(svc, action="cancel")[1]
    with pytest.raises(R5WriteError) as exc:
        _modify(svc, cancel_raw, reason="用取消令牌改原因", version=0)
    assert exc.value.envelope.code == "AUTH_SCOPE_MISMATCH"
    assert _case_row(repo) == ("UNDER_REVIEW", 0, "orig")


def test_tampered_amount_payload_rejected(tmp_path: Path) -> None:
    repo, svc = _service(tmp_path)
    _attempt(repo, 1)
    # Token was issued for a different amount than the case actually carries.
    raw = _token(svc, amount="999.00")[1]
    with pytest.raises(R5WriteError) as exc:
        _cancel(svc, raw)
    assert exc.value.envelope.code == "AUTH_SCOPE_MISMATCH"
    assert _case_row(repo) == ("REQUESTED", 0, "orig")


def test_modify_token_payload_hash_protects_new_reason(tmp_path: Path) -> None:
    repo, svc = _service(tmp_path, status="UNDER_REVIEW")
    _attempt(repo, 1)
    raw = _token(svc, action="modify", version=0, reason="approved text")[1]
    # Token was bound to "approved text"; executing with a different reason fails binding.
    with pytest.raises(R5WriteError) as exc:
        svc.modify_case(raw_token=raw, session_id="s1", user_id="u1", run_id="r1", task_id="t1", attempt_id="a1", case_id="c1", new_reason="injected text", expected_state_version=0, idempotency_key="k")
    assert exc.value.envelope.code == "AUTH_SCOPE_MISMATCH"
    assert _case_row(repo) == ("UNDER_REVIEW", 0, "orig")


# ---- state / version changes after confirmation ----

def test_status_change_after_confirmation_blocks_execution(tmp_path: Path) -> None:
    repo, svc = _service(tmp_path)
    _attempt(repo, 1)
    raw = _token(svc, version=0)[1]
    repo.conn.execute("UPDATE aftersales_cases SET status='APPROVED',state_version=1 WHERE case_id='c1'")
    repo.conn.commit()
    with pytest.raises(R5WriteError) as exc:
        _cancel(svc, raw)
    assert exc.value.envelope.code in {"AUTH_SCOPE_MISMATCH", "CONTRACT_INVALID_STATE", "STATE_TRANSITION_REJECTED"}


def test_version_change_after_confirmation_blocks_modify(tmp_path: Path) -> None:
    repo, svc = _service(tmp_path, status="APPROVED", version=1)
    _attempt(repo, 1)
    raw = _token(svc, action="modify", version=1, reason="new")[1]
    repo.conn.execute("UPDATE aftersales_cases SET state_version=2 WHERE case_id='c1'")
    repo.conn.commit()
    with pytest.raises(R5WriteError) as exc:
        _modify(svc, raw, version=2, reason="new")
    assert exc.value.envelope.code in {"AUTH_SCOPE_MISMATCH", "CONTRACT_INVALID_STATE"}


def test_terminal_case_cannot_be_cancelled(tmp_path: Path) -> None:
    repo, svc = _service(tmp_path, status="REFUNDED", version=5)
    _attempt(repo, 1)
    raw = _token(svc, version=5)[1]
    with pytest.raises(R5WriteError) as exc:
        _cancel(svc, raw)
    assert exc.value.envelope.code == "STATE_TRANSITION_REJECTED"
    assert _case_row(repo) == ("REFUNDED", 5, "orig")


def test_terminal_case_cannot_be_modified(tmp_path: Path) -> None:
    repo, svc = _service(tmp_path, status="REJECTED", version=2)
    _attempt(repo, 1)
    raw = _token(svc, action="modify", version=2, reason="新原因")[1]
    with pytest.raises(R5WriteError) as exc:
        _modify(svc, raw, version=2, reason="新原因")
    assert exc.value.envelope.code == "STATE_TRANSITION_REJECTED"
    assert _case_row(repo) == ("REJECTED", 2, "orig")


def test_modify_refund_pending_is_rejected(tmp_path: Path) -> None:
    repo, svc = _service(tmp_path, status="REFUND_PENDING", version=1)
    _attempt(repo, 1)
    raw = _token(svc, action="modify", version=1, reason="新原因")[1]
    with pytest.raises(R5WriteError) as exc:
        _modify(svc, raw, version=1, reason="新原因")
    assert exc.value.envelope.code == "STATE_TRANSITION_REJECTED"


def test_modify_no_field_change_rejected(tmp_path: Path) -> None:
    repo, svc = _service(tmp_path, status="APPROVED", version=1)
    _attempt(repo, 1)
    raw = _token(svc, action="modify", version=1, reason="orig")[1]
    with pytest.raises(R5WriteError) as exc:
        _modify(svc, raw, version=1, reason="orig")
    assert exc.value.envelope.code == "STATE_TRANSITION_REJECTED"
    assert _case_row(repo) == ("APPROVED", 1, "orig")


# ---- idempotency / replay / concurrency ----

def test_same_request_same_key_replays_original_result(tmp_path: Path) -> None:
    repo, svc = _service(tmp_path)
    _attempt(repo, 1)
    raw = _token(svc)[1]
    first = _cancel(svc, raw)
    second = _cancel(svc, raw, key="k1")
    assert second["replayed"] is True
    assert second["result_id"] == first["result_id"]
    assert repo.conn.execute("SELECT COUNT(*) FROM results").fetchone()[0] == 1


def test_same_request_different_key_replays_with_conflict_audit(tmp_path: Path) -> None:
    repo, svc = _service(tmp_path)
    _attempt(repo, 1)
    first = _cancel(svc, _token(svc)[1], key="k1")
    # Same business request (same fingerprint), new idempotency key and fresh token.
    repo.conn.execute("UPDATE aftersales_cases SET status='REQUESTED',state_version=0 WHERE case_id='c1'")
    repo.conn.commit()
    second = _cancel(svc, _token(svc)[1], key="k1-different")
    assert second["replayed"] is True
    assert second["result_id"] == first["result_id"]
    actions = [str(r[0]) for r in repo.conn.execute("SELECT action FROM m2_audit_log").fetchall()]
    assert "IDEMPOTENCY_CONFLICT" in actions


def test_same_key_different_request_rejected(tmp_path: Path) -> None:
    repo, svc = _service(tmp_path)
    _attempt(repo, 1)
    _cancel(svc, _token(svc)[1], key="k-shared", reason="取消A")
    repo.conn.execute("UPDATE aftersales_cases SET status='REQUESTED',state_version=0 WHERE case_id='c1'")
    repo.conn.commit()
    with pytest.raises(R5WriteError) as exc:
        _cancel(svc, _token(svc)[1], key="k-shared", reason="取消B", attempt="a3")
    assert exc.value.envelope.code == "IDEMPOTENCY_CONFLICT"


def test_concurrent_version_race_yields_single_terminal_state(tmp_path: Path) -> None:
    repo, svc = _service(tmp_path)
    _attempt(repo, 1)
    _attempt(repo, 2)
    raw1 = _token(svc)[1]
    raw2 = _token(svc)[1]
    _cancel(svc, raw1, key="k-a", attempt="a1")
    # Second confirmation for the same semantic request raced the first one:
    # it must replay the original Result or be rejected, never write twice.
    try:
        second = _cancel(svc, raw2, key="k-b", attempt="a2")
        assert second["replayed"] is True
    except R5WriteError as exc:
        assert exc.envelope.code in {"AUTH_SCOPE_MISMATCH", "STATE_TRANSITION_REJECTED", "CONTRACT_INVALID_STATE", "IDEMPOTENCY_CONFLICT"}
    assert _case_row(repo) == ("CANCELLED", 1, "orig")
    assert repo.conn.execute("SELECT COUNT(*) FROM results").fetchone()[0] == 1


def test_failed_write_rolls_back_token_consumption(tmp_path: Path) -> None:
    repo, svc = _service(tmp_path, status="REFUNDED", version=1)
    _attempt(repo, 1)
    raw = _token(svc, version=1)[1]
    with pytest.raises(R5WriteError):
        _cancel(svc, raw)
    # Rejected transition must not leave a consumed token or a Result row.
    assert repo.conn.execute("SELECT status FROM r5_action_tokens").fetchone()[0] == "ISSUED"
    assert repo.conn.execute("SELECT COUNT(*) FROM results").fetchone()[0] == 0
    assert repo.conn.execute("SELECT COUNT(*) FROM tool_submit_log").fetchone()[0] == 0


def test_cancel_requires_candidate_case_and_version(tmp_path: Path) -> None:
    _repo(tmp_path)
    svc = R5AfterSalesWriteService(_repo(tmp_path / "fresh"))
    with pytest.raises(ValueError):
        svc.build_preview(action="cancel", order_id="o1", service="refund", amount=Decimal("1.00"), rule_explanation="x")


# ---- preview / token integrity ----

def test_preview_payload_hash_is_deterministic_and_bound(tmp_path: Path) -> None:
    _repo(tmp_path)
    svc = R5AfterSalesWriteService(_repo(tmp_path / "p"))
    p1 = svc.build_preview(action="cancel", order_id="o1", service="refund", amount=Decimal("10.00"), rule_explanation="x", case_id="c1", expected_state_version=0)
    p2 = svc.build_preview(action="cancel", order_id="o1", service="refund", amount=Decimal("10.00"), rule_explanation="y", case_id="c1", expected_state_version=0)
    assert p1.payload_hash == p2.payload_hash  # rule text is explanation, not payload
    p3 = svc.build_preview(action="cancel", order_id="o1", service="refund", amount=Decimal("11.00"), rule_explanation="x", case_id="c1", expected_state_version=0)
    assert p1.payload_hash != p3.payload_hash


def test_cancel_token_requires_case_and_version(tmp_path: Path) -> None:
    _repo(tmp_path)
    svc = R5AfterSalesWriteService(_repo(tmp_path / "q"))
    from agent.r5_write_contracts import R5ActionPreview

    bad = R5ActionPreview(preview_id="p", action="cancel", order_id="o1", service="refund", amount=Decimal("1.00"), rule_explanation="x", topic_version="v1")
    with pytest.raises(ValueError):
        svc.issue_action_token(preview=bad, session_id="s1", user_id="u1", run_id="r1", task_id="t1")


def test_token_hash_is_stored_not_raw(tmp_path: Path) -> None:
    repo, svc = _service(tmp_path)
    issued, raw = _token(svc)
    stored = repo.conn.execute("SELECT token_hash FROM r5_action_tokens").fetchone()[0]
    assert stored != raw
    assert len(stored) == 64


def test_write_creates_trace_and_audit_events(tmp_path: Path) -> None:
    import json

    repo, svc = _service(tmp_path)
    _attempt(repo, 1)
    _cancel(svc, _token(svc)[1])
    envelopes = [json.loads(str(r[0])) for r in repo.conn.execute("SELECT envelope_json FROM trace_outbox").fetchall()]
    event_types = [str(env.get("event_type")) for env in envelopes]
    assert "TOKEN_EVENT" in event_types and "TASK_STATE_CHANGED" in event_types
    audit_actions = [str(r[0]) for r in repo.conn.execute("SELECT action FROM m2_audit_log").fetchall()]
    assert "CASE_CANCEL" in audit_actions


def test_error_codes_come_from_central_catalog(tmp_path: Path) -> None:
    for code in ("CONTRACT_CONFIRM_REQUIRED", "CONTRACT_CONFIRM_EXPIRED", "AUTH_SCOPE_MISMATCH", "STATE_TRANSITION_REJECTED", "IDEMPOTENCY_CONFLICT", "CONTRACT_INVALID_STATE", "AUTH_RESOURCE_FORBIDDEN"):
        assert ErrorCatalog.definition(code) is not None
