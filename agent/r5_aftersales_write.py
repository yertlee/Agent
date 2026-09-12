"""R5 guarded after-sales write closed loop: cancel and modify (ADR-0002 §5-7).

Only the trusted runtime issues previews and action tokens; the model never
supplies token authority.  Cancel and modify require an action-scoped token
bound to the target case, the expected ``state_version`` and the exact payload
being approved.  Every write commits token consumption, the case CAS update,
the immutable Result, the audit row and the trace outbox event in one
transaction; any failure rolls the whole transaction back.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any
from uuid import uuid4

from .domain.objects import Result, ResultStatus, sha256_json
from .m2_errors import ErrorCatalog
from .r5_write_contracts import (
    R5_ACTION_TOKEN_SCHEMA_VERSION,
    R5_ALLOWED_MODIFY_STATUSES,
    R5_CANCELLABLE_STATUSES,
    R5ActionPreview,
    R5ActionToken,
    action_binding_hash,
    preview_payload_hash,
)
from .storage.repositories import _now
from .trace.events import build_event


CANCEL_TOOL_REF = "aftersales/cancel@v1"
MODIFY_TOOL_REF = "aftersales/modify@v1"


class R5WriteError(RuntimeError):
    def __init__(self, envelope):
        self.envelope = envelope
        super().__init__(envelope.code)


ACTION_TOKEN_TABLE = """
CREATE TABLE IF NOT EXISTS r5_action_tokens (
  token_id TEXT PRIMARY KEY,
  token_hash TEXT NOT NULL UNIQUE,
  schema_version TEXT NOT NULL,
  action TEXT NOT NULL CHECK(action IN ('create','cancel','modify')),
  session_id TEXT NOT NULL,
  user_id TEXT NOT NULL,
  run_id TEXT NOT NULL,
  task_id TEXT NOT NULL,
  order_id TEXT NOT NULL,
  case_id TEXT,
  service TEXT NOT NULL,
  amount TEXT NOT NULL,
  payload_hash TEXT NOT NULL,
  expected_state_version INTEGER,
  topic_version TEXT NOT NULL,
  issued_at TEXT NOT NULL,
  expires_at TEXT NOT NULL,
  status TEXT NOT NULL,
  consumed_at TEXT,
  revoked_at TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_r5_action_token_hash ON r5_action_tokens(token_hash, status);
"""


class R5ActionTokenManager:
    """Hash-only, single-action, one-time confirmation tokens."""

    def __init__(self, repository):
        self.repository = repository
        self.repository.conn.executescript(ACTION_TOKEN_TABLE)
        self.repository.conn.commit()

    def issue(
        self,
        *,
        preview: R5ActionPreview,
        session_id: str,
        user_id: str,
        run_id: str,
        task_id: str,
        ttl_seconds: int = 300,
    ) -> tuple[R5ActionToken, str]:
        if preview.action not in {"create", "cancel", "modify"}:
            raise ValueError("invalid action")
        if preview.action in {"cancel", "modify"} and (not preview.case_id or preview.expected_state_version is None):
            raise ValueError("cancel/modify tokens require case_id and expected_state_version")
        raw = f"rat_{uuid4().hex}"
        now = datetime.now(timezone.utc)
        token = R5ActionToken(
            token_id=f"rtoken_{uuid4().hex}",
            token_hash=hashlib.sha256(raw.encode()).hexdigest(),
            action=preview.action,
            session_id=session_id,
            user_id=user_id,
            run_id=run_id,
            task_id=task_id,
            order_id=preview.order_id,
            case_id=preview.case_id,
            service=preview.service,
            amount=preview.amount,
            payload_hash=preview.payload_hash,
            expected_state_version=preview.expected_state_version,
            topic_version=preview.topic_version,
            issued_at=now,
            expires_at=now + timedelta(seconds=ttl_seconds),
        )
        bound = action_binding_hash(
            action=token.action, session_id=session_id, user_id=user_id, run_id=run_id, task_id=task_id,
            order_id=token.order_id, case_id=token.case_id, service=token.service, amount=token.amount,
            payload_hash=token.payload_hash, expected_state_version=token.expected_state_version, topic_version=token.topic_version,
        )
        try:
            self.repository.conn.execute("BEGIN IMMEDIATE")
            self.repository.conn.execute(
                "INSERT INTO r5_action_tokens VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    token.token_id, token.token_hash, R5_ACTION_TOKEN_SCHEMA_VERSION, token.action, session_id, user_id,
                    run_id, task_id, token.order_id, token.case_id, token.service, str(token.amount), token.payload_hash,
                    token.expected_state_version, token.topic_version, token.issued_at.isoformat(), token.expires_at.isoformat(),
                    "ISSUED", None, None, _now(), _now(),
                ),
            )
            self._token_event_locked(run_id=run_id, session_id=session_id, task_id=task_id, status="ISSUED", action=token.action, token_id=token.token_id, bound_hash=bound)
            self.repository.conn.commit()
            return token, raw
        except Exception:
            self.repository.conn.rollback()
            raise

    def consume_locked(
        self,
        raw_token: str,
        *,
        action: str,
        session_id: str,
        user_id: str,
        run_id: str,
        task_id: str,
        order_id: str,
        case_id: str | None,
        service: str,
        amount: Decimal,
        payload_hash: str,
        expected_state_version: int | None,
        topic_version: str,
    ) -> R5ActionToken:
        """Consume inside the caller's open transaction; never commits."""
        token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
        row = self.repository.conn.execute("SELECT * FROM r5_action_tokens WHERE token_hash=?", (token_hash,)).fetchone()
        if not row:
            raise R5WriteError(ErrorCatalog.envelope("CONTRACT_CONFIRM_REQUIRED"))
        values = tuple(row)
        # columns: 0 token_id,1 token_hash,2 schema_version,3 action,4 session,5 user,6 run,7 task,8 order,9 case,10 service,11 amount,12 payload_hash,13 expected_version,14 topic,15 issued,16 expires,17 status
        if str(values[3]) != str(action):
            raise R5WriteError(ErrorCatalog.envelope("AUTH_SCOPE_MISMATCH", details={"reason": "action_mismatch"}))
        expected = (session_id, user_id, run_id, task_id, order_id, case_id, service, str(amount), payload_hash, expected_state_version, topic_version)
        actual = (str(values[4]), str(values[5]), str(values[6]), str(values[7]), str(values[8]), values[9], str(values[10]), str(values[11]), str(values[12]), values[13], str(values[14]))
        if expected != actual:
            raise R5WriteError(ErrorCatalog.envelope("AUTH_SCOPE_MISMATCH", details={"reason": "token_binding"}))
        now = datetime.now(timezone.utc)
        expires = datetime.fromisoformat(str(values[16]).replace("Z", "+00:00"))
        if str(values[17]) != "ISSUED":
            raise R5WriteError(ErrorCatalog.envelope("CONTRACT_CONFIRM_EXPIRED", details={"reason": "token_not_issued"}))
        if now >= expires:
            self.repository.conn.execute("UPDATE r5_action_tokens SET status='EXPIRED',updated_at=? WHERE token_id=?", (_now(), values[0]))
            raise R5WriteError(ErrorCatalog.envelope("CONTRACT_CONFIRM_EXPIRED", details={"reason": "expired"}))
        bound = action_binding_hash(
            action=action, session_id=session_id, user_id=user_id, run_id=run_id, task_id=task_id, order_id=order_id,
            case_id=case_id, service=service, amount=amount, payload_hash=payload_hash,
            expected_state_version=expected_state_version, topic_version=topic_version,
        )
        updated = self.repository.conn.execute("UPDATE r5_action_tokens SET status='CONSUMED',consumed_at=?,updated_at=? WHERE token_id=? AND status='ISSUED'", (now.isoformat(), _now(), values[0]))
        if updated.rowcount != 1:
            raise R5WriteError(ErrorCatalog.envelope("CONTRACT_CONFIRM_EXPIRED", details={"reason": "token_consumed"}))
        self._token_event_locked(run_id=run_id, session_id=session_id, task_id=task_id, status="CONSUMED", action=action, token_id=values[0], bound_hash=bound)
        return R5ActionToken(
            token_id=values[0], token_hash=values[1], action=values[3], session_id=values[4], user_id=values[5], run_id=values[6],
            task_id=values[7], order_id=values[8], case_id=values[9], service=values[10], amount=Decimal(str(values[11])),
            payload_hash=values[12], expected_state_version=values[13], topic_version=values[14],
            issued_at=datetime.fromisoformat(str(values[15]).replace("Z", "+00:00")), expires_at=expires, status="CONSUMED", consumed_at=now,
        )

    def revoke(self, token_id: str, *, run_id: str, session_id: str, task_id: str, action: str) -> None:
        try:
            self.repository.conn.execute("BEGIN IMMEDIATE")
            cur = self.repository.conn.execute("UPDATE r5_action_tokens SET status='REVOKED',revoked_at=?,updated_at=? WHERE token_id=? AND status='ISSUED'", (_now(), _now(), token_id))
            if cur.rowcount != 1:
                raise R5WriteError(ErrorCatalog.envelope("CONTRACT_CONFIRM_EXPIRED", details={"reason": "token_not_issued"}))
            self._token_event_locked(run_id=run_id, session_id=session_id, task_id=task_id, status="REVOKED", action=action, token_id=token_id, bound_hash="redacted")
            self.repository.conn.commit()
        except Exception:
            self.repository.conn.rollback()
            raise

    def _token_event_locked(self, *, run_id: str, session_id: str, task_id: str, status: str, action: str, token_id: str, bound_hash: str) -> None:
        row = self.repository.conn.execute("SELECT next_seq_no FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            raise ValueError("run does not exist")
        parent = self.repository.conn.execute("SELECT event_id FROM trace_outbox WHERE run_id=? ORDER BY seq_no DESC LIMIT 1", (run_id,)).fetchone()
        event = build_event(
            run_id=run_id, session_id=session_id, event_type="TOKEN_EVENT", seq_no=int(row[0]), plan_revision_id=None,
            task_id=task_id, parent_event_id=str(parent[0]) if parent else None,
            payload={"status": status, "action": action, "token_id_hash": hashlib.sha256(str(token_id).encode()).hexdigest(), "bound_hash": bound_hash},
        )
        self.repository._append_m2_event_locked(event)


class R5AfterSalesWriteService:
    """Preview + guarded cancel/modify with one-transaction write semantics."""

    def __init__(self, repository):
        self.repository = repository
        self.tokens = R5ActionTokenManager(repository)

    # ---- preview (runtime-authored only) ----

    def build_preview(
        self,
        *,
        action: str,
        order_id: str,
        service: str,
        amount: Decimal,
        rule_explanation: str,
        case_id: str | None = None,
        expected_state_version: int | None = None,
        fields_to_change: dict[str, str] | None = None,
        topic_version: str = "v1",
    ) -> R5ActionPreview:
        fields = {str(k): str(v) for k, v in (fields_to_change or {}).items()}
        if action in {"cancel", "modify"} and (case_id is None or expected_state_version is None):
            raise ValueError("cancel/modify preview requires case_id and expected_state_version")
        payload_hash = preview_payload_hash(
            action=action, order_id=order_id, case_id=case_id, service=service, amount=amount,
            fields_to_change=fields, expected_state_version=expected_state_version, topic_version=topic_version,
        )
        preview = R5ActionPreview(
            preview_id=f"preview_{uuid4().hex}", action=action, order_id=order_id, case_id=case_id, service=service,
            amount=amount, fields_to_change=fields, rule_explanation=rule_explanation,
            expected_state_version=expected_state_version, topic_version=topic_version,
        )
        if preview.payload_hash != payload_hash:
            preview = preview.model_copy(update={"payload_hash": payload_hash})
        return preview

    def issue_action_token(self, *, preview: R5ActionPreview, session_id: str, user_id: str, run_id: str, task_id: str, ttl_seconds: int = 300) -> tuple[R5ActionToken, str]:
        return self.tokens.issue(preview=preview, session_id=session_id, user_id=user_id, run_id=run_id, task_id=task_id, ttl_seconds=ttl_seconds)

    # ---- guarded writes ----

    def cancel_case(
        self,
        *,
        raw_token: str,
        session_id: str,
        user_id: str,
        run_id: str,
        task_id: str,
        attempt_id: str,
        case_id: str,
        idempotency_key: str,
        reason: str,
        topic_version: str = "v1",
    ) -> dict[str, Any]:
        return self._write(
            action="cancel", tool_ref=CANCEL_TOOL_REF, raw_token=raw_token, session_id=session_id, user_id=user_id,
            run_id=run_id, task_id=task_id, attempt_id=attempt_id, case_id=case_id, idempotency_key=idempotency_key,
            reason=reason, topic_version=topic_version, target_status="CANCELLED", allowed_statuses=R5_CANCELLABLE_STATUSES,
        )

    def modify_case(
        self,
        *,
        raw_token: str,
        session_id: str,
        user_id: str,
        run_id: str,
        task_id: str,
        attempt_id: str,
        case_id: str,
        new_reason: str,
        expected_state_version: int,
        idempotency_key: str,
        topic_version: str = "v1",
    ) -> dict[str, Any]:
        return self._write(
            action="modify", tool_ref=MODIFY_TOOL_REF, raw_token=raw_token, session_id=session_id, user_id=user_id,
            run_id=run_id, task_id=task_id, attempt_id=attempt_id, case_id=case_id, idempotency_key=idempotency_key,
            reason=new_reason, topic_version=topic_version, target_status=None, allowed_statuses=R5_ALLOWED_MODIFY_STATUSES,
            expected_state_version=expected_state_version, new_reason=new_reason,
        )

    def _write(
        self,
        *,
        action: str,
        tool_ref: str,
        raw_token: str,
        session_id: str,
        user_id: str,
        run_id: str,
        task_id: str,
        attempt_id: str,
        case_id: str,
        idempotency_key: str,
        reason: str,
        topic_version: str,
        target_status: str | None,
        allowed_statuses: frozenset[str],
        expected_state_version: int | None = None,
        new_reason: str | None = None,
    ) -> dict[str, Any]:
        fields = {"reason": str(new_reason)} if action == "modify" else {}
        try:
            self.repository.conn.execute("BEGIN IMMEDIATE")
            row = self.repository.conn.execute("SELECT case_id,user_id,order_id,service,status,state_version,run_id,session_id,task_id,reason,amount FROM aftersales_cases WHERE case_id=?", (case_id,)).fetchone()
            if row is None:
                raise R5WriteError(ErrorCatalog.envelope("CONTRACT_INVALID_STATE", details={"case_id": case_id, "reason": "case_not_found"}))
            case_user = str(row[1])
            if case_user != str(user_id):
                raise R5WriteError(ErrorCatalog.envelope("AUTH_RESOURCE_FORBIDDEN", details={"case_id": case_id}))
            if str(row[6]) != str(run_id) or str(row[7]) != str(session_id) or str(row[8]) != str(task_id):
                raise R5WriteError(ErrorCatalog.envelope("AUTH_SCOPE_MISMATCH", details={"reason": "case_owner_context", "case_id": case_id}))
            case_status = str(row[4])
            case_version = int(row[5])
            if action == "cancel" and expected_state_version is None:
                # The cancel preview must have been built against the current version.
                expected_state_version = case_version
            # Business-request identity excludes state_version: the version is a
            # concurrency/binding guard, not part of what the user asked for.
            # Otherwise a retry after a successful write would look like a new
            # request and could not replay the original Result.
            fingerprint = sha256_json({"action": action, "user_id": user_id, "case_id": case_id, "reason": str(reason)})
            # Idempotent replay is resolved before state validation so a retried
            # successful request returns the original Result instead of a
            # state-transition error.
            replay = self.repository.conn.execute(
                "SELECT result_id,idempotency_key FROM tool_submit_log WHERE user_id=? AND tool_ref=? AND request_fingerprint=?",
                (user_id, tool_ref, fingerprint),
            ).fetchone()
            if replay:
                payload_row = self.repository.conn.execute("SELECT payload_json FROM results WHERE result_id=?", (replay[0],)).fetchone()
                audit_action = "IDEMPOTENCY_REPLAY" if str(replay[1]) == str(idempotency_key) else "IDEMPOTENCY_CONFLICT"
                self.repository.log_audit_locked(run_id=run_id, actor="runtime", action=audit_action, reason=f"{action} replay", trace_id=None)
                self.repository.conn.commit()
                return {"replayed": True, "result_id": replay[0], "payload": payload_row[0] if payload_row else None, "replay_kind": audit_action}
            collision = self.repository.conn.execute("SELECT request_fingerprint FROM tool_submit_log WHERE idempotency_key=?", (idempotency_key,)).fetchone()
            if collision and str(collision[0]) != fingerprint:
                raise R5WriteError(ErrorCatalog.envelope("IDEMPOTENCY_CONFLICT"))
            if case_status not in allowed_statuses:
                raise R5WriteError(ErrorCatalog.envelope("STATE_TRANSITION_REJECTED", details={"status": case_status, "action": action}))
            if action == "modify" and expected_state_version is not None and expected_state_version != case_version:
                raise R5WriteError(ErrorCatalog.envelope("CONTRACT_INVALID_STATE", details={"reason": "expected_version_mismatch", "case_version": case_version}))
            if action == "modify" and str(row[9]) == str(new_reason):
                raise R5WriteError(ErrorCatalog.envelope("STATE_TRANSITION_REJECTED", details={"reason": "no_field_change"}))
            # Token consumption happens inside this same transaction.
            self.tokens.consume_locked(
                raw_token, action=action, session_id=session_id, user_id=user_id, run_id=run_id, task_id=task_id,
                order_id=str(row[2]), case_id=case_id, service=str(row[3]), amount=Decimal(str(row[10] or "0")),
                payload_hash=preview_payload_hash(
                    action=action, order_id=str(row[2]), case_id=case_id, service=str(row[3]),
                    amount=Decimal(str(row[10] or "0")), fields_to_change=fields,
                    expected_state_version=expected_state_version, topic_version=topic_version,
                ),
                expected_state_version=expected_state_version, topic_version=topic_version,
            )
            new_version = case_version + 1
            if action == "cancel":
                cur = self.repository.conn.execute(
                    "UPDATE aftersales_cases SET status=?,state_version=?,idempotency_key=?,updated_at=? WHERE case_id=? AND status=? AND state_version=?",
                    ("CANCELLED", new_version, idempotency_key, _now(), case_id, case_status, case_version),
                )
            else:
                cur = self.repository.conn.execute(
                    "UPDATE aftersales_cases SET reason=?,state_version=?,idempotency_key=?,updated_at=? WHERE case_id=? AND status=? AND state_version=?",
                    (str(new_reason), new_version, idempotency_key, _now(), case_id, case_status, case_version),
                )
            if cur.rowcount != 1:
                raise R5WriteError(ErrorCatalog.envelope("CONTRACT_INVALID_STATE", details={"reason": "concurrent_state_change", "case_id": case_id}))
            final_status = "CANCELLED" if action == "cancel" else case_status
            plan_row = self.repository.conn.execute("SELECT plan_revision_id FROM tasks WHERE task_id=?", (task_id,)).fetchone()
            if plan_row is None:
                raise ValueError("task does not exist")
            result_id = f"result_{uuid4().hex}"
            payload = {"case_id": case_id, "status": final_status, "action": action}
            if action == "modify":
                payload["reason"] = str(new_reason)
            result = Result(
                result_id=result_id, run_id=run_id, plan_revision_id=str(plan_row[0]), task_id=task_id, attempt_id=attempt_id,
                status=ResultStatus.SUCCEEDED, output_contract="aftersales.result.v1", payload=payload, business_code="OK",
            )
            parent = self.repository.conn.execute("SELECT event_id FROM trace_outbox WHERE run_id=? ORDER BY seq_no DESC LIMIT 1", (run_id,)).fetchone()
            transition_event = build_event(
                run_id=run_id, session_id=session_id, event_type="TASK_STATE_CHANGED", seq_no=int(self.repository.conn.execute("SELECT next_seq_no FROM runs WHERE run_id=?", (run_id,)).fetchone()[0]),
                plan_revision_id=str(plan_row[0]), task_id=task_id, attempt_id=attempt_id,
                parent_event_id=str(parent[0]) if parent else None,
                payload={"task_id": task_id, "actor": "aftersales_service", "from": case_status, "to": final_status, "state_version": new_version, "reason": reason, "action": action},
            )
            self.repository._append_m2_event_locked(transition_event)
            self.repository._append_result_locked(result)
            self.repository.conn.execute("INSERT INTO tool_submit_log VALUES (?,?,?,?,?,?,?,?,?,?,?)", (f"submit_{uuid4().hex}", user_id, run_id, task_id, tool_ref, fingerprint, idempotency_key, result_id, "COMMITTED", _now(), _now()))
            self.repository.log_audit_locked(run_id=run_id, actor="aftersales_service", action=f"CASE_{action.upper()}", reason=reason, trace_id=transition_event.trace_id, after_hash=sha256_json({"case_id": case_id, "status": final_status, "state_version": new_version}))
            self.repository.conn.commit()
            return {"replayed": False, "case_id": case_id, "status": final_status, "state_version": new_version, "result_id": result_id, "result": result}
        except R5WriteError:
            self.repository.conn.rollback()
            raise
        except Exception:
            self.repository.conn.rollback()
            raise


__all__ = ["CANCEL_TOOL_REF", "MODIFY_TOOL_REF", "R5ActionTokenManager", "R5AfterSalesWriteService", "R5WriteError"]
