"""Hash-only ConfirmToken issuance and consumption (02 §4-5)."""
from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from agent.domain.objects import HASH_PATTERN, sha256_json
from agent.m2_errors import ErrorCatalog
from agent.storage.repositories import _now
from agent.trace.events import build_event


class TokenError(RuntimeError):
    def __init__(self, envelope):
        self.envelope = envelope
        super().__init__(envelope.code)


class ConfirmToken(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    token_id: str = Field(min_length=1)
    token_hash: str = Field(pattern=HASH_PATTERN)
    session_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    order_id: str = Field(min_length=1)
    service: str = Field(min_length=1)
    amount: Decimal
    payload_hash: str = Field(pattern=HASH_PATTERN)
    topic_version: str = Field(min_length=1)
    issued_at: datetime
    expires_at: datetime
    status: str = "ISSUED"
    consumed_at: datetime | None = None
    revoked_at: datetime | None = None


def token_binding_hash(*, session_id: str, user_id: str, run_id: str, task_id: str, order_id: str, service: str, amount: Decimal, payload_hash: str, topic_version: str) -> str:
    return sha256_json({"session_id": session_id, "user_id": user_id, "run_id": run_id, "task_id": task_id, "order_id": order_id, "service": service, "amount": str(amount), "payload_hash": payload_hash, "topic_version": topic_version})


class ConfirmTokenManager:
    def __init__(self, repository):
        self.repository = repository

    def issue(self, *, session_id: str, user_id: str, run_id: str, task_id: str, order_id: str, service: str, amount: Decimal, payload_hash: str, topic_version: str, ttl_seconds: int = 300) -> tuple[ConfirmToken, str]:
        if service not in {"refund", "return", "exchange"}:
            raise ValueError("invalid token service")
        raw = f"ct_{uuid4().hex}"
        now = datetime.now(timezone.utc)
        token = ConfirmToken(token_id=f"token_{uuid4().hex}", token_hash=hashlib.sha256(raw.encode()).hexdigest(), session_id=session_id, user_id=user_id, run_id=run_id, task_id=task_id, order_id=order_id, service=service, amount=amount, payload_hash=payload_hash, topic_version=topic_version, issued_at=now, expires_at=now + timedelta(seconds=ttl_seconds))
        bound_hash = token_binding_hash(session_id=session_id, user_id=user_id, run_id=run_id, task_id=task_id, order_id=order_id, service=service, amount=amount, payload_hash=payload_hash, topic_version=topic_version)
        try:
            self.repository.conn.execute("BEGIN IMMEDIATE")
            self.repository.conn.execute(
                "INSERT INTO confirm_tokens VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (token.token_id, token.token_hash, session_id, user_id, run_id, task_id, order_id, service, str(amount), payload_hash, topic_version, token.issued_at.isoformat(), token.expires_at.isoformat(), "ISSUED", None, None, _now(), _now()),
            )
            row = self.repository.conn.execute("SELECT next_seq_no FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if not row:
                raise ValueError("run does not exist")
            parent = self.repository.conn.execute("SELECT event_id FROM trace_outbox WHERE run_id=? ORDER BY seq_no DESC LIMIT 1", (run_id,)).fetchone()
            event = build_event(run_id=run_id, session_id=session_id, event_type="TOKEN_EVENT", seq_no=int(row[0]), plan_revision_id=None, task_id=task_id, parent_event_id=str(parent[0]) if parent else None, payload={"token_id_hash": hashlib.sha256(token.token_id.encode()).hexdigest(), "status": "ISSUED", "bound_hash": bound_hash})
            self.repository._append_m2_event_locked(event)
            self.repository.conn.commit()
            return token, raw
        except TokenError as exc:
            if exc.envelope.code == "CONTRACT_CONFIRM_EXPIRED" and exc.envelope.details.get("reason") == "expired":
                self.repository.conn.commit()
            else:
                self.repository.conn.rollback()
            raise
        except Exception:
            self.repository.conn.rollback()
            raise

    def consume(self, raw_token: str, *, session_id: str, user_id: str, run_id: str, task_id: str, order_id: str, service: str, amount: Decimal, payload_hash: str, topic_version: str) -> ConfirmToken:
        try:
            self.repository.conn.execute("BEGIN IMMEDIATE")
            token = self._consume_locked(raw_token, session_id=session_id, user_id=user_id, run_id=run_id, task_id=task_id, order_id=order_id, service=service, amount=amount, payload_hash=payload_hash, topic_version=topic_version)
            self.repository.conn.commit()
            return token
        except TokenError as exc:
            if exc.envelope.code == "CONTRACT_CONFIRM_EXPIRED" and exc.envelope.details.get("reason") == "expired":
                self.repository.conn.commit()
            else:
                self.repository.conn.rollback()
            raise
        except Exception:
            self.repository.conn.rollback()
            raise

    def _consume_locked(self, raw_token: str, *, session_id: str, user_id: str, run_id: str, task_id: str, order_id: str, service: str, amount: Decimal, payload_hash: str, topic_version: str) -> ConfirmToken:
        token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
        row = self.repository.conn.execute("SELECT * FROM confirm_tokens WHERE token_hash=?", (token_hash,)).fetchone()
        if not row:
            raise TokenError(ErrorCatalog.envelope("CONTRACT_CONFIRM_REQUIRED"))
        if tuple(row)[2:12] != (session_id, user_id, run_id, task_id, order_id, service, str(amount), payload_hash, topic_version, row[11]):
            raise TokenError(ErrorCatalog.envelope("AUTH_SCOPE_MISMATCH", details={"reason": "token_binding"}))
        now = datetime.now(timezone.utc)
        expires = datetime.fromisoformat(str(row[12]).replace("Z", "+00:00"))
        if str(row[13]) != "ISSUED":
            raise TokenError(ErrorCatalog.envelope("CONTRACT_CONFIRM_EXPIRED", details={"reason": "token_not_issued"}))
        if now >= expires:
            self.repository.conn.execute("UPDATE confirm_tokens SET status='EXPIRED',updated_at=? WHERE token_id=?", (_now(), row[0]))
            self._token_event_locked(run_id, session_id, task_id, "EXPIRED")
            raise TokenError(ErrorCatalog.envelope("CONTRACT_CONFIRM_EXPIRED", details={"reason": "expired"}))
        expected_bound = token_binding_hash(session_id=session_id, user_id=user_id, run_id=run_id, task_id=task_id, order_id=order_id, service=service, amount=amount, payload_hash=payload_hash, topic_version=topic_version)
        actual_bound = token_binding_hash(session_id=str(row[2]), user_id=str(row[3]), run_id=str(row[4]), task_id=str(row[5]), order_id=str(row[6]), service=str(row[7]), amount=Decimal(str(row[8])), payload_hash=str(row[9]), topic_version=str(row[10]))
        if expected_bound != actual_bound:
            raise TokenError(ErrorCatalog.envelope("AUTH_SCOPE_MISMATCH", details={"reason": "binding_hash"}))
        self.repository.conn.execute("UPDATE confirm_tokens SET status='CONSUMED',consumed_at=?,updated_at=? WHERE token_id=? AND status='ISSUED'", (now.isoformat(), _now(), row[0]))
        self._token_event_locked(run_id, session_id, task_id, "CONSUMED")
        return ConfirmToken(token_id=row[0], token_hash=row[1], session_id=row[2], user_id=row[3], run_id=row[4], task_id=row[5], order_id=row[6], service=row[7], amount=Decimal(row[8]), payload_hash=row[9], topic_version=row[10], issued_at=datetime.fromisoformat(str(row[11]).replace("Z", "+00:00")), expires_at=expires, status="CONSUMED", consumed_at=now)

    def revoke(self, token_id: str, *, run_id: str, session_id: str, task_id: str) -> None:
        try:
            self.repository.conn.execute("BEGIN IMMEDIATE")
            cur = self.repository.conn.execute("UPDATE confirm_tokens SET status='REVOKED',revoked_at=?,updated_at=? WHERE token_id=? AND status='ISSUED'", (_now(), _now(), token_id))
            if cur.rowcount != 1:
                raise TokenError(ErrorCatalog.envelope("CONTRACT_CONFIRM_EXPIRED", details={"reason": "token_not_issued"}))
            self._token_event_locked(run_id, session_id, task_id, "REVOKED")
            self.repository.conn.commit()
        except Exception:
            self.repository.conn.rollback()
            raise

    def _token_event_locked(self, run_id: str, session_id: str, task_id: str, status: str) -> None:
        row = self.repository.conn.execute("SELECT next_seq_no FROM runs WHERE run_id=?", (run_id,)).fetchone()
        parent = self.repository.conn.execute("SELECT event_id FROM trace_outbox WHERE run_id=? ORDER BY seq_no DESC LIMIT 1", (run_id,)).fetchone()
        event = build_event(run_id=run_id, session_id=session_id, event_type="TOKEN_EVENT", seq_no=int(row[0]), task_id=task_id, parent_event_id=str(parent[0]) if parent else None, payload={"status": status, "token_id_hash": "redacted"})
        self.repository._append_m2_event_locked(event)


__all__ = ["ConfirmToken", "ConfirmTokenManager", "TokenError", "token_binding_hash"]
