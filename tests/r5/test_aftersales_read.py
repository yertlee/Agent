"""R5-A after-sales read and eligibility callable tests (ADR-0002 §3-5)."""
from __future__ import annotations

from pathlib import Path

import pytest

from agent.r5_aftersales_repository import (
    demo_user_id,
    seed_aftersales_demo,
)


@pytest.fixture()
def demo_path(tmp_path: Path) -> Path:
    orders = [
        {"order_id": "O1", "phone_last4": "1234", "product_name": "p1", "amount": "199.00", "order_status": "已签收", "pay_status": "已支付", "shipment_status": "已签收", "created_at": "2026-03-20T00:00:00Z", "delivered_at": "2026-03-21T00:00:00Z"},
        {"order_id": "O2", "phone_last4": "5678", "product_name": "p2", "amount": "59.00", "order_status": "CANCELLED", "pay_status": "已支付", "shipment_status": "", "created_at": "2026-03-20T00:00:00Z", "delivered_at": None},
        {"order_id": "O3", "phone_last4": "9999", "product_name": "p3", "amount": "10.00", "order_status": "已签收", "pay_status": "已支付", "shipment_status": "已签收", "created_at": "2026-03-20T00:00:00Z", "delivered_at": "2026-03-21T00:00:00Z"},
    ]
    cases = [
        {"case_id": "C-ACTIVE", "order_id": "O1", "service": "refund", "status": "UNDER_REVIEW", "reason": "r1", "amount": "199.00"},
        {"case_id": "C-DONE", "order_id": "O1", "service": "return", "status": "REFUNDED", "reason": "r2", "amount": "59.00"},
        {"case_id": "C-OTHER", "order_id": "O3", "service": "exchange", "status": "CANCELLED", "reason": "r3", "amount": "10.00"},
    ]
    return seed_aftersales_demo(tmp_path / "aftersales.db", orders_rows=orders, case_rows=cases, dataset_version="test-v1")


def test_query_unknown_order(demo_path: Path) -> None:
    query = __import__("agent.r5_aftersales_repository", fromlist=["build_aftersales_read_callable"]).build_aftersales_read_callable(demo_path)
    result = query(order_id="NOPE", phone_last4="1234")
    assert result["success"] is False and result["code"] == "ORDER_NOT_FOUND"


def test_query_wrong_owner_rejected(demo_path: Path) -> None:
    from agent.r5_aftersales_repository import build_aftersales_read_callable

    query = build_aftersales_read_callable(demo_path)
    result = query(order_id="O1", phone_last4="5678")
    assert result["success"] is False and result["code"] == "AUTH_IDENTITY_MISMATCH"


def test_query_no_case(demo_path: Path) -> None:
    from agent.r5_aftersales_repository import build_aftersales_read_callable

    query = build_aftersales_read_callable(demo_path)
    result = query(order_id="O2", phone_last4="5678")
    assert result["success"] is True
    assert result["data"]["case_summary"] == "NO_CASE"
    assert result["data"]["cases"] == []


def test_query_distinguishes_active_and_terminal(demo_path: Path) -> None:
    from agent.r5_aftersales_repository import build_aftersales_read_callable

    query = build_aftersales_read_callable(demo_path)
    result = query(order_id="O1", phone_last4="1234")
    assert result["success"] is True
    assert result["data"]["case_summary"] == "ACTIVE_CASE"
    assert result["data"]["active_cases"] == [{"case_id": "C-ACTIVE", "service": "refund", "status": "UNDER_REVIEW"}]
    assert {c["status"] for c in result["data"]["cases"]} == {"UNDER_REVIEW", "REFUNDED"}


def test_query_history_only(demo_path: Path) -> None:
    from agent.r5_aftersales_repository import build_aftersales_read_callable

    query = build_aftersales_read_callable(demo_path)
    result = query(order_id="O3", phone_last4="9999")
    assert result["data"]["case_summary"] == "HISTORY_ONLY"


def test_user_id_derivation_is_deterministic() -> None:
    assert demo_user_id("1234") == "demo_user_1234"
    assert demo_user_id("1234") == demo_user_id("1234")


def test_eligibility_allow_and_identity(demo_path: Path) -> None:
    from agent.r5_aftersales_repository import build_eligibility_callable

    check = build_eligibility_callable(demo_path)
    result = check(order_id="O1", phone_last4="1234", service="refund")
    assert result["success"] is True
    assert result["data"]["decision"] == "ALLOW"
    assert result["data"]["policy_version"].startswith("policy.")
    denied_identity = check(order_id="O1", phone_last4="0000", service="refund")
    assert denied_identity["code"] == "AUTH_IDENTITY_MISMATCH"


def test_eligibility_denies_terminal_order(demo_path: Path) -> None:
    from agent.r5_aftersales_repository import build_eligibility_callable

    check = build_eligibility_callable(demo_path)
    result = check(order_id="O2", phone_last4="5678", service="refund")
    assert result["data"]["decision"] == "DENY"
    assert result["data"]["rule_id"] == "ORDER_TERMINAL"


def test_eligibility_denies_unsupported_service(demo_path: Path) -> None:
    from agent.r5_aftersales_repository import build_eligibility_callable

    check = build_eligibility_callable(demo_path)
    result = check(order_id="O1", phone_last4="1234", service="teleport")
    assert result["data"]["decision"] == "DENY"
    assert result["data"]["rule_id"] == "SERVICE_UNSUPPORTED"


def test_eligibility_unknown_order(demo_path: Path) -> None:
    from agent.r5_aftersales_repository import build_eligibility_callable

    check = build_eligibility_callable(demo_path)
    result = check(order_id="NOPE", phone_last4="1234", service="refund")
    assert result["success"] is False and result["code"] == "ORDER_NOT_FOUND"


def test_seed_rejects_case_for_unknown_order(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        seed_aftersales_demo(
            tmp_path / "bad.db",
            orders_rows=[{"order_id": "O1", "phone_last4": "1234"}],
            case_rows=[{"case_id": "C1", "order_id": "GHOST", "service": "refund", "status": "REQUESTED"}],
        )
