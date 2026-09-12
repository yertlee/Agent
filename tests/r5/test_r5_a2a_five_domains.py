"""R5-A3 five-domain A2A integration test.

Each of the five domains (Product, Order, Logistics, Policy, AfterSales) must
be reachable through the same A2A message contract and return a canonical
Result.  Order/logistics/policy read the existing read-only order database;
product/after-sales read isolated R5 demo copies.  No test writes user data.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from agent.domain.objects import sha256_json
from agent.r2_logistics_repository import seed_from_orders
from agent.r5_a2a_runtime import R5A2ARuntime, build_r5_registry
from agent.r5_aftersales_repository import seed_aftersales_demo
from agent.r5_product_repository import seed_products


ROOT = Path(__file__).parents[2]
ORDER_DB = ROOT / "ecommerce.db"


def _order_row() -> tuple[str, str, str, str]:
    with sqlite3.connect(f"file:{ORDER_DB.resolve()}?mode=ro", uri=True) as conn:
        row = conn.execute(
            "SELECT order_id, phone_last4, carrier_code, tracking_no, product_name "
            "FROM orders WHERE carrier_code <> '' AND tracking_no <> '' "
            "ORDER BY order_id LIMIT 1"
        ).fetchone()
    if row is None:
        pytest.skip("read-only ecommerce.db has no order with tracking")
    return tuple(str(value) for value in row)


def _policy_setup() -> tuple[dict, dict]:
    version_tuple = ["r5.manifest.v1", "r5.test.corpus.v1"] + [f"v{i}" for i in range(2, 14)] + ["a" * 64, "r5.runtime.v1", "b" * 64, "c" * 64, "d" * 64, "e" * 64]
    source_version = sha256_json(version_tuple)
    authority = {
        "source": "r3.5.project-authored-kb",
        "source_version": source_version,
        "version_tuple": version_tuple,
        "strategy_checksum": "d" * 64,
        "evidence": [{"evidence_id": "ev-1", "source_id": "source-1", "version": "v1", "chunk_id": "chunk-1", "text_hash": "f" * 64, "locator": "chunk-1"}],
    }
    reader = lambda query: {  # noqa: E731
        "query": query,
        "status": "ANSWERED",
        "source": "r3.5.project-authored-kb",
        "source_version": source_version,
        "source_version_tuple": version_tuple,
        "evidence": [{"evidence_id": "ev-1", "source_id": "source-1", "version": "v1", "chunk_id": "chunk-1", "text_hash": "f" * 64, "locator": "chunk-1"}],
        "claims": [{"claim_id": "claim-1", "text": "verified", "evidence_ids": ["ev-1"]}],
    }
    return reader, authority


@pytest.fixture()
def r5_runtime(tmp_path: Path):
    order_id, phone_last4, _carrier, _tracking, product_name = _order_row()
    product_db = seed_products(
        tmp_path / "product.db",
        [{"sku": "SKU-T-1", "name": product_name, "price": "199.00", "stock": 5}],
        order_sku_rows=[{"order_id": order_id, "sku": "SKU-T-1", "product_name": product_name}],
    )
    aftersales_db = seed_aftersales_demo(
        tmp_path / "aftersales.db",
        orders_rows=[{"order_id": order_id, "phone_last4": phone_last4, "product_name": product_name, "amount": "199.00", "order_status": "DELIVERED"}],
        case_rows=[],
    )
    logistics_db = seed_from_orders(ORDER_DB, tmp_path / "logistics.db")
    reader, authority = _policy_setup()
    runtime = R5A2ARuntime(
        db_path=ORDER_DB,
        product_db_path=product_db,
        aftersales_db_path=aftersales_db,
        logistics_db_path=logistics_db,
        ledger_path=tmp_path / "ledger.sqlite",
        policy_reader=reader,
        policy_authority=authority,
    )
    try:
        yield runtime, order_id, phone_last4, product_name
    finally:
        runtime.close()


def _dispatch(runtime, run_id: str, task_id: str, capability: str, payload: dict):
    request = runtime.build_request(run_id=run_id, plan_revision_id=f"plan-{run_id}", task_id=task_id, capability_ref=capability, payload=payload)
    return runtime.dispatch(request)


def test_registry_admits_five_read_domains(r5_runtime) -> None:
    runtime, *_ = r5_runtime
    refs = runtime.registry.refs()
    for tool_ref in ("product/get@v1", "order/get_info@v1", "logistics/query@v1", "policy/search@v1", "aftersales/query@v1", "aftersales/eligibility@v1"):
        assert tool_ref in refs
    for ref in refs:
        spec = runtime.registry.get(ref)
        assert spec.side_effect == "READ_ONLY" and spec.risk == "READ"


def test_five_domains_each_return_canonical_result(r5_runtime) -> None:
    runtime, order_id, phone_last4, product_name = r5_runtime
    cases = [
        ("order", "order/read@v1", {"order_id": order_id, "phone_last4": phone_last4}),
        ("product", "product/read@v1", {"sku": "SKU-T-1"}),
        ("aftersales", "aftersales/read@v1", {"order_id": order_id, "phone_last4": phone_last4}),
        ("eligibility", "aftersales/eligibility@v1", {"order_id": order_id, "phone_last4": phone_last4, "service": "refund"}),
        ("policy", "policy/read@v1", {"query": "退货政策"}),
    ]
    for task_id, capability, payload in cases:
        result = _dispatch(runtime, "r-five", task_id, capability, payload)
        assert result.status == "SUCCEEDED", (capability, result.error_code)
        assert result.canonical_result is not None
        assert result.specialist_result is not None and result.specialist_result.ok
        assert result.canonical_result.payload["source_version"]


def test_product_and_eligibility_payload_shape(r5_runtime) -> None:
    runtime, order_id, phone_last4, _ = r5_runtime
    product = _dispatch(runtime, "r-shape-p", "product", "product/read@v1", {"sku": "SKU-T-1"})
    assert product.specialist_result.payload["sku"] == "SKU-T-1"
    assert product.specialist_result.payload["source"] == "simulator"
    eligibility = _dispatch(runtime, "r-shape-e", "eligibility", "aftersales/eligibility@v1", {"order_id": order_id, "phone_last4": phone_last4, "service": "refund"})
    assert eligibility.specialist_result.payload["decision"] == "ALLOW"
    assert eligibility.specialist_result.payload["entity_id"] == f"{order_id}:refund"


def test_unknown_product_is_a_domain_failure_not_a_crash(r5_runtime) -> None:
    runtime, *_ = r5_runtime
    result = _dispatch(runtime, "r-unknown", "product", "product/read@v1", {"sku": "SKU-NOPE"})
    assert result.status == "FAILED"
    assert result.error_code == "DATA_MISSING"
    assert result.canonical_result is not None and result.canonical_result.status.value == "FAILED"


def test_wrong_owner_fails_closed(r5_runtime) -> None:
    runtime, order_id, _phone, _ = r5_runtime
    result = _dispatch(runtime, "r-owner", "aftersales", "aftersales/read@v1", {"order_id": order_id, "phone_last4": "0000"})
    assert result.status == "FAILED"
    assert result.error_code == "AUTH_IDENTITY_MISMATCH"


def test_disabled_capability_stays_in_lifecycle(r5_runtime) -> None:
    runtime, order_id, phone_last4, _ = r5_runtime
    runtime.disabled_capabilities = frozenset({"product/read@v1"})
    result = _dispatch(runtime, "r-disabled", "product", "product/read@v1", {"sku": "SKU-T-1"})
    assert result.status == "FAILED"
    assert result.error_code == "SPECIALIST_DISABLED"
    assert runtime.physical_call_counts["product/read@v1"] == 0


def test_build_r5_registry_standalone(tmp_path: Path) -> None:
    order_id, phone_last4, _c, _t, product_name = _order_row()
    product_db = seed_products(tmp_path / "p.db", [{"sku": "S1", "name": product_name, "price": "1.00", "stock": 1}])
    aftersales_db = seed_aftersales_demo(tmp_path / "a.db", orders_rows=[{"order_id": order_id, "phone_last4": phone_last4}], case_rows=[])
    registry = build_r5_registry(db_path=ORDER_DB, product_db_path=product_db, aftersales_db_path=aftersales_db)
    assert "product/get@v1" in registry.refs()
    assert registry.get("product/get@v1").capability_ref == "product/read@v1"
    assert registry.get("aftersales/eligibility@v1").capability_ref == "aftersales/eligibility@v1"
