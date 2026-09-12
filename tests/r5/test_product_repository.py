"""R5-A product repository and product/read@v1 callable tests (ADR-0002 §1)."""
from __future__ import annotations

from pathlib import Path

import pytest

from agent.m2_context import InvocationContext
from agent.r5_contracts import R5_CATALOG_VERSION, R5ToolSpec, product_fact_from_row
from agent.r5_product_repository import (
    R5_PRODUCT_SCHEMA_VERSION,
    R5ProductRepository,
    build_product_read_callable,
    initialize_product_db,
    seed_products,
)


@pytest.fixture()
def catalog_path(tmp_path: Path) -> Path:
    rows = [
        {"sku": "SKU-TEST-1", "name": "测试商品", "category": "测试", "attributes": {"category": "测试"}, "price": "199.00", "currency": "CNY", "stock": 8},
        {"sku": "SKU-TEST-2", "name": "无货商品", "price": "59.00", "stock": 0, "listing_status": "SOLD_OUT"},
        {"sku": "SKU-TEST-3", "name": "缺价商品", "price": None, "stock": 5},
        {"sku": "SKU-TEST-4", "name": "缺库存商品", "price": "10.00", "stock": None},
    ]
    links = [{"order_id": "20260101001", "sku": "SKU-TEST-1", "product_name": "测试商品"}]
    return seed_products(tmp_path / "catalog.db", rows, dataset_version="test-v1", order_sku_rows=links)


def test_schema_and_metadata(catalog_path: Path) -> None:
    repo = R5ProductRepository(catalog_path)
    meta = repo.metadata()
    assert meta["schema_version"] == R5_PRODUCT_SCHEMA_VERSION
    assert meta["catalog_version"] == R5_CATALOG_VERSION
    assert meta["source_version"] == "self_built_product_catalog.v1"


def test_exact_sku_read_returns_versioned_fact(catalog_path: Path) -> None:
    fact = R5ProductRepository(catalog_path).fact("SKU-TEST-1")
    assert fact is not None
    assert fact["sku"] == "SKU-TEST-1"
    assert fact["price"] == "199.00"
    assert fact["currency"] == "CNY"
    assert fact["stock"] == 8
    assert fact["version"] == R5_CATALOG_VERSION
    assert fact["source"] == "simulator"
    assert fact["data_quality"] == "FRESH"
    assert len(fact["snapshot_hash"]) == 64


def test_missing_price_and_stock_are_missing_not_defaulted(catalog_path: Path) -> None:
    repo = R5ProductRepository(catalog_path)
    no_price = product_fact_from_row(repo.get("SKU-TEST-3"))
    assert no_price.price is None and no_price.data_quality == "MISSING"
    no_stock = product_fact_from_row(repo.get("SKU-TEST-4"))
    assert no_stock.stock is None and no_stock.data_quality == "MISSING"


def test_callable_unknown_sku_maps_to_data_missing(catalog_path: Path) -> None:
    read = build_product_read_callable(catalog_path)
    result = read(sku="SKU-DOES-NOT-EXIST")
    assert result["success"] is False
    assert result["code"] == "DATA_MISSING"
    assert result["data"] is None


def test_callable_success_and_missing_quality(catalog_path: Path) -> None:
    read = build_product_read_callable(catalog_path)
    ok = read(sku="SKU-TEST-1")
    assert ok["success"] is True and ok["data"]["stock"] == 8
    missing = read(sku="SKU-TEST-3")
    assert missing["success"] is True and missing["data"]["price"] is None
    assert missing["data"]["data_quality"] == "MISSING"


def test_order_sku_map_lookup(catalog_path: Path) -> None:
    repo = R5ProductRepository(catalog_path)
    link = repo.sku_for_order("20260101001")
    assert link == {"order_id": "20260101001", "sku": "SKU-TEST-1", "product_name": "测试商品", "mapping_version": "order_sku_map.v1"}
    assert repo.sku_for_order("no-such-order") is None


def test_repository_is_read_only(catalog_path: Path) -> None:
    repo = R5ProductRepository(catalog_path)
    with pytest.raises(sqlite3_error()):
        _write_attempt(repo)


def sqlite3_error() -> type:
    import sqlite3

    return sqlite3.OperationalError


def _write_attempt(repo: R5ProductRepository) -> None:
    conn = repo._connect()
    try:
        conn.execute("DELETE FROM products")
    finally:
        conn.close()


def test_initialize_refuses_existing_without_replace(tmp_path: Path) -> None:
    target = tmp_path / "x.db"
    initialize_product_db(target)
    with pytest.raises(FileExistsError):
        initialize_product_db(target)


def test_r5_tool_spec_extends_without_weakening() -> None:
    spec = R5ToolSpec(
        tool_ref="aftersales/eligibility@v1",
        capability_ref="aftersales/eligibility@v1",
        owner="aftersales-agent",
        args_schema="aftersales.eligibility.v1",
        result_schema="eligibility.result.v1",
        risk="READ",
        implementation_mode="SIMULATED",
        side_effect="READ_ONLY",
        timeout_ms=5000,
        allowed_error_codes=("AUTH_IDENTITY_MISMATCH", "ORDER_NOT_FOUND", "DATA_MISSING", "ELIGIBILITY_DENIED", "ELIGIBILITY_MANUAL"),
        callable=lambda **kw: {"success": True, "code": "OK", "data": None},
    )
    spec.validate_candidate_args({"order_id": "o1", "phone_last4": "1234", "service": "refund"})
    with pytest.raises(ValueError):
        spec.validate_candidate_args({"order_id": "o1", "phone_last4": "1234"})
    with pytest.raises(ValueError):
        spec.validate_candidate_args({"sku": "s1"})


def test_reserved_args_rejected_via_inherited_check(tmp_path: Path) -> None:
    rows = [{"sku": "SKU-R", "name": "r", "price": "1.00", "stock": 1}]
    path = seed_products(tmp_path / "r.db", rows)
    read = build_product_read_callable(path)
    assert read(sku="SKU-R")["success"] is True
    base = R5ToolSpec(
        tool_ref="product/get@v1",
        capability_ref="product/read@v1",
        owner="product-agent",
        args_schema="product.read.v1",
        result_schema="product.result.v1",
        risk="READ",
        implementation_mode="SIMULATED",
        side_effect="READ_ONLY",
        timeout_ms=5000,
        allowed_error_codes=("DATA_MISSING",),
        callable=read,
    )
    with pytest.raises(ValueError):
        base.validate_candidate_args({"sku": "SKU-R", "user_id": "attacker"})


def test_invocation_context_acceptance(catalog_path: Path) -> None:
    from datetime import datetime, timedelta, timezone

    read = build_product_read_callable(catalog_path)
    ctx = InvocationContext(
        session_id="s", user_id="u", run_id="r", plan_revision_id="p", task_id="t", attempt_id="a",
        agent_ref="product-agent@v1", auth_scope="product/read@v1", idempotency_key="k",
        deadline=datetime.now(timezone.utc) + timedelta(seconds=5), config_version="r5.v1",
        registry_version="r5.registry.v1", dataset_version="r5-demo-v1", trace_id="tr",
    )
    assert ctx.agent_ref == "product-agent@v1"
    assert read(sku="SKU-TEST-1")["data"]["fact_id"] == "product_SKU-TEST-1"
