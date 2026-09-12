"""No-key smoke tests for the self-contained R5 synthetic fixture."""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "scripts" / "r5_seed_synthetic_data.py"
TARGET_NAMES = (
    "ecommerce.db",
    "r5_demo_v1.db",
    "r5_aftersales_demo_v1.db",
    "r5_logistics_demo_v1.db",
    "r5_synthetic_manifest.json",
)


def _run_seed(output_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--output-dir", str(output_dir)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_seed_creates_readable_order_r5_and_logistics_databases(tmp_path: Path) -> None:
    output_dir = tmp_path / "fixture"
    completed = _run_seed(output_dir)
    assert completed.returncode == 0, completed.stderr

    manifest_path = output_dir / "r5_synthetic_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["manifest_version"] == "r5.synthetic.manifest.v1"
    assert manifest["dataset_version"] == "r5-synthetic-v1"
    assert manifest["source"]["kind"] == "self_built_synthetic_fixture"
    assert manifest["privacy"]["contains_user_provided_data"] is False
    assert manifest["privacy"]["contains_external_platform_data"] is False
    assert len(manifest["files"]) == 4
    assert all((output_dir / name).is_file() for name in TARGET_NAMES)

    from agent.r2_logistics_repository import R2LogisticsRepository, order_ref_hash
    from agent.r5_aftersales_repository import build_aftersales_read_callable, build_eligibility_callable
    from agent.r5_product_repository import build_product_read_callable
    from agent.storage.repository import SQLiteOrderRepository

    with SQLiteOrderRepository(str(output_dir / "ecommerce.db")) as orders:
        order = orders.get_order_for_owner("20260320001", "1234")
    assert order is not None
    assert order["order_status"] == "DELIVERED"
    assert order["tracking_no"] == "7609205232746"

    product = build_product_read_callable(output_dir / "r5_demo_v1.db")(sku="SKU-1001")
    assert product["success"] is True
    assert product["data"]["source"] == "simulator"
    assert product["data"]["price"] == "199.00"

    aftersales = build_aftersales_read_callable(output_dir / "r5_aftersales_demo_v1.db")
    status = aftersales(order_id="20260320001", phone_last4="1234")
    assert status["success"] is True
    assert status["data"]["case_summary"] == "ACTIVE_CASE"
    eligibility = build_eligibility_callable(output_dir / "r5_aftersales_demo_v1.db")
    assert eligibility(order_id="20260320001", phone_last4="1234", service="exchange")["data"]["decision"] == "ALLOW"

    logistics = R2LogisticsRepository(output_dir / "r5_logistics_demo_v1.db").read(
        order_ref=order_ref_hash("20260320001"),
        carrier_code="yuantong",
        tracking_no="7609205232746",
    )
    assert logistics is not None
    assert logistics["delivery_state"] == "DELIVERED"
    assert logistics["data_quality"] == "FRESH"
    assert [event["event_code"] for event in logistics["events"]] == ["PICKED_UP", "DELIVERED"]


def test_seed_is_byte_reproducible_in_two_isolated_directories(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_result = _run_seed(first)
    second_result = _run_seed(second)
    assert first_result.returncode == 0, first_result.stderr
    assert second_result.returncode == 0, second_result.stderr

    for name in TARGET_NAMES:
        assert _sha256(first / name) == _sha256(second / name), name
    assert (first / "r5_synthetic_manifest.json").read_bytes() == (second / "r5_synthetic_manifest.json").read_bytes()


def test_generated_fixture_runs_a_four_domain_r5_plan(tmp_path: Path) -> None:
    output_dir = tmp_path / "fixture"
    completed = _run_seed(output_dir)
    assert completed.returncode == 0, completed.stderr

    from eval.r5_plan_executor import R5PlanExecutor

    plan = {
        "schema_version": "r5.plan.v1",
        "business_goal": "synthetic smoke",
        "nodes": (
            {
                "node_id": "product_read",
                "capability_ref": "product/read@v1",
                "args": {"sku": "SKU-1001"},
                "bindings": {},
                "failure_strategy": "FAIL_RUN",
            },
            {
                "node_id": "order_read",
                "capability_ref": "order/read@v1",
                "args": {"order_id": "20260320001"},
                "bindings": {"phone_last4": {"kind": "context", "context_key": "phone_last4"}},
                "failure_strategy": "FAIL_RUN",
            },
            {
                "node_id": "logistics_read",
                "capability_ref": "logistics/read@v1",
                "args": {},
                "bindings": {
                    "carrier_code": {"kind": "result", "source_node_id": "order_read", "path": "carrier_code"},
                    "tracking_no": {"kind": "result", "source_node_id": "order_read", "path": "tracking_no"},
                },
                "failure_strategy": "FAIL_RUN",
            },
            {
                "node_id": "aftersales_read",
                "capability_ref": "aftersales/read@v1",
                "args": {"order_id": "20260320001"},
                "bindings": {"phone_last4": {"kind": "context", "context_key": "phone_last4"}},
                "failure_strategy": "FAIL_RUN",
            },
        ),
        "edges": ({"upstream_node_id": "order_read", "downstream_node_id": "logistics_read"},),
    }
    outcome = R5PlanExecutor(
        orders_db=output_dir / "ecommerce.db",
        product_db=output_dir / "r5_demo_v1.db",
        aftersales_db=output_dir / "r5_aftersales_demo_v1.db",
        logistics_db=output_dir / "r5_logistics_demo_v1.db",
    ).execute(
        plan,
        trusted_context={"phone_last4": "1234", "user_id": "demo_user_1234"},
        allows_write=False,
    )
    assert outcome.status == "COMPLETED"
    assert outcome.terminal == "ANSWER"
    assert set(outcome.successful_reads) == {
        "product/read@v1",
        "order/read@v1",
        "logistics/read@v1",
        "aftersales/read@v1",
    }


def test_seed_refuses_to_overwrite_existing_target(tmp_path: Path) -> None:
    output_dir = tmp_path / "fixture"
    output_dir.mkdir()
    sentinel = output_dir / "ecommerce.db"
    sentinel.write_bytes(b"caller-owned sentinel")

    completed = _run_seed(output_dir)

    assert completed.returncode != 0
    assert sentinel.read_bytes() == b"caller-owned sentinel"
    assert not (output_dir / "r5_demo_v1.db").exists()


def test_seed_script_has_no_source_seed_dependency() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "scripts.r5_seed_demo_data" not in imported_modules
    assert "seed_from_orders" not in called_names
    assert "OPENAI_API_KEY" not in source
