import json
import shutil
from pathlib import Path

from agent.storage.preflight import inspect_database, run_postflight, run_preflight


def test_preflight_and_postflight_are_metadata_only(tmp_path: Path) -> None:
    source = Path(__file__).parents[2] / "ecommerce.db"
    isolated = tmp_path / "isolated.db"
    shutil.copyfile(source, isolated)
    observed = inspect_database(isolated)
    manifest = tmp_path / "schema.json"
    manifest.write_text(json.dumps({"schema_fingerprint": observed["schema_fingerprint"], "counts": observed["counts"]}), encoding="utf-8")
    preflight = run_preflight(isolated, manifest)
    postflight = run_postflight(isolated, manifest)
    assert preflight["status"] == "pass"
    assert postflight["status"] == "pass"
    assert preflight["input"]["counts"] == {"orders": 20, "aftersales_tickets": 12}


def test_preflight_rejects_count_mismatch(tmp_path: Path) -> None:
    source = Path(__file__).parents[2] / "ecommerce.db"
    observed = inspect_database(source)
    report = run_preflight(source, expected_counts={"orders": observed["counts"]["orders"] + 1})
    assert report["status"] == "fail"
    assert "count_mismatch:orders" in report["errors"]
