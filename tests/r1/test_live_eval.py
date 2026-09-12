from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from eval.r1_live_eval import LiveEvalConfig, LiveEvalRunner, ManifestSafetyError, _resolve_input, _scan_bytes, load_manifest, wilson_interval
from tests.r1.test_interactive_runtime import Provider


ROOT = Path("eval/manifests")
MANIFEST = ROOT / "dev-order-live-v1.yaml"
SCHEMA = ROOT / "dev-order-live-v1.schema.json"
GOLD = ROOT / "dev-order-live-v1.gold.json"


def _executor(**kwargs):
    case = kwargs["case"]
    return {
        "status": case["expected_status"], "code": case.get("expected_code"),
        "intent": case.get("expected_intent"), "tool_path": case.get("expected_tool_path", []),
        "bundle_verified": True, "world_fingerprint": "world-fingerprint", "hard_safety": True,
        "no_fallback": True, "model_called": 1, "model_returned": 1,
        "usage_available": True, "usage_input": 10, "usage_output": 5,
    }


def _external_executor(**kwargs):
    result = _executor(**kwargs)
    if kwargs["case"]["category"] not in {"model_fault", "database_fault"}:
        result.update(external_observation=True, external_model_returned=True)
    return result


def test_manifest_rejects_heldout_and_safety_paths(tmp_path: Path):
    heldout = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    heldout["heldout"] = True
    heldout_path = tmp_path / "heldout-manifest.yaml"
    heldout_path.write_text(yaml.safe_dump(heldout), encoding="utf-8")
    with pytest.raises(ManifestSafetyError):
        load_manifest(heldout_path, SCHEMA, GOLD)
    with pytest.raises(ManifestSafetyError):
        load_manifest(tmp_path / "test20.yaml", SCHEMA, GOLD)
    with pytest.raises(ManifestSafetyError):
        load_manifest(tmp_path / "safety2.yaml", SCHEMA, GOLD)


def test_manifest_rejects_tampered_gold_contract(tmp_path: Path):
    gold = json.loads(GOLD.read_text(encoding="utf-8"))
    gold["cases"][0]["expected_status"] = "FAILED"
    tampered = tmp_path / "gold.json"
    tampered.write_text(json.dumps(gold), encoding="utf-8")
    with pytest.raises(ValueError, match="expected_status"):
        load_manifest(MANIFEST, SCHEMA, tampered)


def test_execution_plan_has_24_cases_and_expected_repetitions():
    runner = LiveEvalRunner(executor=_executor)
    plan = runner.execution_plan()
    assert len(plan) == 64
    assert len({case["case_id"] for case, _ in plan}) == 24
    assert sum(case["category"] == "model_fault" or case["category"] == "database_fault" for case, _ in plan) == 4


def test_all_pass_report_metrics_and_wilson(tmp_path: Path):
    output = tmp_path / "r1-report.json"
    report = LiveEvalRunner(executor=_executor).run(output_path=output)
    assert report["status"] == "PASS"
    assert report["execution"] == {"N_total": 24, "N_model": 20, "N_fault": 4, "observations": 64, "model_repetitions": 3, "fault_repetitions": 1, "max_workers": 2, "resumed": False}
    assert report["denominator"] == {"N_applicable": 64, "N_failed": 0, "N_blocked": 0, "N_cancelled": 0}
    assert report["metrics"]["case_pass"]["value"] == 1.0
    assert report["metrics"]["case_pass"]["wilson95"] == list(wilson_interval(64, 64))
    assert report["latency_ms"]["p50"] >= 1.0
    assert report["external_observation_count"] == 0
    assert output.is_file()


def test_failed_expected_outcome_is_in_denominator(tmp_path: Path):
    def one_failure(**kwargs):
        result = _executor(**kwargs)
        if kwargs["case"]["case_id"] == "order-valid-01":
            result["status"] = "FAILED"
            result["code"] = "MODEL_PROVIDER_ERROR"
        return result

    report = LiveEvalRunner(executor=one_failure).run(output_path=tmp_path / "report.json")
    assert report["denominator"]["N_applicable"] == 64
    assert report["denominator"]["N_failed"] == 3
    assert report["metrics"]["task_completion"]["value"] < 1.0


def test_report_is_pii_and_secret_free_and_faults_do_not_call_provider(tmp_path: Path):
    fault_calls = []

    def provider_factory():
        fault_calls.append("called")
        raise AssertionError("fault cases must not call provider")

    def executor(**kwargs):
        case = kwargs["case"]
        if case["category"] in {"model_fault", "database_fault"}:
            assert kwargs["provider_factory"] is provider_factory
        return _executor(**kwargs)

    output = tmp_path / "report.json"
    LiveEvalRunner(executor=executor, provider_factory=provider_factory, config=LiveEvalConfig(model_repetitions=1, fault_repetitions=1)).run(output_path=output)
    text = output.read_text(encoding="utf-8")
    assert "order_id" not in text and "phone_last4" not in text and "OPENAI_API_KEY" not in text and "test-secret-value" not in text
    assert fault_calls == []


def _eight_row_db(tmp_path: Path) -> Path:
    import sqlite3
    path = tmp_path / "orders.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE orders (order_id TEXT PRIMARY KEY, phone_last4 TEXT, product_name TEXT, amount REAL, order_status TEXT, pay_status TEXT, created_at TEXT, can_apply_aftersales INTEGER)")
    for index in range(8):
        phone = "1234" if index == 0 else f"12{index:02d}"
        conn.execute("INSERT INTO orders VALUES (?,?,?,?,?,?,?,?)", (f"99000000000{index+1}", phone, "fixture", 1.0, "PAID", "PAID", "2026-01-01T00:00:00Z", 1))
    conn.commit(); conn.close()
    return path


def test_default_executor_uses_real_runtime_for_success_and_fault_trace(tmp_path: Path):
    db = _eight_row_db(tmp_path)
    runner = LiveEvalRunner(db_path=db, provider_factory=lambda: Provider(), artifact_root=tmp_path / "artifacts")
    valid_case = next(case for case in runner.loaded.manifest["cases"] if case["category"] == "valid_order")
    observation = runner._execute(valid_case, 1)
    assert observation["status"] == "SUCCEEDED" and observation["tool_path"] == ["order/get_info@v1"]
    fault_case = next(case for case in runner.loaded.manifest["cases"] if case["category"] == "model_fault")
    fault_observation = runner._execute(fault_case, 1)
    assert fault_observation["code"] == "MODEL_SCHEMA_INVALID" and fault_observation["tool_path"] == []
    assert fault_observation["bundle_verified"] is True


def test_resume_uses_atomic_checkpoints_and_verify_failure_blocks(tmp_path: Path):
    artifact_root = tmp_path / "artifacts"
    first = LiveEvalRunner(executor=_executor, artifact_root=artifact_root, max_workers=2).run(output_path=tmp_path / "first.json")
    assert first["status"] == "PASS"

    def should_not_execute(**_kwargs):
        raise AssertionError("resume must use checkpoints")

    resumed = LiveEvalRunner(executor=should_not_execute, artifact_root=artifact_root, resume=True).run(output_path=tmp_path / "resumed.json")
    assert resumed["status"] == "PASS" and resumed["execution"]["resumed"] is True

    def verify_failure(**kwargs):
        row = _executor(**kwargs)
        row["bundle_verified"] = False
        row["hard_safety"] = False
        return row

    failed = LiveEvalRunner(executor=verify_failure, artifact_root=tmp_path / "failed").run()
    assert failed["status"] == "FAIL" and failed["gate"]["trace_freeze"] is False and failed["gate"]["hard_safety"] is False


def test_prompt_variants_and_fixture_resolution_are_distinct_and_grounded(tmp_path: Path):
    db = _eight_row_db(tmp_path)
    prompts = [_resolve_input(case, db)["prompt"] for case in LiveEvalRunner().loaded.manifest["cases"] if case["category"] == "missing_entity"]
    assert any("手机号后四位是" in prompt for prompt in prompts)
    assert any("订单" in prompt and "手机号后四位暂时无法提供" in prompt for prompt in prompts)
    assert any("手机号后四位是" not in prompt and "订单号" not in prompt for prompt in prompts)
    cases = LiveEvalRunner().loaded.manifest["cases"]
    missing = [_resolve_input(case, db)["order_id"] for case in cases if case["category"] == "nonexistent_order"]
    unauthorized = [_resolve_input(case, db)["order_id"] for case in cases if case["category"] == "unauthorized_order"]
    actual = {order_id for order_id, _phone in __import__("sqlite3").connect(db).execute("SELECT order_id, phone_last4 FROM orders")}
    assert len(set(missing)) == 4 and len(set(unauthorized)) == 4 and not actual.intersection(missing)


def test_security_gate_with_real_sensitive_artifact_withholds_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("R1_TEST_API_KEY", "test-secret-value")
    def executor(**kwargs):
        Path(kwargs["artifact_root"], "unsafe.txt").write_text("test-secret-value", encoding="utf-8")
        return _executor(**kwargs)
    output = tmp_path / "report.json"
    report = LiveEvalRunner(executor=executor, artifact_root=tmp_path / "artifacts").run(output_path=output)
    assert report["status"] == "FAIL" and report["hard_gates"]["secret_pii_free_report"] is False
    assert "test-secret-value" not in output.read_text(encoding="utf-8")


def test_phone_tokens_do_not_match_hashes_but_strict_credentials_do():
    assert _scan_bytes(b"sha256=abc1234def5678", phone_tokens=("1234",))["phone_token_hits"] == 0
    assert _scan_bytes(b"trace_id=abcdef12-3456-7890-abcd-ef1234567890", phone_tokens=("3456",))["phone_token_hits"] == 0
    assert _scan_bytes(b'{"duration_ms":1234,"count":1234}', phone_tokens=("1234",))["phone_token_hits"] == 0
    assert _scan_bytes(b'{"phone_last4":"1234"}', phone_tokens=("1234",))["phone_token_hits"] == 1
    assert _scan_bytes(b'{"answer":"phone 1234","payload":{"phone_last4":"1234"}}', phone_tokens=("1234",))["phone_token_hits"] == 1
    assert _scan_bytes(b"sqlite text phone_last4=1234", phone_tokens=("1234",))["phone_token_hits"] == 1
    assert _scan_bytes(b"credential=1234", strict_values=("1234",))["exact_value_hits"] == 1


def test_phone_token_scanner_reads_sqlite_text_fields(tmp_path: Path):
    db = _eight_row_db(tmp_path)
    scan = _scan_bytes(db.read_bytes(), phone_tokens=("1234",))
    assert scan["phone_token_hits"] == 1


def test_checkpoint_provenance_mismatch_reruns_in_new_namespace(tmp_path: Path):
    root = tmp_path / "artifacts"
    first = LiveEvalRunner(executor=_executor, artifact_root=root).run()
    checkpoint = sorted((root / "checkpoints").glob("**/*.json"))[1]
    envelope = json.loads(checkpoint.read_text(encoding="utf-8"))
    envelope["provenance"]["config_hash"] = "tampered"
    checkpoint.write_text(json.dumps(envelope), encoding="utf-8")
    calls = []
    def rerun(**kwargs):
        calls.append(kwargs["case"]["case_id"])
        return _executor(**kwargs)
    second_runner = LiveEvalRunner(executor=rerun, artifact_root=root, resume=True)
    second = second_runner.run()
    assert first["status"] == "PASS" and second["status"] == "PASS" and calls and second_runner.run_namespace != root
    resumed_calls = []
    third_runner = LiveEvalRunner(executor=lambda **kwargs: resumed_calls.append(kwargs["case"]["case_id"]) or _executor(**kwargs), artifact_root=root, resume=True)
    third = third_runner.run()
    assert third["status"] == "PASS" and third_runner.run_namespace == second_runner.run_namespace and resumed_calls == []


def test_macro_is_per_case_and_external_counts_exclude_faults(tmp_path: Path):
    def one_case_failure(**kwargs):
        result = _external_executor(**kwargs)
        if kwargs["case"]["case_id"] == "order-valid-01":
            result.update(status="FAILED", code="MODEL_PROVIDER_ERROR")
        return result
    report = LiveEvalRunner(executor=one_case_failure, artifact_root=tmp_path / "artifacts").run()
    assert report["macro"]["case_pass"] != report["metrics"]["case_pass"]["value"]
    assert report["external_observation_count"] == 60
    assert report["external_returned_call_count"] == 60
    assert report["external_observations_with_return_count"] == 60
    assert all(row["latency_ms"] >= 1 for row in report["rows"])
