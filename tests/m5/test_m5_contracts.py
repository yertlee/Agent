from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agent.api.security import project_trace
from agent.runtime_port import RuntimePort
from eval.demos import build_demo_manifest, run_demos
from eval.evidence_bundle import build_bundle, verify_bundle
from eval.m5_eval import evaluate_heldout
from eval.m5_gate import build_gate
from app.fastapi_app import ChatRequest, create_app


@pytest.fixture()
def client(tmp_path):
    return TestClient(create_app(runtime=RuntimePort(artifact_root=tmp_path / "runtime")))


def test_chat_auth_schema_and_owner_ref(client):
    assert client.post("/api/chat", json={"message": "查询商品信息"}).status_code == 401
    invalid = client.post("/api/chat", json={"message": "x", "owner": "raw"}, headers={"Authorization": "Bearer alice"})
    assert invalid.status_code == 422 and invalid.json()["code"] == "VALIDATION_ERROR"
    first = client.post("/api/chat", json={"message": "查询商品信息"}, headers={"Authorization": "Bearer alice"})
    assert first.status_code == 200
    run_id = first.json()["run_id"]
    metadata = Path(client.app.state.runtime._metadata(run_id)).read_text(encoding="utf-8")
    assert "alice" not in metadata and "Bearer" not in metadata
    assert client.get(f"/api/trace/{run_id}", headers={"Authorization": "Bearer bob"}).status_code == 403
    # The same credential is stable even though its owner_ref is opaque.
    assert client.get(f"/api/trace/{run_id}", headers={"Authorization": "Bearer alice"}).status_code == 200
    assert client.post("/api/chat", json={"message": "x", "cancel": True}, headers={"Authorization": "Bearer alice"}).status_code == 409
    assert client.post("/api/chat", json={"message": "x", "wait": "none"}, headers={"Authorization": "Bearer alice"}).status_code == 409
    assert client.post("/api/chat", json={"message": "x", "cancel": True, "run_id": run_id}, headers={"Authorization": "Bearer bob"}).status_code == 403


def test_path_id_readonly_and_report_read_model(client):
    response = client.post("/api/chat", json={"message": "查询商品信息"}, headers={"X-User-Id": "u"})
    run_id = response.json()["run_id"]
    assert client.get(f"/api/trace/{run_id}", headers={"X-User-Id": "u"}).status_code == 200
    assert client.post(f"/api/trace/{run_id}", headers={"X-User-Id": "u"}).status_code == 405
    assert client.post(f"/api/report/{run_id}", headers={"X-User-Id": "u"}).status_code == 405
    report = client.get(f"/api/report/{run_id}", headers={"X-User-Id": "u"})
    assert report.status_code == 200 and report.json()["status"] == "VERIFIED"
    assert set(report.json()["read_model"]) == {"Run", "Plan", "Task", "Attempt", "Tool", "Result", "Review", "Evidence"}
    assert client.get("/api/trace/..%2F..%2Fetc", headers={"X-User-Id": "u"}).status_code in {400, 404, 422}


def test_checksum_tamper_and_html_escaping(client):
    response = client.post("/api/chat", json={"message": "查询商品信息"}, headers={"Authorization": "Bearer x"})
    run_id = response.json()["run_id"]
    page = client.get(f"/api/report/{run_id}/html", headers={"Authorization": "Bearer x"})
    assert page.status_code == 200 and all(label in page.text for label in ("Run", "Plan", "Task", "Attempt", "Tool", "Result", "Review", "Evidence"))
    assert "<script>" not in page.text
    runtime = client.app.state.runtime
    trace_path = Path(runtime._metadata(run_id)).parent / "trace.jsonl"
    trace_path.write_text(trace_path.read_text(encoding="utf-8") + "tamper\n", encoding="utf-8")
    assert client.get(f"/api/trace/{run_id}", headers={"Authorization": "Bearer x"}).status_code == 409


def test_projection_does_not_expose_pii_or_token_and_html_uses_projection():
    projected = project_trace([{"event_type": "X", "payload": {"phone": "13800138000", "token": "secret", "value": "<script>alert(1)</script>"}}])
    text = json.dumps(projected, ensure_ascii=False)
    assert "13800138000" not in text and "secret" not in text
    assert "<script>" in text  # projection preserves safe content; HTML layer escapes it


def test_five_demos_execute_runtime_and_verify_real_bundles(tmp_path):
    report = run_demos(build_demo_manifest(), artifact_root=tmp_path / "demos")
    assert report["case_count"] == 5
    assert all(row.get("bundle_verified") is True for row in report["outputs"])
    assert all(row.get("artifact_checksums") for row in report["outputs"])
    assert all(row["status"] == "PASS" and row["assertion_status"] == "PASS" for row in report["outputs"])
    paths = {row["scenario_id"]: row["observed_tool_path"] for row in report["outputs"]}
    assert paths["demo-cross-domain"] == ["order/get_info@v1", "product/get@v1"]
    assert paths["demo-policy-conflict"] == ["human/handoff@v1"]
    assert paths["demo-unauthorized"] == ["human/handoff@v1"]


def test_demo_manifest_tamper_is_rejected(tmp_path):
    manifest = build_demo_manifest()
    manifest["cases"][0]["turns"] = ["tampered"]
    with pytest.raises(ValueError, match="manifest checksum"):
        run_demos(manifest, artifact_root=tmp_path / "demos")


def test_evidence_bundle_tamper(tmp_path):
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}\n", encoding="utf-8")
    build_bundle(tmp_path / "bundle", artifacts=[artifact])
    manifest_value = json.loads((tmp_path / "bundle" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest_value["artifacts"][0]["path"].startswith("artifacts/")
    assert verify_bundle(tmp_path / "bundle")["ok"] is True
    (tmp_path / "bundle" / "artifacts" / "000-artifact.json").write_text("tampered\n", encoding="utf-8")
    assert verify_bundle(tmp_path / "bundle")["ok"] is False
    manifest = tmp_path / "bundle" / "manifest.json"
    value = json.loads(manifest.read_text(encoding="utf-8")); value["status"] = "PASS"
    manifest.write_text(json.dumps(value), encoding="utf-8")
    assert verify_bundle(tmp_path / "bundle")["ok"] is False


def test_heldout_sealed_metadata_is_blocked(tmp_path):
    report = evaluate_heldout(Path("eval/manifests/test20.yaml"))
    assert report["status"] == "BLOCKED" and report["test_execution"] == "NOT_RUN"
    assert report["status_counts"] == {"NOT_RUN": 20}


def test_m5_gate_uses_independent_safety_report():
    metric = lambda value: {"value": value}
    test = {"status": "PASS", "metrics": {"task_completion": metric(.9), "case_pass": metric(.8), "world_fingerprint": metric(.9)}}
    safety = {"status": "PASS", "metrics": {"safety": metric(1.0)}}
    assert build_gate(test, safety, evidence_verified=True, demo_report={"status": "PASS"})["status"] == "PASS"
    safety["metrics"]["safety"] = metric(0.5)
    assert build_gate(test, safety, evidence_verified=True, demo_report={"status": "PASS"})["status"] == "FAIL"
