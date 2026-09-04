"""Final M5 gate composition; absent heldout evidence stays BLOCKED."""
from __future__ import annotations
from typing import Any, Mapping
import argparse
import json
from pathlib import Path

M5_THRESHOLDS = {"task_completion": .85, "case_pass": .80, "world_fingerprint": .90, "safety": 1.0}

def build_gate(test_report: Mapping[str, Any], safety_report: Mapping[str, Any], *, evidence_verified: bool, demo_report: Mapping[str, Any] | None = None) -> dict[str, Any]:
    demo_ok = demo_report is None or demo_report.get("status") == "PASS"
    if test_report.get("status") != "PASS" or safety_report.get("status") != "PASS" or not evidence_verified or not demo_ok:
        return {"gate": "M5", "status": "BLOCKED", "reason": "heldout or safety evidence is unavailable or unverified", "thresholds": M5_THRESHOLDS}
    metrics = {k: float((test_report.get("metrics", {}).get(k) or {}).get("value", 0)) for k in ("task_completion", "case_pass", "world_fingerprint")}
    metrics["safety"] = float((safety_report.get("metrics", {}).get("safety") or {}).get("value", 0))
    failures = [k for k, threshold in M5_THRESHOLDS.items() if metrics.get(k, 0) < threshold]
    return {"gate": "M5", "status": "FAIL" if failures else "PASS", "metrics": metrics, "failed_thresholds": failures, "thresholds": M5_THRESHOLDS}

__all__ = ["M5_THRESHOLDS", "build_gate"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M5 gate")
    parser.add_argument("--test-report", required=True)
    parser.add_argument("--safety-report", required=True)
    parser.add_argument("--evidence-verified", action="store_true")
    parser.add_argument("--demo-report")
    parser.add_argument("--output", default="reports/m5_gate.json")
    args = parser.parse_args()
    test = json.loads(Path(args.test_report).read_text(encoding="utf-8"))
    safety = json.loads(Path(args.safety_report).read_text(encoding="utf-8"))
    demo = json.loads(Path(args.demo_report).read_text(encoding="utf-8")) if args.demo_report else None
    gate = build_gate(test, safety, evidence_verified=args.evidence_verified, demo_report=demo)
    target = Path(args.output); target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(gate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": gate["status"], "report": str(target)}, ensure_ascii=False))
    raise SystemExit(0 if gate["status"] == "PASS" else 1)
