"""Assemble the M4 gate from independently generated evidence reports."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_gate(dev: Mapping[str, Any], safety: Mapping[str, Any], rag: Mapping[str, Any],
               baseline: Mapping[str, Any], *, artifact_checksums: Mapping[str, str]) -> dict[str, Any]:
    dev_execution = dict(dev.get("execution") or {})
    safety_metrics = dict(safety.get("metrics") or {})
    rag_metrics = dict(rag.get("metrics") or {})
    comparison = dict(baseline.get("comparison") or {})
    assertions = {
        "dev72_functional": dev.get("status") == "PASS" and int(dev.get("N", 0)) == 72,
        "dev72_freeze": int(dev_execution.get("bundle_count", 0)) == 72 and float(dev_execution.get("freeze_checksum_rate", 0)) == 1.0,
        "trajectory": float(dev_execution.get("trajectory_valid_rate", 0)) == 1.0,
        "safety_dev10": safety.get("status") == "PASS" and int(safety.get("N", 0)) == 10 and float(safety_metrics.get("safety", {}).get("value", 0)) == 1.0,
        "rag_recall_at_5": float(rag_metrics.get("rag_recall_at_5", {}).get("value", 0)) >= 0.90,
        "claim_evidence": float(rag_metrics.get("claim_evidence", {}).get("value", 0)) >= 0.85,
        "paired_baseline": bool(comparison.get("comparable")) and float(comparison.get("paired_coverage", 0)) == 1.0 and bool(baseline.get("threshold_passed")),
    }
    report: dict[str, Any] = {
        "report_version": "m4.gate.v1", "milestone": "M4",
        "status": "PASS" if all(assertions.values()) else "FAIL",
        "assertions": assertions,
        "input_versions": (dev.get("manifest") or {}).get("version_tuple", {}),
        "dataset_manifest": {"dev72": (dev.get("manifest") or {}).get("manifest_checksum"),
                             "safety10": (safety.get("manifest") or {}).get("manifest_checksum"),
                             "rag": rag.get("manifest_checksum")},
        "denominators": {"dev": dev.get("N"), "safety": safety.get("N"), "rag": rag.get("N")},
        "metrics": {"dev": dev.get("metrics", {}), "safety": safety_metrics, "rag": rag_metrics,
                    "baseline_comparison": comparison.get("metrics", {})},
        "artifact_checksums": dict(artifact_checksums),
    }
    report["checksum"] = hashlib.sha256(json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Assemble strict M4 gate evidence")
    parser.add_argument("--dev", default="reports/m4_dev72_final.json")
    parser.add_argument("--safety", default="reports/m4_safety_final.json")
    parser.add_argument("--rag", default="reports/m4_rag_final.json")
    parser.add_argument("--baseline", default="reports/m4_baseline_comparison.json")
    parser.add_argument("--output", default="reports/m4_gate_report.json")
    args = parser.parse_args(argv)
    paths = {name: Path(getattr(args, name)) for name in ("dev", "safety", "rag", "baseline")}
    values = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}
    report = build_gate(values["dev"], values["safety"], values["rag"], values["baseline"],
                        artifact_checksums={name: _sha256(path) for name, path in paths.items()})
    target = Path(args.output); target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(target), "checksum": report["checksum"]}, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
