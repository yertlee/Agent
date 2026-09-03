"""Machine-readable and human-readable M4 evaluation reports."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


M4_THRESHOLDS = {"task_completion": 0.90, "intent_accuracy": 0.90, "tool_path": 0.90,
                 "case_pass": 0.65, "world_fingerprint": 0.85, "safety": 1.0,
                 "rag_recall_at_5": 0.90, "claim_evidence": 0.85,
                 "trajectory_valid_rate": 1.0, "freeze_checksum_rate": 1.0}


class ReportGenerator:
    def __init__(self, *, report_version: str = "m4.report.v1"):
        self.report_version = report_version

    def build(self, evaluation: Mapping[str, Any], *, manifest: Mapping[str, Any] | None = None,
              gate: str = "M4") -> dict[str, Any]:
        metrics = dict(evaluation.get("metrics", {}))
        for name in ("trajectory_valid_rate", "freeze_checksum_rate"):
            if name in evaluation:
                metrics[name] = {"value": float(evaluation[name]), "N": evaluation.get("N", 0), "N_applicable": evaluation.get("N", 0),
                                 "denominator": evaluation.get("N", 0), "numerator": round(float(evaluation[name]) * float(evaluation.get("N", 0))),
                                 "coverage": 1.0, "wilson95": [0.0, 1.0]}
        thresholds: dict[str, Any] = {}
        failures: list[str] = []
        for name, threshold in M4_THRESHOLDS.items():
            if name not in metrics: continue
            value = float(metrics[name].get("value", 0.0)); passed = value >= threshold
            thresholds[name] = {"threshold": threshold, "value": value, "passed": passed}
            if not passed: failures.append(name)
        baseline = evaluation.get("baseline_metrics")
        if baseline is not None:
            baseline_check = {}
            for name, previous in baseline.items():
                if name in metrics:
                    delta = float(metrics[name].get("value", 0.0)) - float(previous)
                    baseline_check[name] = {"baseline": float(previous), "value": float(metrics[name].get("value", 0.0)), "absolute_delta": delta, "threshold": -0.03, "passed": delta >= -0.03}
                    if delta < -0.03: failures.append(f"baseline_delta:{name}")
        else:
            baseline_check = {"status": "not_provided"}
        report = {"report_version": self.report_version, "gate": gate, "status": "PASS" if metrics and not failures else ("BLOCKED" if not metrics else "FAIL"),
                  "evaluator_version": evaluation.get("evaluator_version", ""), "manifest": dict(manifest or {}),
                  "N": evaluation.get("N", 0), "status_counts": dict(evaluation.get("status_counts", {})),
                  "execution": {key: evaluation[key] for key in ("executed_cases", "bundle_count", "trajectory_valid_rate", "freeze_checksum_rate") if key in evaluation},
                  "metrics": metrics, "thresholds": thresholds, "failed_thresholds": failures}
        report["baseline_check"] = baseline_check
        report["checksum"] = hashlib.sha256(json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return report

    def markdown(self, report: Mapping[str, Any]) -> str:
        lines = [f"# {report.get('gate', 'M4')} evaluation report", "", f"- status: {report.get('status')}",
                 f"- evaluator_version: {report.get('evaluator_version', '')}", f"- N: {report.get('N', 0)}", "", "## Metrics", ""]
        for name, value in (report.get("metrics") or {}).items():
            lines.append(f"- {name}: value={value.get('value', 0.0)}, numerator={value.get('numerator', 0)}, denominator={value.get('denominator', 0)}, coverage={value.get('coverage', 0.0)}, Wilson95={value.get('wilson95')}")
        lines.extend(["", "## Status counts", ""])
        for name, count in (report.get("status_counts") or {}).items(): lines.append(f"- {name}: {count}")
        if report.get("thresholds"):
            lines.extend(["", "## Gate thresholds", ""])
            for name, item in report["thresholds"].items(): lines.append(f"- {name}: {item['value']} >= {item['threshold']} ({'PASS' if item['passed'] else 'FAIL'})")
        lines.append("")
        return "\n".join(lines)

    def write(self, report: Mapping[str, Any], output: str | Path, *, markdown_output: str | Path | None = None) -> tuple[str, str | None]:
        target = Path(output); target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        md_path: str | None = None
        if markdown_output is not None:
            md = Path(markdown_output); md.parent.mkdir(parents=True, exist_ok=True); md.write_text(self.markdown(report), encoding="utf-8"); md_path = str(md)
        return str(target), md_path


__all__ = ["M4_THRESHOLDS", "ReportGenerator"]
