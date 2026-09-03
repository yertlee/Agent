"""Compare frozen baseline/candidate dev72 projections with M4 Comparator."""
from __future__ import annotations

import json
from pathlib import Path

from .comparator import BaselineComparator, ComparableRun
from .datasets import load_dataset_manifest


def _sequences(manifest, outputs):
    rows = {str(row.get("scenario_id")): row for row in outputs}
    result = {name: [] for name in ("task_completion", "intent_accuracy", "tool_path", "case_pass", "world_fingerprint")}
    for case in manifest.cases:
        row = rows.get(case.scenario_id, {}); failed = str(row.get("status", "MISSING")) != "PASS"
        result["task_completion"].append(float(not failed and row.get("task_completion", False)))
        result["intent_accuracy"].append(float(not failed and row.get("intent") == case.expected_intent))
        result["tool_path"].append(float(not failed and row.get("tool_path", []) == list(case.expected_tool_path)))
        result["case_pass"].append(float(not failed and row.get("case_pass", False)))
        result["world_fingerprint"].append(float(not failed and row.get("world_fingerprint") == case.expected_world_fingerprint))
    return result


def main() -> int:
    manifest = load_dataset_manifest("eval/manifests/dev72.yaml")
    baseline = json.loads(Path("reports/baseline_outcomes.json").read_text(encoding="utf-8"))
    candidate = json.loads(Path("reports/candidate_outcomes.json").read_text(encoding="utf-8"))
    if baseline["manifest_checksum"] != candidate["manifest_checksum"] or baseline["manifest_checksum"] != manifest.checksum:
        raise ValueError("paired runs use different manifest checksums")
    base_version = dict(manifest.cases[0].version_tuple); base_version.update(code="m3.baseline.59b3cc1", harness="m4.baseline-adapter.v1")
    candidate_version = dict(manifest.cases[0].version_tuple); candidate_version.update(code="m4.candidate.current", harness="m4.harness.v1")
    ids = tuple(case.scenario_id for case in manifest.cases)
    gold_version = f"gold:{manifest.checksum}"
    left = ComparableRun("baseline-59b3cc1", base_version, manifest.dataset_version, ids, gold_version, "m4.evaluator-registry.v1", _sequences(manifest, baseline["outputs"]))
    right = ComparableRun("candidate-current", candidate_version, manifest.dataset_version, ids, gold_version, "m4.evaluator-registry.v1", _sequences(manifest, candidate["outputs"]))
    comparison = BaselineComparator().compare(left, right, allowed_variations={"code", "harness"})
    result = {"baseline_commit": "59b3cc1217ddcc5f2ed893dad00eb619be3032de", "manifest_checksum": manifest.checksum,
              "allowed_variations": ["code", "harness"], "baseline": baseline["evaluation"], "candidate": candidate["evaluation"],
              "comparison": comparison.as_dict(), "regression_threshold": -0.03,
              "threshold_passed": comparison.comparable and all(float(value.get("delta", 0.0)) >= -0.03 for value in comparison.metrics.values())}
    Path("reports/m4_baseline_comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"comparable": comparison.comparable, "reason": comparison.reason, "threshold_passed": result["threshold_passed"]}, ensure_ascii=False))
    return 0 if result["threshold_passed"] else 1


if __name__ == "__main__": raise SystemExit(main())
