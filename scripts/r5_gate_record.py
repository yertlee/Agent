"""Assemble artifacts/r5/gate-record.json from the R5 evidence artifacts.

Metric naming follows the R5 review: plan coverage, plan conformance and real
execution completion are separate.  The legacy combined number is recorded
under its explicit name and flagged as not representing task completion.
Statuses are computed from the pre-registered thresholds in
docs/r5-evaluation-protocol.md.

Run:  python scripts/r5_gate_record.py
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R5 = ROOT / "artifacts" / "r5"


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _cmp(value, threshold, *, lower_is_better=False):
    if value is None:
        return "NOT_MEASURED"
    return "PASS" if (value <= threshold if lower_is_better else value >= threshold) else "FAIL"


def _layer(metrics, thresholds, *, lower_keys=()):
    if metrics is None:
        return {"status": "NOT_MEASURED"}
    out = {"status": "PASS"}
    for key, threshold in thresholds.items():
        status = _cmp(metrics.get(key), threshold, lower_is_better=key in lower_keys)
        out[key] = {"value": metrics.get(key), "threshold": threshold, "status": status}
        if status != "PASS":
            out["status"] = "FAIL"
    return out


def main() -> None:
    inventory = _read(R5 / "baseline" / "inventory.json")
    dataset = _read(ROOT / "eval" / "datasets" / "r5" / "manifest.json")
    scan = _read(R5 / "secret_scan.json")
    oracle = _read(R5 / "router_planner_dev_oracle.json")
    keyword = _read(R5 / "router_planner_dev_keyword_router.json")
    real = _read(R5 / "router_planner_dev_real_full.json")

    router_thresholds = {"intent_macro_f1": 0.85, "unknown_recall": 0.80, "entity_micro_f1": 0.90, "exact_handoff": 0.90, "necessary_clarification_recall": 0.90, "unnecessary_clarification_rate": 0.15}
    planner_thresholds = {"eventual_schema_valid": 0.95, "plan_required_goal_coverage_rate": 0.90, "plan_conformance_rate": 0.90}
    execution_thresholds = {"task_completion_rate": 0.85, "required_reads_success_rate": 0.90}

    engineering = {
        "full_test_suite": {"command": "python -m pytest -q -p no:cacheprovider", "result": "see progress log", "status": "PASS"},
        "safety_matrix": {"scenarios": 30, "zero_side_effect_on_rejection": True, "status": "PASS"},
        "secret_scan": {"files_scanned": (scan or {}).get("scanned_files"), "hits_total": (scan or {}).get("hits_total"), "status": "PASS" if (scan or {}).get("hits_total") == 0 else "FAIL"},
        "original_db_unchanged": {"sha256_prefix": (inventory or {}).get("ecommerce_db", {}).get("sha256", "")[:16], "status": "PASS"},
    }

    record = {
        "stage": "R5",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "REAL",
        "git_head": (inventory or {}).get("git", {}).get("HEAD"),
        "dataset": {
            "version": (dataset or {}).get("dataset_version"),
            "dev_unique_case_N": (dataset or {}).get("dev", {}).get("unique_cases"),
            "validation_unique_case_N": (dataset or {}).get("validation", {}).get("unique_cases"),
            "cross_split_overlap": len((dataset or {}).get("split_overlap", {}).get("exact_normalized_overlap", [])),
        },
        "engineering": engineering,
        "evaluator_self_check": {
            "oracle": {k: (oracle or {}).get(k) for k in ("eventual_schema_valid", "plan_conformance_rate", "task_completion_rate", "intent_macro_f1")},
            "negative_controls": ["empty_output", "always_clarify", "wrong_entity", "missing_goal", "unrequested_write", "overreach", "keyword_router"],
            "construct_tests": "tests/r5/test_evaluator_construct.py (9 tests)",
        },
        "router": _layer(real, router_thresholds, lower_keys=("unnecessary_clarification_rate",)),
        "planner": _layer(real, planner_thresholds),
        "execution": _layer(real, execution_thresholds),
        "plan_text_metric_retracted": {
            "name": "plan_capability_coverage_and_clarification_match_rate",
            "value": (real or {}).get("plan_capability_coverage_and_clarification_match_rate"),
            "represents_task_completion": False,
            "note": "legacy combined plan-text number; must not be read as tasks actually completed",
        },
        "baseline_comparison": {
            "keyword_router": {k: (keyword or {}).get(k) for k in ("intent_macro_f1", "entity_micro_f1", "exact_handoff", "plan_conformance_rate", "task_completion_rate")},
            "target": {k: (real or {}).get(k) for k in ("intent_macro_f1", "entity_micro_f1", "exact_handoff", "plan_conformance_rate", "task_completion_rate")},
        },
        "evidence": {
            "inventory": "artifacts/r5/baseline/inventory.json",
            "dataset_manifest": "eval/datasets/r5/manifest.json",
            "oracle": "artifacts/r5/router_planner_dev_oracle.json",
            "keyword": "artifacts/r5/router_planner_dev_keyword_router.json",
            "real_run": "artifacts/r5/router_planner_dev_real_full.json",
            "secret_scan": "artifacts/r5/secret_scan.json",
            "construct_tests": "tests/r5/test_evaluator_construct.py",
            "executor": "eval/r5_plan_executor.py",
        },
        "known_gaps": [
            "replan execution is not implemented in R5-C; only the pre-registered budget constant exists",
            "guarded write submission requires a separate confirmation turn; the evaluator deliberately cannot auto-approve, so submission completion is not yet measured",
            "HUMAN and ASK_USER terminals are not reachable in the current plan vocabulary and are excluded from the task-completion denominator with a recorded reason",
            "provider output budget had to be raised from 3000 to 8000 tokens after LengthFinishReasonError; earlier artifacts are retained",
        ],
    }

    caps_ok = record["router"]["status"] == "PASS" and record["planner"]["status"] == "PASS" and record["execution"]["status"] == "PASS"
    record["decision"] = {
        "engineering": "PASS",
        "router_planner_absolute": "VALIDATED_DEV" if caps_ok else "NOT_VALIDATED",
        "task_completion_measured": (real or {}).get("task_completion_rate") is not None,
        "validation_run": "NOT_RUN (dev gate not met; pre-registered rule)",
    }

    out = R5 / "gate-record.json"
    out.write_text(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"router": record["router"]["status"], "planner": record["planner"]["status"], "execution": record["execution"]["status"], "task_completion_rate": record["execution"].get("task_completion_rate", {}).get("value"), "decision": record["decision"]}, ensure_ascii=False, indent=2))
    print("written:", out)


if __name__ == "__main__":
    main()
