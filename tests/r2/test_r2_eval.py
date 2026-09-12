from __future__ import annotations

import hashlib
import json
from pathlib import Path

from eval.r2_correction_eval import evaluate


ROOT = Path(__file__).parents[2]
DATASET = ROOT / "eval/datasets/r2_correction/dev-cases.jsonl"


def _rows():
    return [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_frozen_cases_repeat_exact_goal_across_world_states():
    rows = _rows()
    assert len(rows) == 24
    grouped = {}
    for row in rows:
        goal_hash = hashlib.sha256(json.dumps(row["goal"], sort_keys=True).encode()).hexdigest()
        grouped.setdefault(goal_hash, set()).add((row["world"]["logistics"], row["world"].get("fault")))
    repeated = [states for states in grouped.values() if len(states) >= 5]
    assert len(repeated) >= 3
    assert all({state for state, _ in states} >= {"FRESH", "MISSING", "STALE", "CONFLICT"} for states in repeated)


def test_corrected_evaluator_executes_target_and_four_baselines(tmp_path: Path):
    report = evaluate(DATASET, tmp_path / "report.json", tmp_path / "runs")
    assert report["N"] == 24
    assert report["status"] == "ENGINEERING_PASS"
    assert report["quality_claim"] == "NOT_VALIDATED"
    assert report["systems"]["contract_smoke"]["metrics"]["case_accuracy"] == 1.0
    assert report["systems"]["fixed_order_logistics"]["metrics"]["case_accuracy"] < 1.0
    assert report["systems"]["all_tools"]["metrics"]["case_accuracy"] < 1.0
    assert report["systems"]["policy_lookup"]["metrics"]["case_accuracy"] < 1.0
    assert report["systems"]["no_replan"]["metrics"]["case_accuracy"] < 1.0
    assert all(item["bundle_verified"] for item in report["systems"]["contract_smoke"]["observations"])


def test_report_hash_changes_when_dataset_changes(tmp_path: Path):
    rows = _rows()[:1]
    original_data = tmp_path / "original-data.jsonl"
    original_data.write_text(json.dumps(rows[0], sort_keys=True) + "\n", encoding="utf-8")
    altered = tmp_path / "altered.jsonl"
    rows[0]["expected"]["code"] = "ALTERED"
    altered.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    original = evaluate(original_data, tmp_path / "original.json", tmp_path / "original-runs")
    changed = evaluate(altered, tmp_path / "changed.json", tmp_path / "changed-runs")
    assert original["dataset_sha256"] != changed["dataset_sha256"]
    assert changed["systems"]["contract_smoke"]["metrics"]["case_accuracy"] < 1.0
