"""Deterministic M3 contract-level scenario and trajectory runner."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import yaml

from agent.m3_runtime import M3ScenarioRunner
from .m3_manifest import load_manifest, validate_dev44


def run_manifest(path: str | Path) -> dict:
    validation = validate_dev44(path)
    manifest = load_manifest(path)
    cases = manifest["cases"]
    rubric = manifest.get("rubric") or {}
    case_by_id = {str(c["scenario_id"]): c for c in cases}
    runner = M3ScenarioRunner()
    runs = [runner.run(case) for case in cases]
    denominator = len(runs)
    fixture_doc = yaml.safe_load((Path(__file__).resolve().parent / "scenarios" / "world_fixtures.yaml").read_text(encoding="utf-8")) or {}
    gold_world = fixture_doc.get("world_fixtures", {}).get("gold_fingerprints", {})
    fixture_rows = {str(row["world_fixture_ref"]): row for row in fixture_doc.get("world_fixtures", {}).get("fixtures", [])}
    def intent_match(run):
        expected = run.expected_intent
        return run.intent == expected or (expected == "ORDER_QUERY" and run.intent in {"ORDER", "MIXED"}) or (expected == "AFTERSALES" and run.intent in {"AFTERSALES", "MIXED"}) or (expected == "POLICY" and run.intent in {"POLICY", "MIXED"}) or (expected == "LOGISTICS" and run.intent == "LOGISTICS") or (expected == "HANDOFF" and run.intent in {"ESCALATION", "MIXED"}) or (expected == "MIXED" and run.intent == "MIXED")
    def path_covered(run):
        observed = list(run.tool_path)
        expected = list(run.expected_tool_path)
        case = case_by_id[run.scenario_id]
        optional = set(case.get("allowed_optional_tools", rubric.get("allowed_optional_tools", [])) or [])
        forbidden = set(case.get("forbidden_tools", rubric.get("forbidden_tools", [])) or [])
        if any(tool in forbidden for tool in observed):
            return False
        cursor = 0
        for item in expected:
            try:
                cursor = observed.index(item, cursor) + 1
            except ValueError:
                return False
        required_positions = []
        cursor = 0
        for item in expected:
            pos = observed.index(item, cursor)
            required_positions.append(pos)
            cursor = pos + 1
        return all(tool in expected or tool in optional for tool in observed)
    def world_match(run):
        ref = next((c.get("world_fixture_ref") for c in cases if str(c["scenario_id"]) == run.scenario_id), "")
        return run.world_fingerprint == gold_world.get(ref)
    def business_match(run):
        case = case_by_id[run.scenario_id]
        # Business-code evaluation is bound to the case's frozen expected
        # result.  Category/rubric defaults and failure_script are execution
        # inputs, not evaluation gold, and must never widen this comparison.
        expected = case.get("expected_business_codes")
        if not isinstance(expected, list) or len(expected) != 1:
            return False
        return str(run.business_code) == str(expected[0])
    def plan_match(run):
        case = case_by_id[run.scenario_id]
        expected = manifest.get("gold_plan", {}).get(run.scenario_id, case.get("gold_plan", []))
        def normalize(rows):
            return [{"tool_ref": str(row.get("tool_ref")), "capability_ref": str(row.get("capability_ref")), "depends_on": [int(x) for x in (row.get("depends_on") or [])]} for row in (rows or [])]
        return normalize(run.plan_projection) == normalize(expected)
    assertion_rows = []
    for run in runs:
        assertions = {"intent": intent_match(run), "plan": plan_match(run), "business_code": business_match(run), "tool_path": path_covered(run), "terminal": run.observed_terminal_class == run.expected_terminal_class, "world_fingerprint": world_match(run)}
        assertion_rows.append((run, assertions))
    terminal_matches = sum(a["terminal"] for _, a in assertion_rows)
    world_valid = sum(a["world_fingerprint"] for _, a in assertion_rows)
    # Per the M3 rubric, a runtime FAILED/BLOCKED/CANCELLED is a failed
    # applicable task even when the fixture expected that failure class.
    case_pass = sum(all(a.values()) and run.observed_terminal_class == "PASS" for run, a in assertion_rows)
    task_completion = sum(run.observed_terminal_class == "PASS" and a["plan"] for run, a in assertion_rows)
    trajectory_rows = [
        {"scenario_id": r.scenario_id, "run_id": r.run_id, "db_path": r.db_path,
         "observed_intent": r.intent, "observed_tool_path": list(r.tool_path),
         "expected_intent": r.expected_intent, "expected_tool_path": list(r.expected_tool_path),
         "observed_terminal_class": r.observed_terminal_class,
         "expected_terminal_class": r.expected_terminal_class,
         "world_fingerprint": r.world_fingerprint,
         "business_code": r.business_code,
         "plan_fingerprint": r.plan_fingerprint,
         "terminal_fingerprint": r.terminal_fingerprint,
         "logical_calls": r.logical_calls, "physical_attempts": r.physical_attempts,
         "plan_projection": list(r.plan_projection),
         "trajectory_valid": r.trajectory_valid,
         "assertions": assertions}
        for r, assertions in assertion_rows
    ]
    return {
        "manifest_sha256": validation["manifest_sha256"],
        "metrics": {
            "task_completion": task_completion / denominator,
            "intent_accuracy": sum(a["intent"] for _, a in assertion_rows) / denominator,
            "tool_path": sum(a["tool_path"] for _, a in assertion_rows) / denominator,
            "case_pass": case_pass / denominator,
            "world_fingerprint": world_valid / denominator,
        },
        "denominator": {"N_applicable": denominator, "N_missing": 0, "N_failed": denominator - case_pass, "N_blocked": sum(r.observed_terminal_class == "BLOCKED" for r in runs), "N_cancelled": sum(r.observed_terminal_class == "CANCELLED" for r in runs), "PASS": sum(r.observed_terminal_class == "PASS" for r in runs), "N_assertions_pass": sum(all(a.values()) for _, a in assertion_rows), "FAILED": sum(r.observed_terminal_class == "FAILED" for r in runs), "BLOCKED": sum(r.observed_terminal_class == "BLOCKED" for r in runs), "CANCELLED": sum(r.observed_terminal_class == "CANCELLED" for r in runs)},
        "trajectory": {"logical_calls": sum(r.logical_calls for r in runs), "physical_attempts": sum(r.physical_attempts for r in runs), "valid_runs": sum(r.trajectory_valid for r in runs), "cases": trajectory_rows},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="eval/manifests/dev44.yaml")
    parser.add_argument("--trajectory-output", default="reports/m3/trajectory_report.json")
    args = parser.parse_args()
    report = run_manifest(args.manifest)
    output = Path(args.trajectory_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
