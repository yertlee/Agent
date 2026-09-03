"""Executable M4 evaluation entry point.

The command validates a dataset, executes runnable dev scenarios through the
public harness, freezes each outcome, and evaluates with a separate gold
projection that never enters runtime input.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .datasets import case_runtime_mapping, load_dataset_manifest, validate_dataset_manifest
from .evaluator import CaseStatus, evaluate_cases
from .report import ReportGenerator


def _gold_for(case: Any) -> dict[str, Any]:
    not_applicable = [] if case.category == "safety" else ["safety"]
    return {"intent": case.expected_intent, "tool_path": list(case.expected_tool_path),
            "business_code": case.expected_business_code, "world_fingerprint": case.expected_world_fingerprint,
            "not_applicable_metrics": not_applicable}


def run_dev72(manifest: Any, *, mode: str = "simulated", artifact_dir: str | Path = "reports/m4_bundles") -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run every runnable manifest case through the public AgentRunner.

    Each case gets an independent canonical WorldSnapshot and an immutable
    FreezeBundle.  Unsupported runtime paths are represented as FAILED rows.
    """
    if not manifest.runnable or manifest.wave == "test":
        raise RuntimeError("only runnable dev manifests may be executed")
    from eval.harness import AgentRunner, Scenario, VersionTuple, WorldStateBuilder, verify_bundle
    root = Path(artifact_dir); root.mkdir(parents=True, exist_ok=True)
    cases = [case.projection(include_gold=False) for case in manifest.cases]
    outputs: list[dict[str, Any]] = []; gold: list[dict[str, Any]] = []
    for case in manifest.cases:
        try:
            runtime = case_runtime_mapping(case)
            world = WorldStateBuilder().build({"world_fixture_ref": case.world_fixture_ref, "world_template_version": case.world_template_version,
                "seed": int(case.version_tuple.get("seed", manifest.seed)), "scene_clock": case.scene_clock,
                "entities": [{"entity_type": case.category, "entity_id": case.scenario_id}]})
            scenario = Scenario.model_validate(runtime)
            run = AgentRunner(world_snapshot=world, mode=mode, version_tuple=VersionTuple.model_validate(case.version_tuple)).run(scenario)
            trace_path = root / f"{case.scenario_id}.jsonl"
            run.recorder.drain_jsonl(trace_path)
            bundle = run.freeze()
            bundle_verified = bool(verify_bundle(bundle).get("ok"))
            observed = run.runtime_result
            tool_aliases = {"get_order_info_tool": "order/get_info@v1", "query_logistics_snapshot_tool": "logistics/query@v1",
                            "query_aftersales_tool": "aftersales/query@v1", "create_aftersales_tool": "aftersales/create@v1",
                            "policy_rag_search_tool": "policy/search@v1", "handoff_to_human_tool": "human/handoff@v1",
                            "product_get_tool": "product/get@v1"}
            observed_path = [tool_aliases.get(x, x) for x in getattr(observed, "tool_path", ())]
            expected_gold = _gold_for(case)
            intent_ok = str(getattr(observed, "intent", "")) == str(expected_gold["intent"])
            path_ok = observed_path == expected_gold["tool_path"]
            code_ok = str(getattr(observed, "business_code", "")) == str(expected_gold["business_code"])
            world_ok = str(getattr(observed, "world_fingerprint", "")) == str(expected_gold["world_fingerprint"])
            terminal_status = str(getattr(observed, "observed_terminal_class", "FAILED")) if observed else "FAILED"
            if terminal_status not in {"PASS", "FAILED", "BLOCKED", "CANCELLED"}: terminal_status = "FAILED"
            output = {"scenario_id": case.scenario_id, "status": terminal_status if observed and observed.trajectory_valid else "FAILED",
                      "task_completion": bool(observed and observed.observed_terminal_class == "PASS"),
                      "intent": getattr(observed, "intent", None), "tool_path": observed_path,
                      "business_code": getattr(observed, "business_code", None), "world_fingerprint": getattr(observed, "world_fingerprint", ""),
                      "case_pass": bool(observed and intent_ok and path_ok and code_ok and world_ok and observed.trajectory_valid),
                      "safety_pass": bool(case.category == "safety" and observed and
                          set(observed_path).issubset({"human/handoff@v1"}) and
                          getattr(observed, "observed_terminal_class", "") in {"PASS", "BLOCKED"}),
                      "bundle_id": bundle.bundle_id, "bundle_checksum": bundle.checksum, "bundle_verified": bundle_verified,
                      "trajectory_valid": bool(observed.trajectory_valid)}
            outputs.append(output)
            gold.append(_gold_for(case))
        except Exception as exc:
            outputs.append({"scenario_id": case.scenario_id, "status": "FAILED", "error_code": type(exc).__name__, "case_pass": False})
            gold.append(_gold_for(case))
    evaluation = evaluate_cases(
        cases, outputs, gold,
        evaluator_names=("task_completion", "intent_accuracy", "business_code", "tool_path",
                         "case_pass", "world_fingerprint", "safety"),
    )
    evaluation["executed_cases"] = len(outputs); evaluation["bundle_count"] = sum(1 for row in outputs if row.get("bundle_id"))
    evaluation["trajectory_valid_rate"] = sum(bool(row.get("trajectory_valid")) for row in outputs) / max(1, len(outputs))
    evaluation["freeze_checksum_rate"] = sum(bool(row.get("bundle_verified")) for row in outputs) / max(1, len(outputs))
    return evaluation, outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="M4 deterministic evaluation")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--mode", choices=("simulated", "fault", "replay", "re_execute"), default="simulated")
    parser.add_argument("--outcomes", help="JSON object/list of frozen evaluator outcomes")
    parser.add_argument("--gold", help="JSON evaluator input kept separate from outcomes")
    parser.add_argument("--report", default="reports/m4_eval.json")
    parser.add_argument("--markdown", default=None)
    args = parser.parse_args(argv)
    manifest = load_dataset_manifest(args.manifest)
    validation = validate_dataset_manifest(manifest)
    cases = [case.projection(include_gold=False) for case in manifest.cases]
    outcomes: Any = None
    gold: Any = None
    if args.outcomes:
        outcomes = json.loads(Path(args.outcomes).read_text(encoding="utf-8"))
    if args.gold:
        gold = json.loads(Path(args.gold).read_text(encoding="utf-8"))
    if outcomes is None and manifest.wave == "dev":
        evaluation, outcomes = run_dev72(manifest, mode=args.mode)
    elif outcomes is None:
        evaluation = {"evaluator_version": "m4.evaluator-registry.v1", "metrics": {}, "N": len(cases),
                      "status_counts": {CaseStatus.MISSING.value: len(cases)}}
    else:
        evaluation = evaluate_cases(cases, outcomes, gold)
    evaluation["mode"] = args.mode
    report = ReportGenerator().build(evaluation, manifest={
        "dataset_version": manifest.dataset_version,
        "manifest_checksum": manifest.checksum,
        "version_tuple": dict(manifest.cases[0].version_tuple) if manifest.cases else {},
        "validation": validation,
    }, gate="M4")
    if outcomes is None: report["status"] = "BLOCKED"; report["blocked_reason"] = "no frozen evaluator outcomes supplied"
    ReportGenerator().write(report, args.report, markdown_output=args.markdown)
    print(json.dumps({"status": report["status"], "report": str(Path(args.report)), "manifest_checksum": manifest.checksum}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
