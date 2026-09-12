"""Run the registered R3.5 dev-v2 systems and suggest, without freezing, a candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.r3_5_eval import ORDERED_MODES, evaluate, load_jsonl, retrieval_screen_pass


PRODUCTION_CANDIDATES = frozenset(("bm25", "dense", "hybrid_no_reranker", "hybrid_reranker"))
DIAGNOSTIC_MODES = frozenset(ORDERED_MODES) - PRODUCTION_CANDIDATES


def _component_count(report: dict) -> float:
    rows = report.get("by_case", [])
    return sum(len(set(row.get("components_called", []))) for row in rows) / max(len(rows), 1)


def _latency(report: dict) -> float:
    rows = report.get("by_case", [])
    return sum(float(row.get("latency_ms", 0)) for row in rows) / max(len(rows), 1)


def phase_one_eligible(mode: str, report: dict) -> bool:
    """Return whether a mode may enter the provisional-candidate pool."""
    return mode in PRODUCTION_CANDIDATES and retrieval_screen_pass(report)


def choose_provisional(reports: dict[str, dict]) -> tuple[str, ...]:
    eligible = [
        (report["metrics"]["overall_task_success"]["value"], -_component_count(report), -_latency(report), mode)
        for mode, report in reports.items()
        if phase_one_eligible(mode, report)
    ]
    return tuple(item[-1] for item in sorted(eligible, reverse=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/r3_corpus_v1/manifest-r3_5.json")
    parser.add_argument("--data", default="eval/datasets/r3_5")
    parser.add_argument("--output", default="artifacts/r3_5/dev-v2")
    args = parser.parse_args(argv)
    data = Path(args.data)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    reports: dict[str, dict] = {}
    for mode in ORDERED_MODES:
        split = "mutation" if mode == "mutation" else "dev-v2"
        report = evaluate(args.manifest, load_jsonl(data / f"{split}-inputs.jsonl"), load_jsonl(data / f"{split}-gold.jsonl"), mode)
        report["selection_order"] = list(ORDERED_MODES)
        report["candidate_system"] = mode in PRODUCTION_CANDIDATES
        report["phase1_retrieval_eligible"] = phase_one_eligible(mode, report)
        (output / f"{mode}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        reports[mode] = report
    eligible = choose_provisional(reports)
    recommended = eligible[0] if eligible else None
    selection = {
        "report_version": "r3.5.dev-selection.v2",
        "split": "dev-v2",
        "candidate_frozen": False,
        "selection_rule": ["phase1 allowlist: bm25/dense/hybrid_no_reranker/hybrid_reranker", "recall >= 0.90", "answerable task success >= 0.90", "stale version error count = 0", "trace completeness = 1.00", "manifest integrity = 1.00", "runtime failure count = 0", "maximize overall task success", "tie-break fewer enabled components then lower latency"],
        "phase1": {
            "name": "retrieval_dev_screen",
            "eligible_modes": list(eligible),
            "provisional_candidate": recommended,
            "candidate_frozen": False,
        },
        "phase2": {
            "name": "grounded_llm_dev",
            "required_before_freeze": True,
            "executed": False,
            "status": "PENDING_EXTERNAL_LLM_DEV_EVAL",
            "candidate_frozen": False,
            "selection_metrics": ["answerable task success >= 0.90", "safe abstention >= 0.85", "provider success = 1.00", "schema validity = 1.00", "grounding / quotation validation = 1.00", "full trace persistence = 1.00", "stale version error count = 0"],
        },
        "recommended_mode": recommended,
        "provisional_candidate": recommended,
        "eligible_modes": list(eligible),
        "diagnostic_modes": sorted(DIAGNOSTIC_MODES),
        "systems": {mode: {"selection_eligible": report["phase1_retrieval_eligible"], "safety_eligible_legacy": report["selection_eligible"], "phase1_retrieval_eligible": report["phase1_retrieval_eligible"], "candidate_system": report["candidate_system"], "overall_task_success": report["metrics"]["overall_task_success"], "safe_abstention_accuracy": report["metrics"]["safe_abstention_accuracy"], "failures": report["failures"]} for mode, report in reports.items()},
    }
    (output / "selection.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(selection, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
