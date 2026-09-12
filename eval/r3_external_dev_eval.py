"""External-provider dev entry point for R3-C.

Default execution is retrieval-only and never calls an external provider.  A
caller must explicitly pass ``--live`` to construct the existing
OpenAI-compatible provider boundary; this module is not run by R3 tests.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from agent.r3_grounded_qa import R3GroundedQARuntime
from agent.r3_rag_runtime import R3Retriever
from agent.r3_rag import load_cases
from eval.r3_eval import CANONICAL_STATUSES, SAFE_ABSTENTION_STATUSES, evaluate_case_outcome


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/r3_corpus_v1/manifest.json")
    parser.add_argument("--input", default="eval/datasets/r3/dev-inputs.jsonl")
    parser.add_argument("--gold", default="eval/datasets/r3/dev-gold.jsonl")
    parser.add_argument("--db-path", default=".")
    parser.add_argument("--output")
    parser.add_argument("--live", action="store_true", help="explicitly enable the configured external provider")
    args = parser.parse_args(argv)
    retriever = R3Retriever.from_manifest(args.manifest, require_dense=True)
    runtime = R3GroundedQARuntime.from_environment(retriever, db_path=args.db_path) if args.live else None
    gold_by_id = {item["case_id"]: item for item in load_cases(args.gold)}
    rows = []
    for case in load_cases(args.input):
        expected = gold_by_id[case["case_id"]]["expected_status"]
        if runtime is None:
            result = retriever.retrieve(case["query"], as_of=date.fromisoformat(case["as_of"]), mode="hybrid_reranker")
            gold_ids = set(gold_by_id[case["case_id"]].get("gold_chunk_ids", []))
            grounded_pass = result.status == "ANSWERED" and any(hit.chunk.chunk_id in gold_ids for hit in result.hits)
            outcomes = evaluate_case_outcome(expected, result.status, grounded_pass)
            rows.append({"case_id": case["case_id"], "expected_status": expected, "status": result.status, "status_exact": result.status == expected, "provider_called": False, "provider_returned": False, "error_code": None, "schema_error": None, "grounding_pass": grounded_pass, **outcomes})
        else:
            result = runtime.answer(case["query"], as_of=date.fromisoformat(case["as_of"]))
            outcomes = evaluate_case_outcome(expected, result.status, result.trace["grounding_pass"])
            rows.append({"case_id": case["case_id"], "expected_status": expected, "status": result.status, "status_exact": result.status == expected, "provider_called": result.trace["provider_called"], "provider_returned": result.trace["provider_returned"], "error_code": result.trace["error"], "schema_error": result.trace["schema_error"], "grounding_pass": result.trace["grounding_pass"], **outcomes, "answer": result.answer, "claims": [claim.model_dump() for claim in result.claims]})
    status_counts = {status: sum(1 for row in rows if row["status"] == status) for status in CANONICAL_STATUSES}
    confusion = {expected: {actual: sum(1 for row in rows if row["expected_status"] == expected and row["status"] == actual) for actual in CANONICAL_STATUSES} for expected in CANONICAL_STATUSES}
    report = {
        "live": args.live,
        "N": len(rows),
        "status_exact": sum(1 for row in rows if row["status_exact"]),
        "status_exact_accuracy": sum(1 for row in rows if row["status_exact"]) / len(rows) if rows else 0.0,
        "answerable_success": sum(1 for row in rows if row["answerable_success"]) / sum(1 for row in rows if row["expected_status"] == "ANSWERED") if any(row["expected_status"] == "ANSWERED" for row in rows) else 0.0,
        "abstention_accuracy": sum(1 for row in rows if row["abstention_success"]) / sum(1 for row in rows if row["expected_status"] != "ANSWERED") if any(row["expected_status"] != "ANSWERED" for row in rows) else 0.0,
        "overall_task_success": sum(1 for row in rows if row["task_success"]) / len(rows) if rows else 0.0,
        "provider_called": sum(1 for row in rows if row["provider_called"]),
        "provider_returned": sum(1 for row in rows if row["provider_returned"]),
        "schema_errors": sum(1 for row in rows if row["schema_error"]),
        "grounding_pass": sum(1 for row in rows if row["grounding_pass"]),
        "status_accuracy": sum(1 for row in rows if row["status_exact"]) / len(rows) if rows else 0.0,
        "provider_return_rate": sum(1 for row in rows if row["provider_returned"]) / sum(1 for row in rows if row["provider_called"]) if any(row["provider_called"] for row in rows) else 0.0,
        "schema_valid_rate": sum(1 for row in rows if row["provider_returned"] and not row["schema_error"]) / sum(1 for row in rows if row["provider_returned"]) if any(row["provider_returned"] for row in rows) else 0.0,
        "grounding_pass_rate": sum(1 for row in rows if row["grounding_pass"]) / sum(1 for row in rows if row["provider_returned"]) if any(row["provider_returned"] for row in rows) else 0.0,
        "status_counts": status_counts,
        "status_confusion": confusion,
        "rows": rows,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        if output.exists():
            raise FileExistsError("refusing to overwrite an existing R3-C result")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
