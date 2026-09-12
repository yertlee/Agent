"""R3.5 provisional-candidate grounded dev evaluator.

The command is retrieval-only unless ``--live`` is explicitly supplied.  A
live run uses the existing OpenAI-compatible configuration boundary; this
module records only non-sensitive provider metadata and never emits secrets.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

from agent.r3_5_rag import R35Retriever, load_r35_manifest
from agent.r3_grounded_qa import R3GroundedQARuntime
from eval.r3_5_eval import CANONICAL_STATUSES, SAFE_ABSTENTION, load_jsonl, ratio, wilson


def _metric(successes: int, total: int, *, required: float = 1.0) -> dict[str, Any]:
    result = ratio(successes, total)
    result["required"] = required
    result["pass"] = total > 0 and result["value"] >= required
    return result


def _claims(result: Any) -> list[dict[str, Any]]:
    return [claim.model_dump() if hasattr(claim, "model_dump") else dict(claim) for claim in result.claims]


def _trace_complete(row: Mapping[str, Any]) -> bool:
    required = (
        "query", "mode", "scores", "version_tuple", "evidence_ids", "components_called",
        "provider_called", "provider_returned", "model", "latency_ms", "schema_error",
        "grounding_pass", "error", "status", "answer", "claims", "attempt_count", "attempts", "final_result", "total_latency_ms",
    )
    return all(key in row for key in required)


def _run_rows(runtime: R3GroundedQARuntime, retriever: R35Retriever, cases: Sequence[Mapping[str, Any]], gold: Mapping[str, Mapping[str, Any]], mode: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        case_id = str(case["case_id"])
        expected = str(gold[case_id].get("expected_status", case.get("expected_status", "FAILED")))
        as_of = date.fromisoformat(str(case["as_of"]))
        # Keep retrieval fields from the same shared R35Retriever used by the
        # grounded runtime.  Both are deterministic for a fixed mode/seed.
        retrieval = retriever.retrieve(str(case["query"]), as_of=as_of, mode=mode)
        qa = runtime.answer(str(case["query"]), as_of=as_of, mode=mode)
        rt = retrieval.trace
        qt = qa.trace
        stale_error = bool(
            retrieval.status == "ANSWERED"
            and any(not hit.chunk.active_at(as_of) for hit in retrieval.hits)
        )
        error = qt.get("error") or rt.get("error") or retrieval.failure_code
        row = {
            "case_id": case_id,
            "expected_status": expected,
            "query": str(case["query"]),
            "mode": mode,
            "scores": list(rt.get("scores", [])),
            "version_tuple": list(qt.get("retrieval_version_tuple") or rt.get("version_tuple", [])),
            "evidence_ids": list(qt.get("evidence_ids") or rt.get("evidence_ids", [])),
            "components_called": list(rt.get("components_called", [])),
            "provider_called": bool(qt.get("provider_called", False)),
            "provider_returned": bool(qt.get("provider_returned", False)),
            "model": qt.get("model"),
            "latency_ms": int(qt.get("latency_ms", 0) or 0),
            "schema_error": qt.get("schema_error"),
            "grounding_pass": qt.get("grounding_pass"),
            "error": error,
            "status": qa.status,
            "final_status": qa.status,
            "status_exact": qa.status == expected,
            "answer": qa.answer,
            "claims": _claims(qa),
            "attempt_count": int(qt.get("attempt_count", 0) or 0),
            "attempts": list(qt.get("attempts", [])),
            "final_result": qt.get("final_result", qa.status),
            "total_latency_ms": int(qt.get("total_latency_ms", qt.get("latency_ms", 0)) or 0),
            "stale_version_error": stale_error,
        }
        row["trace_complete"] = _trace_complete(row)
        rows.append(row)
    return rows


def evaluate_external(
    manifest_path: str | Path,
    cases: Sequence[Mapping[str, Any]],
    gold: Sequence[Mapping[str, Any]],
    *,
    mode: str = "hybrid_reranker",
    live: bool = False,
    db_path: str | Path = ".",
    provider: Any | None = None,
    config: Any | None = None,
) -> dict[str, Any]:
    manifest = load_r35_manifest(manifest_path)
    retriever = R35Retriever(manifest)
    gold_by_id = {str(row["case_id"]): row for row in gold}
    if live:
        runtime = R3GroundedQARuntime.from_environment(retriever, db_path=str(db_path), provider=provider)
    else:
        # Supplying a provider/config is reserved for deterministic unit tests;
        # the command line default supplies neither and cannot call a model.
        runtime = R3GroundedQARuntime(retriever, config=config, provider=provider)
    rows = _run_rows(runtime, retriever, cases, gold_by_id, mode)
    answerable = [row for row in rows if row["expected_status"] == "ANSWERED"]
    abstention = [row for row in rows if row["expected_status"] != "ANSWERED"]
    returned = [row for row in rows if row["provider_returned"]]
    called = [row for row in rows if row["provider_called"]]
    status_counts = {status: sum(row["status"] == status for row in rows) for status in CANONICAL_STATUSES}
    confusion = {expected: {actual: sum(row["expected_status"] == expected and row["status"] == actual for row in rows) for actual in CANONICAL_STATUSES} for expected in CANONICAL_STATUSES}
    answerable_success = sum(row["expected_status"] == "ANSWERED" and row["status"] == "ANSWERED" and row["grounding_pass"] is True for row in rows)
    abstention_success = sum(row["expected_status"] != "ANSWERED" and row["status"] in SAFE_ABSTENTION for row in rows)
    schema_valid = sum(row["provider_returned"] and not row["schema_error"] for row in rows)
    grounding_pass = sum(row["grounding_pass"] is True for row in rows if row["provider_returned"])
    status_exact = sum(row["status_exact"] for row in rows)
    stale_errors = sum(row["stale_version_error"] for row in rows)
    trace_complete = sum(row["trace_complete"] for row in rows)
    metrics = {
        "answerable_task_success": _metric(answerable_success, len(answerable), required=0.90),
        "safe_abstention": _metric(abstention_success, len(abstention), required=0.85),
        "provider_success": _metric(len(returned), len(called), required=1.00),
        "schema_validity": _metric(schema_valid, len(returned), required=1.00),
        "grounding_quotation_validation": _metric(grounding_pass, len(returned), required=1.00),
        "full_trace_persistence": _metric(trace_complete, len(rows), required=1.00),
        "stale_version_error_count": {"value": stale_errors, "numerator": stale_errors, "denominator": len(rows), "required": 0, "pass": stale_errors == 0},
        "status_exact_accuracy": _metric(status_exact, len(rows), required=1.00),
        "provider_called": _metric(len(called), len(rows), required=0.0),
        "provider_returned": _metric(len(returned), len(rows), required=0.0),
    }
    phase2_pass = all(metrics[name]["pass"] for name in (
        "answerable_task_success", "safe_abstention", "provider_success", "schema_validity",
        "grounding_quotation_validation", "full_trace_persistence", "stale_version_error_count",
    ))
    return {
        "report_version": "r3.5.external-dev.v1",
        "split": "dev-v2",
        "live": bool(live),
        "mode": mode,
        "N": len(rows),
        "manifest_version_tuple": list(manifest.version_tuple),
        "strategy_checksum": manifest.strategy_checksum,
        "phase2": {"name": "grounded_llm_dev", "executed": bool(live or provider is not None), "pass": phase2_pass, "status": "PASS" if phase2_pass else "FAIL_OR_NOT_RUN", "metrics": metrics},
        "metrics": metrics,
        "status_counts": status_counts,
        "status_confusion": confusion,
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/r3_corpus_v1/manifest-r3_5.json")
    parser.add_argument("--input", default="eval/datasets/r3_5/dev-v2-inputs.jsonl")
    parser.add_argument("--gold", default="eval/datasets/r3_5/dev-v2-gold.jsonl")
    parser.add_argument("--mode", choices=("bm25", "dense", "hybrid_no_reranker", "hybrid_reranker"), default="hybrid_reranker")
    parser.add_argument("--db-path", default=".")
    parser.add_argument("--output")
    parser.add_argument("--live", action="store_true", help="explicitly enable the configured external provider")
    args = parser.parse_args(argv)
    if args.output and Path(args.output).exists():
        raise FileExistsError("refusing to overwrite an existing R3.5 external-dev result")
    report = evaluate_external(args.manifest, load_jsonl(args.input), load_jsonl(args.gold), mode=args.mode, live=args.live, db_path=args.db_path)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
