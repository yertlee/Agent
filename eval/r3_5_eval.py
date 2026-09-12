"""R3.5 dev-v2 evaluator with component-isolated baselines and Wilson CI."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

from agent.r3_5_rag import R35_MODES, R35Retriever, load_r35_manifest, mutate_manifest, remap_gold_ids
from agent.r3_rag import Claim, RetrievalHit, RetrievalResult, build_claim_evidence, load_cases

CANONICAL_STATUSES = ("ANSWERED", "NO_HITS", "WEAK_EVIDENCE", "CONFLICT", "STALE_ONLY", "FAILED")
SAFE_ABSTENTION = frozenset(("NO_HITS", "WEAK_EVIDENCE", "CONFLICT", "STALE_ONLY"))
ORDERED_MODES = ("no_retrieval", "random", "bm25", "dense", "hybrid_no_reranker", "hybrid_reranker", "oracle", "remove_version_filter", "remove_abstention", "mutation")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def wilson(successes: int, total: int, z: float = 1.96) -> list[float]:
    if total <= 0:
        return [0.0, 0.0]
    p = successes / total
    denom = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denom
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total) / denom
    return [max(0.0, centre - margin), min(1.0, centre + margin)]


def ratio(successes: int, total: int) -> dict[str, Any]:
    return {"value": successes / total if total else 0.0, "numerator": successes, "denominator": total, "wilson95": wilson(successes, total)}


def retrieval_screen_pass(report: Mapping[str, Any]) -> bool:
    """Apply only the frozen phase-one retrieval screen.

    Safe abstention and external-provider metrics belong to phase two and are
    intentionally absent from this predicate.
    """
    metrics = report["metrics"]
    return (
        metrics["recall_at_5"]["value"] >= 0.90
        and metrics["answerable_success"]["value"] >= 0.90
        and metrics["stale_selection_rate"]["numerator"] == 0
        and metrics["trace_completeness"]["value"] == 1.0
        and metrics["manifest_completeness"]["value"] == 1.0
        and int(report.get("failures", 0)) == 0
    )


def mean_score(values: Sequence[float]) -> dict[str, Any]:
    """Report continuous ranking/citation scores without mislabeling them binomial."""
    n = len(values)
    if not n:
        return {"value": 0.0, "numerator": 0.0, "denominator": 0, "ci95": [0.0, 0.0], "ci_method": "normal_mean"}
    mean = sum(values) / n
    variance = sum((value - mean) ** 2 for value in values) / n
    margin = 1.96 * math.sqrt(variance / n)
    return {"value": mean, "numerator": sum(values), "denominator": n, "ci95": [max(0.0, mean - margin), min(1.0, mean + margin)], "ci_method": "normal_mean"}


def _oracle_result(runtime: R35Retriever, case: Mapping[str, Any], gold: Mapping[str, Any], gold_ids: Sequence[str]) -> RetrievalResult:
    chunks = {c.chunk_id: c for c in runtime.manifest.chunks}
    hits = [RetrievalHit(chunks[cid], 1.0) for cid in gold_ids if cid in chunks]
    evidence = runtime._evidence(hits)
    claims = [Claim("oracle-context", "oracle context", tuple(e.evidence_id for e in evidence))] if evidence else []
    expected = str(case.get("expected_status", gold.get("expected_status")))
    status = expected if expected != "ANSWERED" else ("ANSWERED" if hits else "NO_HITS")
    return RetrievalResult(status, str(case["query"]), "oracle", hits, evidence, claims, trace={
        "query": case["query"], "mode": "oracle", "as_of": case["as_of"], "top_k": runtime.manifest.top_k,
        "version_tuple": runtime.manifest.version_tuple, "strategy_checksum": runtime.manifest.strategy_checksum,
        "component_matrix": runtime.manifest.component_matrix["oracle"], "components_called": ["gold"],
        "evidence_ids": [e.evidence_id for e in evidence], "scores": [{"chunk_id": h.chunk.chunk_id, "score": h.score} for h in hits],
        "provider_called": False, "provider_returned": False, "model": None, "latency_ms": 0,
        "schema_error": None, "grounding_pass": None, "error": None,
    })


def _outcomes(expected: str, actual: str, grounded: bool) -> dict[str, bool]:
    answerable = expected == "ANSWERED"
    answerable_success = answerable and actual == "ANSWERED" and grounded
    abstention_success = not answerable and actual in SAFE_ABSTENTION
    return {"answerable_success": answerable_success, "abstention_success": abstention_success, "task_success": answerable_success or abstention_success}


def _run_case(runtime: R35Retriever, case: Mapping[str, Any], gold: Mapping[str, Any], mode: str, index: int, mapping: Mapping[str, str] | None = None) -> RetrievalResult:
    ids = remap_gold_ids(gold.get("gold_chunk_ids", []), mapping or {})
    if mode == "oracle":
        return _oracle_result(runtime, case, gold, ids)
    if mode == "mutation":
        return runtime.retrieve(str(case["query"]), as_of=date.fromisoformat(str(case["as_of"])), mode="mutation", seed=index + 71)
    return runtime.retrieve(str(case["query"]), as_of=date.fromisoformat(str(case["as_of"])), mode=mode, seed=index + 71)


def _case_row(result: RetrievalResult, case: Mapping[str, Any], gold: Mapping[str, Any], mapping: Mapping[str, str] | None = None) -> dict[str, Any]:
    gold_ids = set(remap_gold_ids(gold.get("gold_chunk_ids", []), mapping or {}))
    ranked = [h.chunk.chunk_id for h in result.hits]
    found = [cid for cid in ranked if cid in gold_ids]
    rank = next((i + 1 for i, cid in enumerate(ranked) if cid in gold_ids), None)
    dcg = sum(1.0 / math.log2(i + 2) for i, cid in enumerate(ranked) if cid in gold_ids)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold_ids), 5)))
    expected = str(case.get("expected_status", gold.get("expected_status")))
    grounded = bool(found)
    outcomes = _outcomes(expected, result.status, grounded)
    cited = {eid for claim in result.claims for eid in claim.evidence_ids}
    cited_chunks = {e.chunk_id for e in result.evidence if e.evidence_id in cited}
    cited_found = cited_chunks & gold_ids
    trace = result.trace
    trace_complete = all(k in trace for k in ("query", "mode", "top_k", "version_tuple", "components_called", "evidence_ids", "scores", "provider_called", "provider_returned", "model", "latency_ms", "schema_error", "grounding_pass", "error"))
    return {
        "case_id": case["case_id"], "family_id": case.get("family_id"), "source_family": case.get("source_family"), "claim_family": case.get("claim_family"), "mutation_type": case.get("mutation_type"), "annotation_provenance": case.get("annotation_provenance"),
        "query": case["query"], "mode": result.mode, "expected_status": expected, "status": result.status, "status_exact": result.status == expected,
        "answerable_success": outcomes["answerable_success"], "abstention_success": outcomes["abstention_success"], "task_success": outcomes["task_success"],
        "hit": grounded, "rr": 1.0 / rank if rank else 0.0, "ndcg_at_5": dcg / ideal if ideal else 0.0,
        "citation_precision": len(cited_found) / len(cited_chunks) if cited_chunks else 0.0,
        "citation_recall": len(cited_found) / len(gold_ids) if gold_ids else 0.0,
        "claim_grounding": build_claim_evidence(result), "quote_grounding": None,
        "stale_selected": bool(result.status == "ANSWERED" and any(not h.chunk.active_at(date.fromisoformat(str(case["as_of"]))) for h in result.hits)),
        "provider_called": bool(trace.get("provider_called")), "provider_returned": bool(trace.get("provider_returned")), "schema_error": trace.get("schema_error"),
        "grounding_pass": trace.get("grounding_pass"), "trace_complete": trace_complete, "manifest_complete": bool(trace.get("version_tuple")),
        "evidence_ids": list(trace.get("evidence_ids", [])), "scores": list(trace.get("scores", [])), "components_called": list(trace.get("components_called", [])),
        "version_tuple": list(trace.get("version_tuple", [])), "error": trace.get("error"), "failure_code": result.failure_code,
    }


def evaluate(manifest_path: str | Path, cases: list[dict[str, Any]], gold: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    manifest = load_r35_manifest(manifest_path)
    mapping: dict[str, str] = {}
    if mode == "mutation":
        manifest, mapping = mutate_manifest(manifest)
    runtime = R35Retriever(manifest)
    gold_by_id = {row["case_id"]: row for row in gold}
    rows = [_case_row(_run_case(runtime, case, gold_by_id[case["case_id"]], mode, i, mapping), case, gold_by_id[case["case_id"]], mapping) for i, case in enumerate(cases)]
    answerable = [r for r in rows if r["expected_status"] == "ANSWERED"]
    abstention = [r for r in rows if r["expected_status"] != "ANSWERED"]
    def count(group: Sequence[dict[str, Any]], key: str) -> dict[str, Any]:
        return ratio(sum(bool(row.get(key)) for row in group), len(group))
    status_counts = {s: sum(row["status"] == s for row in rows) for s in CANONICAL_STATUSES}
    confusion = {e: {a: sum(row["expected_status"] == e and row["status"] == a for row in rows) for a in CANONICAL_STATUSES} for e in CANONICAL_STATUSES}
    metrics = {
        "recall_at_5": count(answerable, "hit"), "mrr": mean_score([row["rr"] for row in answerable]),
        "ndcg_at_5": mean_score([row["ndcg_at_5"] for row in answerable]),
        "answerable_success": count(answerable, "answerable_success"), "safe_abstention_accuracy": count(abstention, "abstention_success"),
        "citation_precision": mean_score([row["citation_precision"] for row in answerable]),
        "citation_recall": mean_score([row["citation_recall"] for row in answerable]),
        "quote_grounding": ratio(sum(row["quote_grounding"] is True for row in rows), sum(row["provider_returned"] for row in rows)),
        "claim_grounding": count(answerable, "claim_grounding"), "stale_selection_rate": count(rows, "stale_selected"),
        "trace_completeness": count(rows, "trace_complete"), "manifest_completeness": count(rows, "manifest_complete"),
        "provider_called": ratio(sum(row["provider_called"] for row in rows), len(rows)),
        "provider_returned": ratio(sum(row["provider_returned"] for row in rows), len(rows)),
        "provider_return_rate": ratio(sum(row["provider_returned"] for row in rows), sum(row["provider_called"] for row in rows)),
        "schema_valid_rate": ratio(sum(not row["schema_error"] for row in rows if row["provider_returned"]), sum(row["provider_returned"] for row in rows)),
        "grounding_pass_rate": ratio(sum(row["grounding_pass"] is True for row in rows), sum(row["provider_returned"] for row in rows)),
        "canonical_status_exact": count(rows, "status_exact"), "overall_task_success": count(rows, "task_success"),
    }
    failures = sum(bool(row["failure_code"] or row["error"] or row["schema_error"]) for row in rows)
    safe = metrics["safe_abstention_accuracy"]["value"]
    eligible = failures == 0 and safe >= 0.85 and metrics["stale_selection_rate"]["value"] == 0.0 and metrics["trace_completeness"]["value"] == 1.0 and metrics["manifest_completeness"]["value"] == 1.0
    return {
        "report_version": "r3.5.dev-eval.v1", "split": "dev-v2", "mode": mode, "N": len(rows), "unique_case_N": len({r["case_id"] for r in rows}), "observation_N": len(rows),
        "answerable_N": len(answerable), "abstention_N": len(abstention), "manifest_version_tuple": list(manifest.version_tuple), "strategy_checksum": manifest.strategy_checksum,
        "failures": failures, "selection_eligible": eligible, "stage1_retrieval_eligible": retrieval_screen_pass({"metrics": metrics, "failures": failures}), "metrics": metrics, "status_counts": status_counts, "status_confusion": confusion, "by_case": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/r3_corpus_v1/manifest-r3_5.json")
    parser.add_argument("--input", default="eval/datasets/r3_5/dev-v2-inputs.jsonl")
    parser.add_argument("--gold", default="eval/datasets/r3_5/dev-v2-gold.jsonl")
    parser.add_argument("--mode", choices=R35_MODES, default="hybrid_reranker")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    report = evaluate(args.manifest, load_jsonl(args.input), load_jsonl(args.gold), args.mode)
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
