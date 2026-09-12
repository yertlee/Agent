"""Executable R3-A/R3-B development evaluator.

It consumes only the separately stored dev input and dev gold.  It never calls
an answer model and never creates validation or heldout gold.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

from agent.r3_rag import (
    TOP_K,
    Claim,
    EvidenceRef,
    RetrievalHit,
    RetrievalResult,
    build_claim_evidence,
    load_cases,
)
from agent.r3_rag_runtime import R3Retriever

MODES = (
    "no_retrieval", "bm25", "dense", "hybrid_no_reranker", "random",
    "oracle", "hybrid_reranker",
)
SAFE_ABSTENTION_STATUSES = frozenset({"NO_HITS", "WEAK_EVIDENCE", "CONFLICT", "STALE_ONLY"})
CANONICAL_STATUSES = ("ANSWERED", "NO_HITS", "WEAK_EVIDENCE", "CONFLICT", "STALE_ONLY", "FAILED")


def evaluate_case_outcome(expected_status: str, actual_status: str, grounded_pass: bool) -> dict[str, bool]:
    is_answerable = expected_status == "ANSWERED"
    is_abstention = not is_answerable
    answerable_success = is_answerable and actual_status == "ANSWERED" and grounded_pass
    abstention_success = is_abstention and actual_status in SAFE_ABSTENTION_STATUSES
    return {
        "answerable_success": answerable_success,
        "abstention_success": abstention_success,
        "task_success": answerable_success or abstention_success,
    }


def _oracle_result(retriever: R3Retriever, case: dict[str, Any], gold: dict[str, Any]) -> RetrievalResult:
    expected = case.get("expected_status", gold.get("expected_status"))
    trace = {"version_tuple": retriever.manifest.version_tuple, "oracle": True, "query": case["query"], "top_k": TOP_K, "rewrite": {"status": "NOT_APPLIED"}}
    if expected != "ANSWERED":
        return RetrievalResult(expected, case["query"], "oracle", trace=trace)
    chunks = {c.chunk_id: c for c in retriever.manifest.chunks}
    hits = [RetrievalHit(chunks[cid], 1.0) for cid in gold.get("gold_chunk_ids", []) if cid in chunks]
    evidence = retriever._evidence(hits)
    claims = [Claim("oracle-claim", "oracle context", tuple(e.evidence_id for e in evidence))] if evidence else []
    return RetrievalResult("ANSWERED" if evidence else "NO_HITS", case["query"], "oracle", hits, evidence, claims, trace=trace)


def _random_result(retriever: R3Retriever, case: dict[str, Any], seed: int) -> RetrievalResult:
    rng = random.Random(seed)
    active = [c for c in retriever.manifest.chunks if c.active_at(date.fromisoformat(case["as_of"]))]
    rng.shuffle(active)
    hits = [RetrievalHit(c, 1.0) for c in active[:TOP_K]]
    evidence = retriever._evidence(hits)
    return RetrievalResult("ANSWERED" if hits else "NO_HITS", case["query"], "random", hits, evidence, trace={"version_tuple": retriever.manifest.version_tuple, "random_seed": seed, "query": case["query"], "top_k": TOP_K, "rewrite": {"status": "NOT_APPLIED"}})


def run_case(retriever: R3Retriever, case: dict[str, Any], gold: dict[str, Any], mode: str, index: int) -> RetrievalResult:
    if mode == "oracle":
        return _oracle_result(retriever, case, gold)
    if mode == "random":
        return _random_result(retriever, case, index + 71)
    return retriever.retrieve(case["query"], as_of=date.fromisoformat(case["as_of"]), mode=mode)


def _case_metrics(result: RetrievalResult, case: dict[str, Any], gold: dict[str, Any]) -> dict[str, Any]:
    gold_ids = set(gold.get("gold_chunk_ids", []))
    ranked = [h.chunk.chunk_id for h in result.hits]
    found = [cid for cid in ranked if cid in gold_ids]
    rank = next((i + 1 for i, cid in enumerate(ranked) if cid in gold_ids), None)
    ndcg = 0.0
    if gold_ids:
        dcg = sum((1.0 / math.log2(i + 2)) for i, cid in enumerate(ranked) if cid in gold_ids)
        ideal = sum((1.0 / math.log2(i + 2)) for i in range(min(len(gold_ids), TOP_K)))
        ndcg = dcg / ideal if ideal else 0.0
    expected = case.get("expected_status", gold.get("expected_status"))
    status_ok = result.status == expected
    grounded_pass = bool(found)
    outcomes = evaluate_case_outcome(expected, result.status, grounded_pass)
    answer_ok = outcomes["answerable_success"]
    cited_ids = {eid for claim in result.claims for eid in claim.evidence_ids}
    cited_chunk_ids = {e.chunk_id for e in result.evidence if e.evidence_id in cited_ids}
    cited_found = cited_chunk_ids & gold_ids
    citation_precision = len(cited_found) / len(cited_chunk_ids) if cited_chunk_ids else 0.0
    citation_recall = len(cited_found) / len(gold_ids) if gold_ids else 0.0
    stale_selected = any(not h.chunk.active_at(date.fromisoformat(case["as_of"])) for h in result.hits) and result.status == "ANSWERED"
    return {
        "case_id": case["case_id"], "status": result.status, "expected_status": expected,
        "status_correct": status_ok, "answer_correct": answer_ok,
        "grounded_pass": grounded_pass, **outcomes,
        "hit": bool(found), "rr": 1.0 / rank if rank else 0.0, "ndcg_at_5": ndcg,
        "citation_precision": citation_precision, "citation_recall": citation_recall,
        "faithful": build_claim_evidence(result), "cited_count": len(cited_chunk_ids), "stale_selected": stale_selected,
        "trace_complete": all(k in result.trace for k in ("version_tuple", "query", "top_k", "rewrite")),
        "manifest_complete": bool(result.trace.get("version_tuple")),
        "failure_code": result.failure_code,
    }


def evaluate(retriever: R3Retriever, cases: list[dict[str, Any]], gold: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    by_id = {g["case_id"]: g for g in gold}
    rows = [_case_metrics(run_case(retriever, case, by_id[case["case_id"]], mode, i), case, by_id[case["case_id"]]) for i, case in enumerate(cases)]
    answerable = [row for row in rows if row["expected_status"] == "ANSWERED"]
    abstention = [row for row in rows if row["expected_status"] != "ANSWERED"]
    n = len(rows)
    mean = lambda group, key: sum(float(row[key]) for row in group) / len(group) if group else 0.0
    status_counts = {status: sum(1 for row in rows if row["status"] == status) for status in CANONICAL_STATUSES}
    confusion = {expected: {actual: sum(1 for row in rows if row["expected_status"] == expected and row["status"] == actual) for actual in CANONICAL_STATUSES} for expected in CANONICAL_STATUSES}
    return {
        "mode": mode, "N": n, "answerable_N": len(answerable), "abstention_N": len(abstention),
        "retrieval_denominator": len(answerable), "abstention_denominator": len(abstention),
        "recall_at_5": mean(answerable, "hit"), "mrr": mean(answerable, "rr"), "ndcg_at_5": mean(answerable, "ndcg_at_5"),
        "status_exact_accuracy": mean(rows, "status_correct"),
        "answerable_success": mean(answerable, "answerable_success"),
        "abstention_success": mean(abstention, "abstention_success"),
        "overall_task_success": mean(rows, "task_success"),
        "answer_correctness": mean(answerable, "answer_correct"), "citation_precision": mean(answerable, "citation_precision"),
        "citation_recall": mean(answerable, "citation_recall"), "deterministic_claim_faithfulness": mean(answerable, "faithful"),
        "abstention_accuracy": mean(abstention, "abstention_success"), "stale_document_selection_rate": mean(rows, "stale_selected"),
        "trace_completeness": mean(rows, "trace_complete"), "manifest_checksum_completeness": mean(rows, "manifest_complete"),
        "failures": sum(1 for row in rows if row["failure_code"]),
        "status_counts": status_counts, "status_confusion": confusion,
        "by_case": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/r3_corpus_v1/manifest.json")
    parser.add_argument("--input", default="eval/datasets/r3/dev-inputs.jsonl")
    parser.add_argument("--gold", default="eval/datasets/r3/dev-gold.jsonl")
    parser.add_argument("--mode", choices=MODES, default="hybrid_reranker")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    cases = load_cases(args.input)
    gold = load_cases(args.gold)
    retriever = R3Retriever.from_manifest(args.manifest, require_dense=args.mode in {"dense", "hybrid_no_reranker", "hybrid_reranker"})
    result = evaluate(retriever, cases, gold, args.mode)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
