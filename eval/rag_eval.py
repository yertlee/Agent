"""Executable deterministic RAG manifest/retrieval check."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .evaluator import wilson_interval
from .rag_manifest import HybridRetriever, build_kb_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="M4 RAG evaluation protocol")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--query", default="")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    config = yaml.safe_load(Path(args.manifest).read_text(encoding="utf-8")) or {}
    root = Path(config.get("source_root", "docs/kb"))
    if not root.is_absolute():
        root = Path(args.manifest).resolve().parent / root
    manifest = build_kb_manifest(root, kb_version=str(config.get("kb_version", "kb.m4.v1")),
        manifest_version=str(config.get("manifest_version", "kb-manifest.m4.v1")),
        embedding_model=str(config.get("embedding_model", "deterministic-dense.v1")), tokenizer=str(config.get("tokenizer", "unicode-nfkc.v1")),
        builder_version=str(config.get("builder_version", "kb-builder.m4.v1")), rrf_constant=int(config.get("rrf_constant", 60)),
        candidate_k=int(config.get("candidate_k", 20)), score_threshold=float(config.get("score_threshold", 0.0)))
    payload = {"kb_version": manifest.kb_version, "manifest_checksum": manifest.checksum, "source_count": len(manifest.sources), "chunk_count": len(manifest.chunks), "top_k": manifest.top_k}
    expected_checksum = str(config.get("manifest_checksum", ""))
    if expected_checksum and expected_checksum != manifest.checksum:
        raise ValueError("RAG source manifest checksum mismatch")
    retriever = HybridRetriever(manifest)
    rag_cases = list(config.get("rag_cases") or [])
    if rag_cases:
        rows = []
        for case in rag_cases:
            result = retriever.retrieve(str(case["query"]))
            hit_ids = {hit.chunk_id for hit in result.hits}
            hit_pairs = {(hit.source_id, hit.chunk_id) for hit in result.hits}
            gold_chunks = {str(value) for value in case.get("gold_chunk_ids", [])}
            gold_pairs = {(str(item["source_id"]), str(item["chunk_id"])) for item in case.get("gold_evidence", [])}
            recall = bool(hit_ids & gold_chunks)
            claim_evidence = bool(gold_pairs) and gold_pairs.issubset(hit_pairs)
            rows.append({"scenario_id": str(case["scenario_id"]), "status": result.status,
                         "retrieved_chunk_ids": [hit.chunk_id for hit in result.hits],
                         "recall_at_5": recall, "claim_evidence": claim_evidence})
        n = len(rows)
        recall_n = sum(row["recall_at_5"] for row in rows)
        claim_n = sum(row["claim_evidence"] for row in rows)
        payload.update({"N": n, "cases": rows, "metrics": {
            "rag_recall_at_5": {"value": recall_n / n if n else 0.0, "numerator": recall_n, "denominator": n,
                                "wilson95": list(wilson_interval(recall_n, n))},
            "claim_evidence": {"value": claim_n / n if n else 0.0, "numerator": claim_n, "denominator": n,
                               "wilson95": list(wilson_interval(claim_n, n))},
        }})
    if args.query: payload["retrieval"] = retriever.retrieve(args.query).__dict__
    # Dataclasses are converted explicitly so report output remains canonical JSON.
    if args.query:
        result = payload["retrieval"]; payload["retrieval"] = {"status": result["status"], "query": result["query"], "top_k": result["top_k"], "conflict_set": list(result["conflict_set"]), "hits": [{"chunk_id": h.chunk_id, "source_id": h.source_id, "source_uri": h.source_uri, "score": h.score, "text": h.text, "injection_data_only": h.injection_data_only} for h in result["hits"]]}
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output: Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__": raise SystemExit(main())
