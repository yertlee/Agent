from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
import sys

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

from agent.r3_rag import DenseModelUnavailable, R3Retriever, RetrievalResult, build_claim_evidence, load_manifest, mutate_chunk_ids
from eval.r3_eval import evaluate, evaluate_case_outcome, load_cases


MANIFEST = ROOT / "data/r3_corpus_v1/manifest.json"
INPUT = ROOT / "eval/datasets/r3/dev-inputs.jsonl"
GOLD = ROOT / "eval/datasets/r3/dev-gold.jsonl"


def test_frozen_corpus_and_stable_ids():
    manifest = load_manifest(MANIFEST)
    assert manifest.top_k == 5
    assert len(manifest.chunks) == 24
    assert len({chunk.chunk_id for chunk in manifest.chunks}) == len(manifest.chunks)
    assert all(chunk.text_hash for chunk in manifest.chunks)
    assert manifest.manifest_checksum


def test_dense_is_real_or_explicitly_unavailable():
    manifest = load_manifest(MANIFEST)
    try:
        retriever = R3Retriever.from_manifest(MANIFEST, require_dense=True)
    except DenseModelUnavailable as exc:
        assert "DENSE_MODEL_UNAVAILABLE" in str(exc)
        return
    assert retriever.dense is not None
    assert retriever.dense.model_name == manifest.embedding_model
    assert getattr(retriever.dense, "vectors", None) is not None


def test_target_and_safety_states():
    retriever = R3Retriever.from_manifest(MANIFEST, require_dense=False)
    answered = retriever.retrieve("演示商品电池的有限保修期多久", as_of=date(2026, 6, 1), mode="bm25")
    stale = retriever.retrieve("会员历史演示政策退货窗口是多少", as_of=date(2026, 6, 1), mode="bm25")
    no_hits = retriever.retrieve("演示平台仓库今天的库存数量是多少", as_of=date(2026, 6, 1), mode="bm25")
    conflict = retriever.retrieve("标准商品退货窗口的演示资料是否存在互相冲突的版本", as_of=date(2025, 6, 1), mode="bm25")
    assert answered.status == "ANSWERED"
    assert stale.status == "STALE_ONLY"
    assert no_hits.status == "NO_HITS"
    assert conflict.status == "CONFLICT"
    assert build_claim_evidence(answered)


def test_active_evidence_wins_over_historical_score_for_current_query():
    retriever = R3Retriever.from_manifest(MANIFEST, require_dense=False)
    result = retriever.retrieve("当前演示标准商品 15 个自然日退货窗口", as_of=date(2026, 6, 1), mode="bm25")
    assert result.status == "ANSWERED"


def test_dev_and_validation_datasets_are_separate_and_manifested():
    assert INPUT.exists() and GOLD.exists()
    validation_input = ROOT / "eval/datasets/r3/validation-inputs.jsonl"
    validation_gold = ROOT / "eval/datasets/r3/validation-gold.jsonl"
    assert validation_input.exists() and validation_gold.exists()
    assert INPUT != validation_input and GOLD != validation_gold
    manifest = json.loads((ROOT / "eval/datasets/r3/dataset-manifest.json").read_text(encoding="utf-8"))
    for path in (INPUT, GOLD, validation_input, validation_gold):
        key = path.name
        assert key in manifest["files"]
        raw = path.read_bytes()
        assert manifest["files"][key]["sha256"] == hashlib.sha256(raw).hexdigest()
        assert manifest["files"][key]["lines"] == len(path.read_text(encoding="utf-8").splitlines())
    retriever = R3Retriever.from_manifest(MANIFEST, require_dense=False)
    result = evaluate(retriever, load_cases(INPUT), load_cases(GOLD), "bm25")
    assert result["N"] == 24
    assert result["answerable_N"] == 16
    assert result["abstention_N"] == 8
    assert result["retrieval_denominator"] == 16
    assert "by_case" in result


def test_oracle_is_perfect_for_answerable_and_returns_gold_status_for_abstentions():
    retriever = R3Retriever.from_manifest(MANIFEST, require_dense=False)
    result = evaluate(retriever, load_cases(INPUT), load_cases(GOLD), "oracle")
    assert result["recall_at_5"] == 1.0
    assert result["mrr"] == 1.0
    assert result["ndcg_at_5"] == 1.0
    assert result["abstention_accuracy"] == 1.0


def test_empty_claims_are_not_vacuously_faithful():
    result = RetrievalResult("NO_HITS", "unknown", "bm25")
    assert build_claim_evidence(result) is False


def test_failed_is_not_a_successful_abstention():
    failed = evaluate_case_outcome("NO_HITS", "FAILED", False)
    safe = evaluate_case_outcome("NO_HITS", "WEAK_EVIDENCE", False)
    assert failed["abstention_success"] is False
    assert failed["task_success"] is False
    assert safe["abstention_success"] is True


def test_chunk_id_and_document_order_mutation_is_deterministic():
    manifest = load_manifest(MANIFEST)
    mutated = mutate_chunk_ids(manifest, seed=17)
    assert mutated.manifest_checksum != manifest.manifest_checksum
    assert {c.chunk_id for c in mutated.chunks} != {c.chunk_id for c in manifest.chunks}
    assert [c.chunk_id for c in mutated.chunks] == [c.chunk_id for c in mutate_chunk_ids(manifest, seed=17).chunks]
