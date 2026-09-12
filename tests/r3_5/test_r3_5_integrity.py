from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

import agent.r3_5_rag as r35
from agent.r3_5_rag import R35Retriever, load_r35_manifest, mutate_manifest
from eval.r3_5_eval import evaluate, load_jsonl
from eval.r3_5_select import PRODUCTION_CANDIDATES, choose_provisional, phase_one_eligible
from eval.r3_5_external_eval import evaluate_external, main as external_main
from agent.interactive_runtime import ModelCallError


ROOT = Path(__file__).parents[2]
MANIFEST = ROOT / "data/r3_corpus_v1/manifest-r3_5.json"
DATA = ROOT / "eval/datasets/r3_5"


def test_strategy_and_component_matrix_are_versioned():
    manifest = load_r35_manifest(MANIFEST)
    assert manifest.evidence_threshold == 0.02
    assert manifest.time_strategy == "source-metadata-effective-interval.v1"
    assert manifest.strategy_checksum in manifest.version_tuple
    assert set(manifest.component_matrix) == set(r35.R35_MODES)


def test_no_retrieval_does_not_construct_retrievers(monkeypatch):
    class Explode:
        def __init__(self, *args, **kwargs):
            raise AssertionError("disabled provider was constructed")

    monkeypatch.setattr(r35, "BM25Index", Explode)
    monkeypatch.setattr(r35, "DenseIndex", Explode)
    result = R35Retriever.from_manifest(MANIFEST).retrieve("任意演示问题", as_of=date(2026, 6, 1), mode="no_retrieval")
    assert result.status == "NO_HITS"
    assert result.trace["components_called"] == []


def test_random_does_not_call_bm25_or_dense(monkeypatch):
    class Explode:
        def __init__(self, *args, **kwargs):
            raise AssertionError("disabled provider was constructed")

    monkeypatch.setattr(r35, "BM25Index", Explode)
    monkeypatch.setattr(r35, "DenseIndex", Explode)
    result = R35Retriever.from_manifest(MANIFEST).retrieve("演示问题", as_of=date(2026, 6, 1), mode="random", seed=3)
    assert result.trace["components_called"] == ["seeded_random"]
    assert "bm25" not in result.trace["components_called"]
    assert "dense" not in result.trace["components_called"]


def test_random_with_chunks_is_answered_without_abstention(monkeypatch):
    class FakeBM25:
        def __init__(self, chunks):
            raise AssertionError("random baseline called BM25")

    class FakeDense:
        def __init__(self, chunks, model_name):
            raise AssertionError("random baseline called dense")

    monkeypatch.setattr(r35, "BM25Index", FakeBM25)
    monkeypatch.setattr(r35, "DenseIndex", FakeDense)
    result = R35Retriever.from_manifest(MANIFEST).retrieve("无关查询", as_of=date(2035, 1, 1), mode="random", seed=3)
    assert result.status == "ANSWERED"
    assert result.trace["abstention_applied"] is False
    assert result.trace["conflict_set"] == []


def test_dense_does_not_call_bm25(monkeypatch):
    class ExplodeBM25:
        def __init__(self, *args, **kwargs):
            raise AssertionError("dense baseline called BM25")

    class FakeDense:
        def __init__(self, chunks, model_name):
            self.chunks = tuple(chunks)

        def rank(self, query, chunks=None):
            return [(chunk, 0.9) for chunk in tuple(chunks or self.chunks)]

    monkeypatch.setattr(r35, "BM25Index", ExplodeBM25)
    monkeypatch.setattr(r35, "DenseIndex", FakeDense)
    result = R35Retriever.from_manifest(MANIFEST).retrieve("演示问题", as_of=date(2026, 6, 1), mode="dense")
    assert result.trace["components_called"] == ["dense"]


def test_hybrid_calls_declared_components(monkeypatch):
    class FakeBM25:
        def __init__(self, chunks):
            self.chunks = tuple(chunks)

        def rank(self, query, chunks=None):
            return [(chunk, 1.0) for chunk in tuple(chunks or self.chunks)]

    class FakeDense:
        def __init__(self, chunks, model_name):
            self.chunks = tuple(chunks)

        def rank(self, query, chunks=None):
            return [(chunk, 0.9) for chunk in tuple(chunks or self.chunks)]

    monkeypatch.setattr(r35, "BM25Index", FakeBM25)
    monkeypatch.setattr(r35, "DenseIndex", FakeDense)
    result = R35Retriever.from_manifest(MANIFEST).retrieve("演示问题", as_of=date(2026, 6, 1), mode="hybrid_reranker")
    assert {"bm25", "dense", "rrf", "reranker"} <= set(result.trace["components_called"])


def test_mutation_changes_ids_and_order_deterministically():
    manifest = load_r35_manifest(MANIFEST)
    mutated, mapping = mutate_manifest(manifest, seed=17)
    assert mutated.strategy_checksum == manifest.strategy_checksum
    assert mutated.version_tuple != manifest.version_tuple
    assert set(mapping) == {chunk.chunk_id for chunk in manifest.chunks}
    assert {chunk.chunk_id for chunk in mutated.chunks} == set(mapping.values())


def test_dev_v2_provenance_and_manifest_are_complete():
    manifest = json.loads((DATA / "dataset-manifest.json").read_text(encoding="utf-8"))
    for name in ("dev-v2-inputs.jsonl", "dev-v2-gold.jsonl", "mutation-inputs.jsonl", "mutation-gold.jsonl"):
        path = DATA / name
        assert path.exists()
        rows = load_jsonl(path)
        assert len(rows) == manifest["files"][name]["lines"]
        assert all({"family_id", "source_family", "claim_family", "mutation_type", "annotation_provenance"} <= set(row) for row in rows)
    assert len(load_jsonl(DATA / "dev-v2-inputs.jsonl")) == 32
    assert len(load_jsonl(DATA / "mutation-inputs.jsonl")) == 12


def test_evaluator_reports_denominators_and_wilson_ci(tmp_path: Path):
    report = evaluate(MANIFEST, load_jsonl(DATA / "dev-v2-inputs.jsonl"), load_jsonl(DATA / "dev-v2-gold.jsonl"), "no_retrieval")
    assert report["unique_case_N"] == report["observation_N"] == 32
    assert report["metrics"]["safe_abstention_accuracy"]["denominator"] > 0
    assert len(report["metrics"]["safe_abstention_accuracy"]["wilson95"]) == 2
    assert all("components_called" in row and "version_tuple" in row for row in report["by_case"])


def _fake_report(score: float, components: list[str]) -> dict:
    return {
        "failures": 0,
        "metrics": {
            "recall_at_5": {"value": 1.0}, "answerable_success": {"value": 1.0},
            "stale_selection_rate": {"numerator": 0}, "trace_completeness": {"value": 1.0},
            "manifest_completeness": {"value": 1.0}, "overall_task_success": {"value": score},
        },
        "by_case": [{"components_called": components, "latency_ms": 10.0}],
    }


def test_selection_excludes_diagnostic_modes_and_prefers_fewer_components():
    assert PRODUCTION_CANDIDATES == {"bm25", "dense", "hybrid_no_reranker", "hybrid_reranker"}
    reports = {
        "no_retrieval": _fake_report(1.0, []), "random": _fake_report(1.0, ["seeded_random"]),
        "oracle": _fake_report(1.0, ["gold"]), "remove_abstention": _fake_report(1.0, ["bm25", "dense", "rrf"]),
        "bm25": _fake_report(0.8, ["bm25"]),
        "hybrid_no_reranker": _fake_report(0.9, ["bm25", "dense", "rrf"]),
        "hybrid_reranker": _fake_report(0.9, ["bm25", "dense", "rrf", "reranker"]),
    }
    assert all(not phase_one_eligible(mode, reports[mode]) for mode in ("no_retrieval", "random", "oracle", "remove_abstention"))
    provisional = choose_provisional(reports)
    assert provisional[0] == "hybrid_no_reranker"
    assert not set(provisional) & {"no_retrieval", "random", "oracle", "remove_abstention"}


def test_selection_tie_prefers_fewer_components_then_latency():
    reports = {
        "bm25": _fake_report(0.9, ["bm25"]),
        "dense": _fake_report(0.9, ["dense"]),
    }
    assert choose_provisional(reports)[0] == "dense" or choose_provisional(reports)[0] == "bm25"
    reports["dense"]["by_case"][0]["components_called"] = ["dense", "extra"]
    assert choose_provisional(reports)[0] == "bm25"


def test_external_default_does_not_call_provider_and_keeps_full_trace():
    cases = load_jsonl(DATA / "dev-v2-inputs.jsonl")[:2]
    gold = load_jsonl(DATA / "dev-v2-gold.jsonl")[:2]
    report = evaluate_external(MANIFEST, cases, gold, mode="bm25")
    assert report["live"] is False
    assert report["phase2"]["executed"] is False
    assert all(row["provider_called"] is False and row["provider_returned"] is False for row in report["rows"])
    required = {"query", "mode", "scores", "version_tuple", "evidence_ids", "components_called", "provider_called", "provider_returned", "model", "latency_ms", "schema_error", "grounding_pass", "error", "status", "answer", "claims"}
    assert all(required <= set(row) and row["trace_complete"] for row in report["rows"])


def test_external_mock_provider_records_grounded_trace():
    cases = load_jsonl(DATA / "dev-v2-inputs.jsonl")[:2]
    gold = load_jsonl(DATA / "dev-v2-gold.jsonl")[:2]

    class FakeConfig:
        model = "fake-test-model"
        timeout_seconds = 1.0
        max_retries = 0

    def provider(prompt, schema):
        packet = json.loads(prompt.split("\n", 1)[1])
        evidence = packet["evidence"][0]
        return schema(decision="ANSWERED", answer="依据证据回答", claims=[{
            "claim_id": "fake-claim", "text": "依据证据回答", "evidence_ids": [evidence["evidence_id"]], "supporting_quote": evidence["text"],
        }])

    report = evaluate_external(MANIFEST, cases, gold, mode="bm25", provider=provider, config=FakeConfig())
    assert report["phase2"]["executed"] is True
    assert report["metrics"]["provider_success"]["numerator"] == report["metrics"]["provider_success"]["denominator"]
    assert all(row["provider_called"] and row["provider_returned"] and row["grounding_pass"] is True for row in report["rows"] if row["provider_returned"])


def _answerable_case():
    cases = load_jsonl(DATA / "dev-v2-inputs.jsonl")
    gold = load_jsonl(DATA / "dev-v2-gold.jsonl")
    expected = {row["case_id"]: row for row in gold}
    case = next(row for row in cases if expected[row["case_id"]]["expected_status"] == "ANSWERED")
    return [case], [expected[case["case_id"]]]


def _fake_grounded_provider(mode: str):
    calls = []

    def provider(prompt, schema):
        packet = json.loads(prompt.rsplit("\n", 1)[-1])
        evidence = packet["evidence"][0]
        calls.append(prompt)
        quote = evidence["text"] if mode == "valid" or (mode == "repair" and len(calls) > 1) else "not an exact quote"
        return schema(decision="ANSWERED", answer="依据证据回答", claims=[{
            "claim_id": "fake-claim", "text": "依据证据回答", "evidence_ids": [evidence["evidence_id"]], "supporting_quote": quote,
        }])

    return provider, calls


def test_grounded_repair_retries_once_after_quote_error_and_records_attempts():
    cases, gold = _answerable_case()
    provider, calls = _fake_grounded_provider("repair")

    class FakeConfig:
        model = "fake-test-model"
        timeout_seconds = 1.0
        max_retries = 0

    report = evaluate_external(MANIFEST, cases, gold, mode="bm25", provider=provider, config=FakeConfig())
    row = report["rows"][0]
    assert len(calls) == 2
    assert row["attempt_count"] == 2
    assert [attempt["returned"] for attempt in row["attempts"]] == [True, True]
    assert row["attempts"][0]["error"] == "GROUNDING_INVALID_QUOTE"
    assert row["attempts"][1]["error"] is None
    assert row["status"] == "ANSWERED" and row["grounding_pass"] is True


def test_grounded_repair_stops_after_two_failures():
    cases, gold = _answerable_case()
    provider, calls = _fake_grounded_provider("always-invalid")

    class FakeConfig:
        model = "fake-test-model"
        timeout_seconds = 1.0
        max_retries = 0

    report = evaluate_external(MANIFEST, cases, gold, mode="bm25", provider=provider, config=FakeConfig())
    row = report["rows"][0]
    assert len(calls) == 2
    assert row["attempt_count"] == 2
    assert len(row["attempts"]) == 2
    assert row["status"] == "FAILED"
    assert row["error"] == "GROUNDING_INVALID_QUOTE"


def test_grounded_valid_output_uses_one_attempt():
    cases, gold = _answerable_case()
    provider, calls = _fake_grounded_provider("valid")

    class FakeConfig:
        model = "fake-test-model"
        timeout_seconds = 1.0
        max_retries = 0

    report = evaluate_external(MANIFEST, cases, gold, mode="bm25", provider=provider, config=FakeConfig())
    row = report["rows"][0]
    assert len(calls) == 1
    assert row["attempt_count"] == 1
    assert len(row["attempts"]) == 1
    assert row["status"] == "ANSWERED"


def test_grounded_repair_retries_schema_error_once():
    cases, gold = _answerable_case()
    calls = []

    class FakeConfig:
        model = "fake-test-model"
        timeout_seconds = 1.0
        max_retries = 0

    def provider(prompt, schema):
        calls.append(prompt)
        if len(calls) == 1:
            return {"decision": "NOT_A_LEGAL_DECISION"}
        packet = json.loads(prompt.rsplit("\n", 1)[-1])
        evidence = packet["evidence"][0]
        return schema(decision="ANSWERED", answer="依据证据回答", claims=[{
            "claim_id": "schema-repair", "text": "依据证据回答", "evidence_ids": [evidence["evidence_id"]], "supporting_quote": evidence["text"],
        }])

    report = evaluate_external(MANIFEST, cases, gold, mode="bm25", provider=provider, config=FakeConfig())
    row = report["rows"][0]
    assert len(calls) == 2
    assert row["attempt_count"] == 2
    assert row["attempts"][0]["error"] == "MODEL_SCHEMA_INVALID"
    assert row["status"] == "ANSWERED"


def test_grounded_repair_retries_provider_error_once():
    cases, gold = _answerable_case()
    calls = []

    class FakeConfig:
        model = "fake-test-model"
        timeout_seconds = 1.0
        max_retries = 0

    def provider(prompt, schema):
        calls.append(prompt)
        if len(calls) == 1:
            raise ModelCallError("MODEL_PROVIDER_ERROR")
        packet = json.loads(prompt.rsplit("\n", 1)[-1])
        evidence = packet["evidence"][0]
        return schema(decision="ANSWERED", answer="依据证据回答", claims=[{
            "claim_id": "provider-repair", "text": "依据证据回答", "evidence_ids": [evidence["evidence_id"]], "supporting_quote": evidence["text"],
        }])

    report = evaluate_external(MANIFEST, cases, gold, mode="bm25", provider=provider, config=FakeConfig())
    row = report["rows"][0]
    assert len(calls) == 2
    assert row["attempt_count"] == 2
    assert row["attempts"][0]["returned"] is False
    assert row["attempts"][0]["error"] == "MODEL_PROVIDER_ERROR"
    assert row["status"] == "ANSWERED"


def test_grounded_repair_stops_after_two_provider_errors():
    cases, gold = _answerable_case()
    calls = []

    class FakeConfig:
        model = "fake-test-model"
        timeout_seconds = 1.0
        max_retries = 0

    def provider(prompt, schema):
        calls.append(prompt)
        raise ModelCallError("MODEL_PROVIDER_ERROR")

    report = evaluate_external(MANIFEST, cases, gold, mode="bm25", provider=provider, config=FakeConfig())
    row = report["rows"][0]
    assert len(calls) == 2
    assert row["attempt_count"] == 2
    assert len(row["attempts"]) == 2
    assert row["status"] == "FAILED"
    assert row["error"] == "MODEL_PROVIDER_ERROR"


def test_grounded_repair_preserves_first_validation_error_over_second_provider_error():
    cases, gold = _answerable_case()
    calls = []

    class FakeConfig:
        model = "fake-test-model"
        timeout_seconds = 1.0
        max_retries = 0

    def provider(prompt, schema):
        calls.append(prompt)
        if len(calls) == 1:
            packet = json.loads(prompt.rsplit("\n", 1)[-1])
            evidence = packet["evidence"][0]
            return schema(decision="ANSWERED", answer="依据证据回答", claims=[{
                "claim_id": "bad-quote", "text": "依据证据回答", "evidence_ids": [evidence["evidence_id"]], "supporting_quote": "not an exact quote",
            }])
        raise ModelCallError("MODEL_PROVIDER_ERROR")

    report = evaluate_external(MANIFEST, cases, gold, mode="bm25", provider=provider, config=FakeConfig())
    row = report["rows"][0]
    assert len(calls) == 2
    assert row["status"] == "FAILED"
    assert row["error"] == "GROUNDING_INVALID_QUOTE"
    assert row["attempts"][0]["error"] == "GROUNDING_INVALID_QUOTE"
    assert row["attempts"][1]["error"] == "MODEL_PROVIDER_ERROR"


def test_external_output_refuses_overwrite(tmp_path: Path):
    output = tmp_path / "existing.json"
    output.write_text("sentinel", encoding="utf-8")
    with pytest.raises(FileExistsError):
        external_main(["--manifest", str(MANIFEST), "--input", str(DATA / "dev-v2-inputs.jsonl"), "--gold", str(DATA / "dev-v2-gold.jsonl"), "--mode", "bm25", "--output", str(output)])
    assert output.read_text(encoding="utf-8") == "sentinel"


def test_r35_runtime_has_no_case_or_gold_literals():
    source = (ROOT / "agent/r3_5_rag.py").read_text(encoding="utf-8")
    assert "r35-dev-" not in source
    assert "gold_chunk_ids" not in source
    assert "OPENAI_API_KEY" not in source
