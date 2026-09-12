from __future__ import annotations

import json
from datetime import date
from pathlib import Path
import sys

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

from agent.interactive_runtime import LLMRuntimeConfig
from agent.r3_grounded_qa import R3GroundedQARuntime
from agent.r3_rag_runtime import R3Retriever
from eval.r3_external_dev_eval import main as external_eval_main
from eval.r3_eval import main as eval_main


MANIFEST = ROOT / "data/r3_corpus_v1/manifest.json"


class FakeProvider:
    def __init__(self, mode: str = "valid") -> None:
        self.mode = mode
        self.calls = 0

    def __call__(self, prompt: str, output_schema: type[object]) -> object:
        self.calls += 1
        packet = json.loads(prompt.split("\n", 1)[1])
        evidence = packet["evidence"][0]
        evidence_id = evidence["evidence_id"] if self.mode != "forged_evidence" else "ev-forged"
        quote = evidence["text"][:12] if self.mode != "forged_quote" else "quote not present"
        if self.mode == "insufficient":
            return {"decision": "INSUFFICIENT", "answer": "", "claims": []}
        if self.mode == "answered_empty":
            return {"decision": "ANSWERED", "answer": "", "claims": [{"claim_id": "claim-1", "text": "演示资料回答", "evidence_ids": [evidence_id], "supporting_quote": quote}]}
        return {"decision": "ANSWERED", "answer": "演示资料回答", "claims": [{"claim_id": "claim-1", "text": "演示资料回答", "evidence_ids": [evidence_id], "supporting_quote": quote}]}


def _runtime(provider: FakeProvider) -> R3GroundedQARuntime:
    retriever = R3Retriever.from_manifest(MANIFEST, require_dense=False)
    config = LLMRuntimeConfig.from_environment(db_path=".", env={"ECOMMERCE_DB_PATH": "."})
    return R3GroundedQARuntime(retriever, config=config, provider=provider)


def test_valid_claims_and_quotes_are_returned_with_safe_trace():
    provider = FakeProvider()
    result = _runtime(provider).answer("演示商品电池的有限保修期多久", as_of=date(2026, 6, 1), mode="bm25")
    assert result.status == "ANSWERED"
    assert provider.calls == 1
    assert result.claims[0].evidence_ids[0].startswith("ev-")
    assert result.trace["provider_called"] is True
    assert result.trace["provider_returned"] is True
    assert result.trace["retrieval_version_tuple"]
    assert "api_key" not in json.dumps(result.trace).lower()


def test_forged_evidence_fails_closed():
    result = _runtime(FakeProvider("forged_evidence")).answer("演示商品电池的有限保修期多久", as_of=date(2026, 6, 1), mode="bm25")
    assert result.status == "FAILED"
    assert result.trace["error"] == "GROUNDING_INVALID_EVIDENCE"


def test_forged_quote_fails_closed():
    result = _runtime(FakeProvider("forged_quote")).answer("演示商品电池的有限保修期多久", as_of=date(2026, 6, 1), mode="bm25")
    assert result.status == "FAILED"
    assert result.trace["error"] == "GROUNDING_INVALID_QUOTE"


def test_model_can_second_refuse_with_insufficient_decision():
    provider = FakeProvider("insufficient")
    result = _runtime(provider).answer("演示商品电池的有限保修期多久", as_of=date(2026, 6, 1), mode="bm25")
    assert result.status == "WEAK_EVIDENCE"
    assert result.claims == ()
    assert result.trace["provider_called"] is True
    assert result.trace["grounding_pass"] is True
    assert result.answer == "现有项目演示证据不足，无法可靠回答。"


def test_answered_with_empty_answer_fails_contract():
    result = _runtime(FakeProvider("answered_empty")).answer("演示商品电池的有限保修期多久", as_of=date(2026, 6, 1), mode="bm25")
    assert result.status == "FAILED"
    assert result.trace["error"] == "GROUNDING_ANSWERED_WITHOUT_ANSWER"


def test_no_hits_never_calls_model():
    provider = FakeProvider()
    result = _runtime(provider).answer("演示手机摄像头像素是多少", as_of=date(2026, 6, 1), mode="bm25")
    assert result.status == "NO_HITS"
    assert provider.calls == 0
    assert result.trace["provider_called"] is False


def test_evaluator_output_creates_nested_parent(tmp_path: Path):
    output = tmp_path / "nested" / "r3" / "report.json"
    assert eval_main(["--mode", "no_retrieval", "--output", str(output)]) == 0
    assert output.exists()


def test_external_dev_output_has_exact_status_and_provider_fields(tmp_path: Path):
    output = tmp_path / "nested" / "external" / "report.json"
    assert external_eval_main(["--output", str(output)]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["N"] == 24
    assert report["provider_called"] == 0
    assert {"status_accuracy", "status_exact_accuracy", "answerable_success", "abstention_accuracy", "overall_task_success", "provider_return_rate", "schema_valid_rate", "grounding_pass_rate", "status_counts", "status_confusion"} <= set(report)
    assert all({"expected_status", "status", "status_exact", "provider_called", "provider_returned", "error_code", "schema_error", "grounding_pass", "answerable_success", "abstention_success", "task_success"} <= set(row) for row in report["rows"])
