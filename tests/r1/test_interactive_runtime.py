from __future__ import annotations

import sqlite3
import json
from pathlib import Path

import pytest

from agent.interactive_runtime import (
    ClaimV1,
    InteractiveRuntime,
    IntentV1,
    LLMRuntimeConfig,
    ModelResult,
    PlanCandidateV1,
    ResponseV1,
    UnsupportedExecutionMode,
)
from eval.harness.contracts import ExecutionMode


def _db(tmp_path: Path) -> Path:
    path = tmp_path / "orders.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE orders (order_id TEXT PRIMARY KEY, phone_last4 TEXT, product_name TEXT, amount REAL, order_status TEXT, pay_status TEXT, created_at TEXT, can_apply_aftersales INTEGER)")
    conn.execute("INSERT INTO orders VALUES ('990000000001','1234','fixture',1.0,'PAID','PAID','2026-01-01T00:00:00Z',1)")
    conn.commit(); conn.close()
    return path


class Provider:
    def __init__(self, *, invalid: bool = False, timeout: bool = False):
        self.invalid, self.timeout, self.calls = invalid, timeout, []

    def __call__(self, prompt: str, schema: type):
        self.calls.append(schema.__name__)
        if self.timeout:
            raise TimeoutError("provider timeout")
        if self.invalid:
            return {"not": "the contract"}
        if schema is IntentV1:
            return ModelResult(IntentV1(intent="ORDER_READ", order_id="990000000001", phone_last4="1234", confidence=0.99))
        if schema is PlanCandidateV1:
            return ModelResult(PlanCandidateV1(tool_ref="order/get_info@v1", order_id="990000000001", phone_last4="1234"))
        if schema is ResponseV1:
            facts_text, _, evidence_hash = prompt.split("facts_json=", 1)[1].partition(" evidence_hash=")
            facts = json.loads(facts_text)
            key = next(iter(facts))
            return ModelResult(ResponseV1(message_code="ORDER_FACTS_V1", claim_fields=(key,), evidence_ref=evidence_hash))
        return ModelResult(ResponseV1(message_code="ORDER_FACTS_V1", claim_fields=("order_status",), evidence_ref="0" * 64))


def test_execution_mode_rejects_harness_modes(tmp_path: Path):
    runtime = InteractiveRuntime(db_path=_db(tmp_path), provider=Provider())
    for mode in (ExecutionMode.FAULT, ExecutionMode.REPLAY, ExecutionMode.RE_EXECUTE, "legacy", "evaluation"):
        with pytest.raises(UnsupportedExecutionMode):
            runtime.chat(user_id="u", message="查询订单", mode=mode)


def test_live_failure_does_not_fallback_to_simulated(tmp_path: Path):
    provider = Provider(timeout=True)
    runtime = InteractiveRuntime(db_path=_db(tmp_path), provider=provider, timeout_seconds=0.01, max_retries=0)
    result = runtime.chat(user_id="u", message="查询订单", mode=ExecutionMode.LIVE)
    assert result.status == "FAILED" and result.code == "MODEL_TIMEOUT"
    assert all(event.payload.get("mode") == "live" for event in result.trace_events if "mode" in event.payload)
    assert not any(event.event_type == "TOOL_CALLED" for event in result.trace_events)
    assert result.plan is None and result.freeze_bundle and not result.freeze_bundle.results


def test_invalid_structured_output_is_typed_failure(tmp_path: Path):
    result = InteractiveRuntime(db_path=_db(tmp_path), provider=Provider(invalid=True), timeout_seconds=1, max_retries=0).chat(user_id="u", message="查询订单", mode="live")
    assert result.status == "FAILED" and result.code == "MODEL_SCHEMA_INVALID"
    assert result.plan is None and result.freeze_bundle and not result.freeze_bundle.results


def test_intent_enum_rejects_synonyms_at_structured_boundary(tmp_path: Path):
    class SynonymProvider(Provider):
        def __call__(self, prompt: str, schema: type):
            if schema is IntentV1:
                return {"intent": "ORDER_LOOKUP", "order_id": "990000000001", "phone_last4": "1234", "confidence": 0.9}
            return super().__call__(prompt, schema)
    result = InteractiveRuntime(db_path=_db(tmp_path), provider=SynonymProvider(), timeout_seconds=1, max_retries=0).chat(user_id="u", message="查询订单", mode="live")
    assert result.status == "FAILED" and result.code == "MODEL_SCHEMA_INVALID"


def test_live_order_uses_m2_port_executor_and_freezes_trace(tmp_path: Path):
    provider = Provider()
    result = InteractiveRuntime(db_path=_db(tmp_path), provider=provider, artifact_root=tmp_path / "trace").chat(user_id="u", message="查询订单", mode=ExecutionMode.LIVE)
    assert result.status == "SUCCEEDED" and result.code == "OK"
    assert [event.event_type for event in result.trace_events] == [
        "MODEL_CALLED", "MODEL_RETURNED", "MODEL_CALLED", "MODEL_RETURNED", "PLAN_CREATED", "PLAN_VALIDATED",
        "TOOL_CALLED", "TOOL_RETURNED", "MODEL_CALLED", "MODEL_RETURNED", "RESULT_WRITTEN", "RUN_FROZEN",
    ]
    assert result.trace_path and Path(result.trace_path).is_file()
    trace_text = Path(result.trace_path).read_text(encoding="utf-8")
    assert "phone_last4" not in trace_text and "OPENAI_API_KEY" not in trace_text
    assert result.plan is not None and result.plan.tasks[0].capability_refs == ["order/read@v1"]


def test_ownership_not_found_and_clarification_are_safe(tmp_path: Path):
    class Variants(Provider):
        def __init__(self, order_id: str, phone: str | None):
            super().__init__(); self.order_id, self.phone = order_id, phone
        def __call__(self, prompt: str, schema: type):
            if schema is IntentV1:
                return ModelResult(IntentV1(intent="ORDER_READ", order_id=self.order_id, phone_last4=self.phone, confidence=.9, needs_clarification=self.phone is None))
            if schema is PlanCandidateV1:
                return ModelResult(PlanCandidateV1(tool_ref="order/get_info@v1", order_id=self.order_id, phone_last4=self.phone))
            return super().__call__(prompt, schema)
    db = _db(tmp_path)
    mismatch = InteractiveRuntime(db_path=db, provider=Variants("990000000001", "0000")).chat(user_id="u", message="x", mode="live")
    missing = InteractiveRuntime(db_path=db, provider=Variants("990000000099", "1234")).chat(user_id="u", message="x", mode="live")
    clarify = InteractiveRuntime(db_path=db, provider=Variants("", None)).chat(user_id="u", message="x", mode="live")
    assert mismatch.code == "AUTH_IDENTITY_MISMATCH" and missing.code == "ORDER_NOT_FOUND"
    assert clarify.status == "NEEDS_CLARIFICATION" and clarify.code == "CLARIFICATION_REQUIRED"
    assert mismatch.plan is not None and missing.plan is not None
    assert mismatch.freeze_bundle and len(mismatch.freeze_bundle.results) == 1
    assert missing.freeze_bundle and len(missing.freeze_bundle.results) == 1
    assert clarify.plan is None and clarify.freeze_bundle and not clarify.freeze_bundle.results
    unavailable = InteractiveRuntime(db_path=tmp_path / "missing.sqlite", provider=Provider()).chat(user_id="u", message="x", mode="live")
    assert unavailable.code == "CONFIG_MISSING" and unavailable.plan is not None
    assert unavailable.freeze_bundle and len(unavailable.freeze_bundle.results) == 1


def test_config_hash_excludes_secret_values(tmp_path: Path):
    db = _db(tmp_path)
    first = LLMRuntimeConfig.from_environment(db_path=db, env={"ECOMMERCE_DB_PATH": str(db), "OPENAI_API_KEY": "one"})
    second = LLMRuntimeConfig.from_environment(db_path=db, env={"ECOMMERCE_DB_PATH": str(db), "OPENAI_API_KEY": "two"})
    assert first.config_hash == second.config_hash
    assert all("one" not in repr(first.evidence()) and "two" not in repr(second.evidence()) for _ in [0])


def test_grounding_rejects_claim_not_in_tool_result(tmp_path: Path):
    class Hallucinating(Provider):
        def __call__(self, prompt: str, schema: type):
            if schema is ResponseV1:
                return ModelResult(ResponseV1(message_code="ORDER_FACTS_V1", claim_fields=("unknown_field",), evidence_ref="0" * 64))
            return super().__call__(prompt, schema)
    result = InteractiveRuntime(db_path=_db(tmp_path), provider=Hallucinating()).chat(user_id="u", message="x", mode="live")
    assert result.status == "FAILED" and result.code == "GROUNDING_FAILED"


def test_grounding_rejects_wrong_evidence_hash(tmp_path: Path):
    class WrongEvidence(Provider):
        def __call__(self, prompt: str, schema: type):
            if schema is ResponseV1:
                return ModelResult(ResponseV1(message_code="ORDER_FACTS_V1", claim_fields=("order_status",), evidence_ref="0" * 64))
            return super().__call__(prompt, schema)
    result = InteractiveRuntime(db_path=_db(tmp_path), provider=WrongEvidence()).chat(user_id="u", message="x", mode="live")
    assert result.status == "FAILED" and result.code == "GROUNDING_FAILED"


def test_grounding_rejects_empty_claim_fields(tmp_path: Path):
    class EmptyFields(Provider):
        def __call__(self, prompt: str, schema: type):
            if schema is ResponseV1:
                return {"schema_version": "response.v1", "message_code": "ORDER_FACTS_V1", "claim_fields": [], "evidence_ref": "0" * 64}
            return super().__call__(prompt, schema)
    result = InteractiveRuntime(db_path=_db(tmp_path), provider=EmptyFields()).chat(user_id="u", message="x", mode="live")
    assert result.status == "FAILED" and result.code == "MODEL_SCHEMA_INVALID"


def test_plan_empty_entity_fields_are_bound_from_validated_intent(tmp_path: Path):
    class EmptyEntityPlan(Provider):
        def __call__(self, prompt: str, schema: type):
            if schema is PlanCandidateV1:
                return ModelResult(PlanCandidateV1(tool_ref="order/get_info@v1"))
            return super().__call__(prompt, schema)

    result = InteractiveRuntime(db_path=_db(tmp_path), provider=EmptyEntityPlan()).chat(user_id="u", message="x", mode="live", run_id="r1_plan_empty")
    assert result.status == "SUCCEEDED" and result.code == "OK"


def test_plan_nonempty_mismatched_entity_fails_closed(tmp_path: Path):
    class WrongEntityPlan(Provider):
        def __call__(self, prompt: str, schema: type):
            if schema is PlanCandidateV1:
                return ModelResult(PlanCandidateV1(tool_ref="order/get_info@v1", order_id="990000000099"))
            return super().__call__(prompt, schema)

    result = InteractiveRuntime(db_path=_db(tmp_path), provider=WrongEntityPlan()).chat(user_id="u", message="x", mode="live", run_id="r1_plan_wrong")
    assert result.status == "FAILED" and result.code == "PLAN_CONTRACT_INVALID"
