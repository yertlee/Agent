"""R3-C grounded QA boundary.

The model is called only after the shared R3 retriever returns ``ANSWERED``.
The provider/config adapter is reused from ``interactive_runtime``; secrets
are reduced to presence markers by that adapter and never enter this module's
trace or response objects.
"""

from __future__ import annotations

import json
import time
from datetime import date
from typing import Any, Literal, Mapping, Protocol

from pydantic import Field

from .interactive_runtime import (
    ChatOpenAIStructuredProvider,
    LLMRuntimeConfig,
    ModelCallError,
    ModelClient,
    ModelConfigurationError,
    StructuredContract,
)
from .r3_rag_contracts import Claim, EvidenceRef
from .r3_rag_runtime import R3Retriever


class GroundedClaim(StructuredContract):
    claim_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    evidence_ids: tuple[str, ...] = Field(min_length=1)
    supporting_quote: str = Field(min_length=1)


class GroundedAnswer(StructuredContract):
    decision: Literal["ANSWERED", "INSUFFICIENT"]
    answer: str = ""
    claims: tuple[GroundedClaim, ...] = ()


class GroundedProvider(Protocol):
    def __call__(self, prompt: str, output_schema: type[Any]) -> Any: ...


class GroundedQAResult:
    def __init__(self, *, status: str, answer: str, claims: tuple[GroundedClaim, ...], trace: Mapping[str, Any]) -> None:
        self.status = status
        self.answer = answer
        self.claims = claims
        self.trace = dict(trace)

    def as_dict(self) -> dict[str, Any]:
        return {"status": self.status, "answer": self.answer, "claims": [claim.model_dump() for claim in self.claims], "trace": self.trace}


class R3GroundedQARuntime:
    def __init__(self, retriever: R3Retriever, *, config: LLMRuntimeConfig | None = None, provider: GroundedProvider | None = None) -> None:
        self.retriever = retriever
        self.config = config
        self.provider = provider

    @classmethod
    def from_environment(cls, retriever: R3Retriever, *, db_path: str, provider: GroundedProvider | None = None) -> "R3GroundedQARuntime":
        config = LLMRuntimeConfig.from_environment(db_path=db_path)
        return cls(retriever, config=config, provider=provider)

    @staticmethod
    def _abstention_answer(status: str) -> str:
        return {
            "NO_HITS": "没有找到足够的项目演示证据，无法回答。",
            "WEAK_EVIDENCE": "现有项目演示证据不足，无法可靠回答。",
            "CONFLICT": "项目演示资料存在冲突，无法给出唯一结论。",
            "STALE_ONLY": "只找到已过期的项目演示资料，无法给出当前结论。",
        }.get(status, "检索未完成，无法回答。")

    def _prompt(self, query: str, evidence: list[EvidenceRef], text_by_id: Mapping[str, str]) -> str:
        packet = [{"evidence_id": ref.evidence_id, "source_id": ref.source_id, "version": ref.version, "locator": ref.locator, "text": text_by_id[ref.evidence_id]} for ref in evidence]
        return (
            "Answer the user query only from the supplied project-authored evidence. "
            "The exact output contract is: GroundedAnswer fields are decision, answer, claims. "
            "GroundedClaim fields are claim_id, text, evidence_ids, supporting_quote. "
            "Use only one of these pure placeholder structures: "
            "{\"decision\":\"ANSWERED\",\"answer\":\"<non-empty answer>\",\"claims\":[{\"claim_id\":\"<id>\",\"text\":\"<claim>\",\"evidence_ids\":[\"<evidence_id>\"],\"supporting_quote\":\"<exact quote>\"}]} "
            "or {\"decision\":\"INSUFFICIENT\",\"answer\":\"\",\"claims\":[]}. "
            "The only legal decision values are ANSWERED and INSUFFICIENT. "
            "Return decision=ANSWERED only when the evidence states the specific requested attribute or condition. "
            "If the evidence is merely topically related but does not answer the concrete question, return "
            "decision=INSUFFICIENT with an empty claims list. For ANSWERED, return structured answer and claims. "
            "Every claim must cite one or more evidence_id values and include a supporting_quote copied exactly "
            "from the corresponding evidence text. Do not invent facts.\n"
            + json.dumps({"query": query, "evidence": packet}, ensure_ascii=False, sort_keys=True)
        )

    def _repair_prompt(self, query: str, evidence: list[EvidenceRef], text_by_id: Mapping[str, str], error_category: str) -> str:
        """Build one generic repair request over the unchanged query/evidence.

        The prior model output is deliberately omitted.  The provider receives
        only the original contract/evidence plus a stable validation category.
        """
        return (
            "Repair the structured response using the same query and evidence. "
            f"Validation error category: {error_category}. "
            "Return only the original GroundedAnswer contract, with exact evidence quotes.\n"
            + self._prompt(query, evidence, text_by_id)
        )

    def _validate_grounding(self, output: GroundedAnswer, evidence: list[EvidenceRef], text_by_id: Mapping[str, str]) -> None:
        if output.decision == "INSUFFICIENT":
            if output.claims:
                raise ValueError("GROUNDING_INSUFFICIENT_WITH_CLAIMS")
            return
        if output.decision != "ANSWERED" or not output.claims:
            raise ValueError("GROUNDING_ANSWERED_WITHOUT_CLAIMS")
        if not output.answer.strip():
            raise ValueError("GROUNDING_ANSWERED_WITHOUT_ANSWER")
        valid_ids = {ref.evidence_id for ref in evidence}
        seen_claims: set[str] = set()
        for claim in output.claims:
            if claim.claim_id in seen_claims or not set(claim.evidence_ids) <= valid_ids:
                raise ValueError("GROUNDING_INVALID_EVIDENCE")
            if not any(claim.supporting_quote in text_by_id[evidence_id] for evidence_id in claim.evidence_ids):
                raise ValueError("GROUNDING_INVALID_QUOTE")
            seen_claims.add(claim.claim_id)

    def answer(self, query: str, *, as_of: date, mode: str = "hybrid_reranker") -> GroundedQAResult:
        retrieval = self.retriever.retrieve(query, as_of=as_of, mode=mode)
        evidence = retrieval.evidence
        text_by_id = {f"ev-{hit.chunk.chunk_id}": hit.chunk.text for hit in retrieval.hits}
        trace: dict[str, Any] = {
            "provider_called": False,
            "provider_returned": False,
            "model": self.config.model if self.config else None,
            "latency_ms": 0,
            "retrieval_status": retrieval.status,
            "retrieval_version_tuple": retrieval.trace.get("version_tuple"),
            "evidence_ids": [ref.evidence_id for ref in evidence],
            "error": None,
            "schema_error": None,
            "grounding_pass": False,
            "attempt_count": 0,
            "attempts": [],
            "final_result": None,
            "total_latency_ms": 0,
        }
        if retrieval.status != "ANSWERED":
            trace["final_result"] = retrieval.status
            return GroundedQAResult(status=retrieval.status, answer=self._abstention_answer(retrieval.status), claims=(), trace=trace)
        if self.config is None:
            trace["error"] = "MODEL_CONFIG_MISSING"
            trace["final_result"] = "FAILED"
            return GroundedQAResult(status="FAILED", answer="模型配置不可用，无法生成有依据回答。", claims=(), trace=trace)
        provider = self.provider
        if provider is None:
            try:
                provider = ChatOpenAIStructuredProvider(self.config)
            except ModelConfigurationError as exc:
                trace["error"] = exc.code
                trace["final_result"] = "FAILED"
                return GroundedQAResult(status="FAILED", answer="模型配置不可用，无法生成有依据回答。", claims=(), trace=trace)
        client = ModelClient(self.config, provider=provider)
        trace["provider_called"] = True
        total_started = time.perf_counter()
        first_validation_error: str | None = None
        for attempt_no in range(1, 3):
            trace["attempt_count"] = attempt_no
            prompt = self._prompt(query, evidence, text_by_id) if attempt_no == 1 else self._repair_prompt(query, evidence, text_by_id, str(trace.get("error") or "GROUNDING_VALIDATION"))
            started = time.perf_counter()
            returned = False
            error: str | None = None
            try:
                output, _usage, latency = client.complete(prompt, GroundedAnswer)
                returned = True
                self._validate_grounding(output, evidence, text_by_id)
                elapsed = latency or int((time.perf_counter() - started) * 1000)
                trace["attempts"].append({"attempt": attempt_no, "returned": True, "error": None, "latency_ms": elapsed})
                trace["provider_returned"] = True
                trace["grounding_pass"] = True
                trace["error"] = None
                trace["schema_error"] = None
                trace["final_result"] = "WEAK_EVIDENCE" if output.decision == "INSUFFICIENT" else "ANSWERED"
                trace["total_latency_ms"] = int((time.perf_counter() - total_started) * 1000)
                trace["latency_ms"] = trace["total_latency_ms"]
                if output.decision == "INSUFFICIENT":
                    return GroundedQAResult(status="WEAK_EVIDENCE", answer=self._abstention_answer("WEAK_EVIDENCE"), claims=(), trace=trace)
                return GroundedQAResult(status="ANSWERED", answer=output.answer, claims=output.claims, trace=trace)
            except (ModelCallError, ValueError) as exc:
                error = getattr(exc, "code", str(exc))
                # MODEL_SCHEMA_INVALID means the provider returned a payload
                # that failed local schema validation; grounding ValueErrors
                # likewise follow a returned payload.
                returned = isinstance(exc, ValueError) or error == "MODEL_SCHEMA_INVALID"
                trace["provider_returned"] = trace["provider_returned"] or returned
                trace["error"] = error
                trace["schema_error"] = error if error.startswith("MODEL_SCHEMA") else None
                elapsed = int((time.perf_counter() - started) * 1000)
                trace["attempts"].append({"attempt": attempt_no, "returned": returned, "error": error, "latency_ms": elapsed})
                if attempt_no == 1 and (error == "MODEL_SCHEMA_INVALID" or error.startswith("GROUNDING_")):
                    first_validation_error = error
                retryable = error in {"MODEL_SCHEMA_INVALID", "MODEL_PROVIDER_ERROR", "MODEL_TIMEOUT"} or error.startswith("GROUNDING_")
                if retryable and attempt_no == 1:
                    continue
                if first_validation_error and error in {"MODEL_PROVIDER_ERROR", "MODEL_TIMEOUT"}:
                    trace["error"] = first_validation_error
                trace["final_result"] = "FAILED"
                trace["total_latency_ms"] = int((time.perf_counter() - total_started) * 1000)
                trace["latency_ms"] = trace["total_latency_ms"]
                return GroundedQAResult(status="FAILED", answer="模型输出未通过证据校验，无法回答。", claims=(), trace=trace)
        trace["final_result"] = "FAILED"
        trace["total_latency_ms"] = int((time.perf_counter() - total_started) * 1000)
        trace["latency_ms"] = trace["total_latency_ms"]
        return GroundedQAResult(status="FAILED", answer="模型输出未通过证据校验，无法回答。", claims=(), trace=trace)
