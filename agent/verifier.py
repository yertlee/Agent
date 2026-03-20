from __future__ import annotations

from typing import List

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from .llm import build_chat_model
from .prompts import VERIFIER_GRAYZONE_SYSTEM_PROMPT, build_verifier_grayzone_prompt
from .state import (
    ActionType,
    AgentState,
    IntentType,
    Observation,
    ObservationSource,
    ResponseMode,
    RetrievalEvidence,
    VerificationResult,
    VerifiedFact,
    get_current_step,
)
from .langsmith_utils import traceable


BUSINESS_FINALIZABLE_CODES = {
    "OK",
    "AFTERSALES_ALREADY_EXISTS",
    "AFTERSALES_NOT_ALLOWED",
    "AFTERSALES_NOT_FOUND",
}

ASK_USER_CODES = {
    "MISSING_SLOTS",
    "PHONE_MISMATCH",
    "ORDER_NOT_FOUND",
    "INVALID_PARAMS",
}

HANDOFF_CODES = {
    "DB_ERROR",
    "DB_NOT_FOUND",
    "TOOL_ERROR",
    "NO_TOOL",
}


class GrayzoneDecision(BaseModel):
    can_finalize: bool = False
    recommended_response_mode: ResponseMode = ResponseMode.EXPLAIN_LIMIT
    missing_evidence_types: List[str] = Field(default_factory=list)


def _has_more_pending_steps(state: AgentState) -> bool:
    plan = state.get("current_plan") or []
    current_index = int(state.get("current_step_index") or 0)
    return current_index + 1 < len(plan)


def _too_many_failures(state: AgentState, observation: Observation | None) -> bool:
    if observation is None:
        return False
    if not observation.success and observation.source_name:
        retry_count = (state.get("tool_retry_counts") or {}).get(observation.source_name, 0)
        if retry_count >= 2:
            return True
    return (state.get("replan_count") or 0) >= 2


def _needs_grayzone_check(state: AgentState, observation: Observation | None) -> bool:
    step = get_current_step(state)
    if observation is None or step is None:
        return False
    if step.action_type == ActionType.QUERY_POLICY and bool(state.get("retrieval_evidence")):
        return True
    if step.action_type == ActionType.GENERAL_RESPONSE:
        return False
    if bool(state.get("verified_facts")) and not _has_more_pending_steps(state):
        return True
    return False


@traceable(name="agent_v3_verifier_grayzone")
def _run_grayzone_check(
    *,
    intent_type: IntentType,
    step,
    observation: Observation | None,
    verified_facts: List[VerifiedFact],
    retrieval_evidence: List[RetrievalEvidence],
) -> GrayzoneDecision:
    llm = build_chat_model(temperature=0.0, max_tokens=250, tags=["verifier"])
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", VERIFIER_GRAYZONE_SYSTEM_PROMPT),
            ("human", "{user_prompt}"),
        ]
    )
    chain = prompt | llm.with_structured_output(GrayzoneDecision)
    return chain.invoke(
        {
            "user_prompt": build_verifier_grayzone_prompt(
                intent_type=intent_type,
                step=step,
                observation=observation,
                verified_facts=verified_facts,
                retrieval_evidence=retrieval_evidence,
            )
        }
    )


@traceable(name="agent_v3_verify_state")
def verify_state(state: AgentState) -> VerificationResult:
    step = get_current_step(state)
    observation = state.get("last_observation")
    awaited_slots = list(state.get("awaited_slots") or [])
    verified_facts = list(state.get("verified_facts") or [])
    retrieval_evidence = list(state.get("retrieval_evidence") or [])

    if state.get("handoff_reason"):
        return VerificationResult(
            should_handoff=True,
            recommended_response_mode=ResponseMode.HANDOFF,
            guardrail_flags=["handoff_reason_present"],
        )

    if awaited_slots or state.get("pending_question"):
        return VerificationResult(
            must_ask_user=True,
            recommended_response_mode=ResponseMode.ASK_USER,
            missing_evidence_types=list(awaited_slots),
            guardrail_flags=["awaiting_user_slots"],
        )

    if observation is None:
        if step and step.action_type == ActionType.GENERAL_RESPONSE:
            return VerificationResult(
                can_finalize=True,
                recommended_response_mode=ResponseMode.FINALIZE,
            )
        return VerificationResult(
            should_replan=True,
            recommended_response_mode=ResponseMode.IDLE,
            guardrail_flags=["missing_observation"],
        )

    if _too_many_failures(state, observation):
        return VerificationResult(
            should_handoff=True,
            recommended_response_mode=ResponseMode.HANDOFF,
            guardrail_flags=["retry_limit_exceeded"],
        )

    if observation.source_type == ObservationSource.USER_CLARIFICATION:
        return VerificationResult(
            must_ask_user=True,
            recommended_response_mode=ResponseMode.ASK_USER,
            missing_evidence_types=observation.missing_slots,
            guardrail_flags=["user_clarification_required"],
        )

    if observation.source_type == ObservationSource.HANDOFF:
        return VerificationResult(
            should_handoff=True,
            recommended_response_mode=ResponseMode.HANDOFF,
            guardrail_flags=["handoff_observation"],
        )

    if observation.source_type == ObservationSource.TOOL and not observation.success:
        if observation.code in ASK_USER_CODES:
            return VerificationResult(
                must_ask_user=True,
                recommended_response_mode=ResponseMode.ASK_USER,
                missing_evidence_types=observation.missing_slots or awaited_slots,
                guardrail_flags=[f"tool_failure:{observation.code}"],
            )
        if observation.code in HANDOFF_CODES:
            return VerificationResult(
                should_handoff=True,
                recommended_response_mode=ResponseMode.HANDOFF,
                guardrail_flags=[f"tool_failure:{observation.code}"],
            )
        if observation.code == "NO_HITS":
            return VerificationResult(
                can_finalize=True,
                recommended_response_mode=ResponseMode.EXPLAIN_LIMIT,
                unsupported_answer_risk=False,
                missing_evidence_types=["policy_evidence"],
                guardrail_flags=["rag_no_hit"],
            )
        if observation.code in BUSINESS_FINALIZABLE_CODES:
            if _has_more_pending_steps(state):
                return VerificationResult(
                    should_replan=True,
                    recommended_response_mode=ResponseMode.IDLE,
                    guardrail_flags=["advance_plan_after_business_result"],
                )
            return VerificationResult(
                can_finalize=True,
                recommended_response_mode=ResponseMode.FINALIZE,
                unsupported_answer_risk=False,
            )
        return VerificationResult(
            should_handoff=True,
            recommended_response_mode=ResponseMode.HANDOFF,
            guardrail_flags=[f"unhandled_tool_failure:{observation.code}"],
        )

    if observation.source_type == ObservationSource.RETRIEVAL and not observation.success:
        if state.get("rag_retry_count", 0) >= 1:
            return VerificationResult(
                can_finalize=True,
                recommended_response_mode=ResponseMode.EXPLAIN_LIMIT,
                missing_evidence_types=["policy_evidence"],
                guardrail_flags=["rag_retry_exhausted"],
            )
        return VerificationResult(
            should_replan=True,
            recommended_response_mode=ResponseMode.IDLE,
            guardrail_flags=["rag_retry_available"],
        )

    if step and step.action_type == ActionType.GENERAL_RESPONSE:
        return VerificationResult(
            can_finalize=True,
            recommended_response_mode=ResponseMode.FINALIZE,
        )

    if _has_more_pending_steps(state):
        return VerificationResult(
            should_replan=True,
            recommended_response_mode=ResponseMode.IDLE,
            guardrail_flags=["next_step_available"],
        )

    if not verified_facts and not retrieval_evidence:
        return VerificationResult(
            can_finalize=True,
            recommended_response_mode=ResponseMode.EXPLAIN_LIMIT,
            unsupported_answer_risk=True,
            missing_evidence_types=["tool_or_retrieval_evidence"],
            guardrail_flags=["no_evidence_for_finalize"],
        )

    if _needs_grayzone_check(state, observation):
        try:
            gray = _run_grayzone_check(
                intent_type=state.get("intent_type") or IntentType.UNKNOWN,
                step=step,
                observation=observation,
                verified_facts=verified_facts,
                retrieval_evidence=retrieval_evidence,
            )
            return VerificationResult(
                can_finalize=gray.can_finalize,
                recommended_response_mode=gray.recommended_response_mode,
                unsupported_answer_risk=not gray.can_finalize,
                missing_evidence_types=gray.missing_evidence_types,
                guardrail_flags=["grayzone_llm_check"],
            )
        except Exception:
            pass

    return VerificationResult(
        can_finalize=True,
        recommended_response_mode=ResponseMode.FINALIZE,
        unsupported_answer_risk=False,
    )

