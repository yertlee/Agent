from __future__ import annotations

from typing import List, Tuple

from .langsmith_utils import traceable
from .state import (
    ActionType,
    AgentState,
    IntentType,
    Observation,
    ObservationSource,
    ResponseMode,
    SpecialistName,
    VerificationResult,
    get_current_step,
)


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

LOGISTICS_RETRY_CODES = {
    "SIMULATOR_RETRYABLE",
}

LOGISTICS_ESCALATION_CODES = {
    "SIMULATOR_UNAVAILABLE",
    "SIMULATOR_INVALID_RESPONSE",
}

HUMAN_REQUEST_KEYWORDS = [
    "转人工",
    "人工客服",
    "找人工",
    "找客服主管",
    "找主管",
    "你处理不了",
]

COMPLAINT_KEYWORDS = [
    "我要投诉",
    "投诉",
    "我要举报",
    "举报",
]

COMPENSATION_KEYWORDS = [
    "赔偿",
    "赔付",
    "补偿",
]

DELIVERY_DISPUTE_KEYWORDS = [
    "丢件",
    "丢包",
    "没收到却显示签收",
    "显示签收但没收到",
    "签收争议",
    "物流一直不更新",
    "物流不更新",
]

REJECT_CURRENT_RESULT_KEYWORDS = [
    "不接受当前结果",
    "不接受这个结果",
    "系统结果错误",
    "这个结果不对",
]

RULE_FACT_CONFLICT_KEYWORDS = [
    "规则说可以退",
    "规则说可以",
    "政策说可以退",
    "系统又不给我退",
    "系统不给退",
    "规则和系统对不上",
]


def _contains_any(text: str, keywords: List[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _user_escalation_signal(state: AgentState) -> Tuple[str, str] | None:
    text = str(state.get("user_input") or "")
    if not text:
        return None
    if _contains_any(text, HUMAN_REQUEST_KEYWORDS):
        return ("HUMAN_REQUEST", "用户明确要求人工接管。")
    if _contains_any(text, COMPENSATION_KEYWORDS):
        return ("COMPENSATION_DISPUTE", "用户提出赔偿或赔付争议。")
    if _contains_any(text, DELIVERY_DISPUTE_KEYWORDS):
        return ("DELIVERY_DISPUTE", "用户提出物流或签收争议。")
    if _contains_any(text, COMPLAINT_KEYWORDS):
        return ("COMPLAINT", "用户明确提出投诉或升级诉求。")
    if _contains_any(text, REJECT_CURRENT_RESULT_KEYWORDS):
        return ("RULE_FACT_CONFLICT", "用户明确表示不接受当前结果。")
    return None


def _failure_stage_key(observation: Observation | None) -> str:
    if observation is None:
        return ""
    return f"{observation.step_id}:{observation.source_name or 'unknown'}"


def _stage_retry_count(state: AgentState, observation: Observation | None) -> int:
    stage_key = _failure_stage_key(observation)
    if not stage_key:
        return 0
    return int((state.get("retry_count_by_stage") or {}).get(stage_key) or 0)


def _logistics_fact_conflict(state: AgentState) -> str:
    order_context = state.get("order_context")
    logistics = state.get("logistics_snapshot")
    if order_context is None or logistics is None:
        return ""

    order_status = str(getattr(order_context, "order_status", "") or "")
    delivery_state = str(getattr(logistics, "delivery_state", "") or "")
    delivery_state_name = str(getattr(logistics, "delivery_state_name", "") or "")
    if ("已签收" in order_status or "已完成" in order_status) and delivery_state in {"in_transit", "delivering", "abnormal", "not_found"}:
        return f"订单状态显示{order_status}，但物流状态显示{delivery_state_name or delivery_state}。"
    if ("运输中" in order_status or "待收货" in order_status) and getattr(logistics, "is_signed", False):
        return f"订单状态显示{order_status}，但物流已显示签收。"
    return ""


def _rule_fact_conflict(state: AgentState) -> str:
    text = str(state.get("user_input") or "")
    if _contains_any(text, RULE_FACT_CONFLICT_KEYWORDS):
        return "用户描述规则结论与系统处理结果冲突。"

    if not state.get("retrieval_evidence"):
        return ""

    aftersales_context = state.get("aftersales_context")
    if aftersales_context is not None:
        eligibility = str(getattr(aftersales_context, "eligibility", "") or "")
        denial_reason = str(getattr(aftersales_context, "denial_reason", "") or "")
        if eligibility in {"NOT_ALLOWED", "NEED_MANUAL"} and denial_reason:
            return f"规则证据已命中，但售后结果为{denial_reason}"
    return ""


def _verify_logistics_failure(state: AgentState, observation: Observation, awaited_slots: List[str]) -> VerificationResult | None:
    if observation.source_name != "query_logistics_snapshot_tool":
        return None

    payload = observation.structured_data or {}
    error_code = str(observation.code or payload.get("error_code") or "")
    suggested_action = str(payload.get("suggested_action") or "")
    missing_slots = list(observation.missing_slots or awaited_slots or payload.get("missing_slots") or [])
    stage_retry_count = _stage_retry_count(state, observation)
    guardrail = f"logistics_failure:{error_code or suggested_action or 'unknown'}"

    if error_code == "408" or missing_slots:
        return VerificationResult(
            must_ask_user=True,
            recommended_response_mode=ResponseMode.ASK_USER,
            missing_evidence_types=missing_slots or ["phone_last4"],
            guardrail_flags=[guardrail],
        )

    if error_code == "QUERY_TOO_FREQUENT":
        has_cached_snapshot = bool(state.get("logistics_snapshot"))
        return VerificationResult(
            can_finalize=True,
            recommended_response_mode=(ResponseMode.FINALIZE if has_cached_snapshot else ResponseMode.EXPLAIN_LIMIT),
            missing_evidence_types=([] if has_cached_snapshot else ["logistics_snapshot"]),
            guardrail_flags=[guardrail, ("use_cached_logistics" if has_cached_snapshot else "logistics_query_rate_limited")],
        )

    if error_code in LOGISTICS_RETRY_CODES or suggested_action == "retry_later":
        if stage_retry_count < 1:
            return VerificationResult(
                retry_same_step=True,
                recommended_response_mode=ResponseMode.IDLE,
                guardrail_flags=[guardrail, "retry_same_step"],
            )
        return VerificationResult(
            should_escalate=True,
            recommended_response_mode=ResponseMode.HANDOFF,
            escalation_type="SYSTEM_FAILURE_ESCALATION",
            escalation_reason="物流服务连续失败，建议升级人工继续处理。",
            guardrail_flags=[guardrail, "retry_exhausted"],
        )

    if error_code == "400" or suggested_action == "ask_user":
        return VerificationResult(
            can_finalize=True,
            recommended_response_mode=ResponseMode.EXPLAIN_LIMIT,
            missing_evidence_types=["logistics_snapshot"],
            guardrail_flags=[guardrail, "logistics_friendly_fallback"],
        )

    if error_code in LOGISTICS_ESCALATION_CODES or suggested_action == "handoff":
        return VerificationResult(
            should_escalate=True,
            recommended_response_mode=ResponseMode.HANDOFF,
            escalation_type="SYSTEM_FAILURE_ESCALATION",
            escalation_reason="物流服务异常且无法自动恢复，需要人工接管。",
            guardrail_flags=[guardrail],
        )

    return None


@traceable(name="agent_v3_verify_state")
def verify_state(state: AgentState) -> VerificationResult:
    step = get_current_step(state)
    observation = state.get("last_observation")
    awaited_slots = list(state.get("awaited_slots") or [])

    user_signal = _user_escalation_signal(state)
    if user_signal is not None and not state.get("escalation_decision") and (
        step is None or step.owner_agent != SpecialistName.ESCALATION
    ):
        escalation_type, escalation_reason = user_signal
        return VerificationResult(
            should_escalate=True,
            recommended_response_mode=ResponseMode.HANDOFF,
            escalation_type=escalation_type,
            escalation_reason=escalation_reason,
            guardrail_flags=["user_requested_escalation"],
        )

    if state.get("handoff_reason"):
        return VerificationResult(
            should_escalate=True,
            recommended_response_mode=ResponseMode.HANDOFF,
            escalation_type="SYSTEM_FAILURE_ESCALATION",
            escalation_reason=str(state.get("handoff_reason") or ""),
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

    if not state.get("escalation_decision") and (step is None or step.owner_agent != SpecialistName.ESCALATION):
        fact_conflict = _logistics_fact_conflict(state)
        if fact_conflict:
            return VerificationResult(
                should_escalate=True,
                recommended_response_mode=ResponseMode.HANDOFF,
                escalation_type="DELIVERY_DISPUTE",
                escalation_reason=fact_conflict,
                guardrail_flags=["order_logistics_conflict"],
            )

        rule_conflict = _rule_fact_conflict(state)
        if rule_conflict:
            return VerificationResult(
                should_escalate=True,
                recommended_response_mode=ResponseMode.HANDOFF,
                escalation_type="RULE_FACT_CONFLICT",
                escalation_reason=rule_conflict,
                guardrail_flags=["rule_fact_conflict"],
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
            can_finalize=True,
            recommended_response_mode=ResponseMode.FINALIZE,
            guardrail_flags=["handoff_observation"],
        )

    if observation.source_type == ObservationSource.TOOL and not observation.success:
        logistics_decision = _verify_logistics_failure(state, observation, awaited_slots)
        if logistics_decision is not None:
            return logistics_decision

        if observation.code in ASK_USER_CODES:
            return VerificationResult(
                must_ask_user=True,
                recommended_response_mode=ResponseMode.ASK_USER,
                missing_evidence_types=observation.missing_slots or awaited_slots,
                guardrail_flags=[f"tool_failure:{observation.code}"],
            )

        if observation.code in HANDOFF_CODES:
            return VerificationResult(
                should_escalate=True,
                recommended_response_mode=ResponseMode.HANDOFF,
                escalation_type="SYSTEM_FAILURE_ESCALATION",
                escalation_reason=f"{observation.source_name} 连续失败，需要人工接管。",
                guardrail_flags=[f"tool_failure:{observation.code}"],
            )

        if observation.code == "NO_HITS":
            return VerificationResult(
                can_finalize=True,
                recommended_response_mode=ResponseMode.EXPLAIN_LIMIT,
                missing_evidence_types=["policy_evidence"],
                guardrail_flags=["rag_no_hit"],
            )

        stage_retry_count = _stage_retry_count(state, observation)
        if stage_retry_count >= 1:
            return VerificationResult(
                should_escalate=True,
                recommended_response_mode=ResponseMode.HANDOFF,
                escalation_type="SYSTEM_FAILURE_ESCALATION",
                escalation_reason="同一处理阶段已连续失败，建议升级人工继续处理。",
                guardrail_flags=["retry_limit_exceeded"],
            )

        if step and step.action_type == ActionType.CREATE_AFTERSALES:
            return VerificationResult(
                should_escalate=True,
                recommended_response_mode=ResponseMode.HANDOFF,
                escalation_type="SYSTEM_FAILURE_ESCALATION",
                escalation_reason="售后创建失败且当前原因无法自动恢复，需要人工介入。",
                guardrail_flags=["aftersales_create_unrecoverable"],
            )

        return VerificationResult(
            retry_same_step=True,
            recommended_response_mode=ResponseMode.IDLE,
            guardrail_flags=["retry_same_step"],
        )

    if observation.source_type == ObservationSource.RETRIEVAL and not observation.success:
        if int(state.get("rag_retry_count") or 0) < 1:
            return VerificationResult(
                retry_same_step=True,
                recommended_response_mode=ResponseMode.IDLE,
                guardrail_flags=["rag_retry_available", "retry_same_step"],
            )
        return VerificationResult(
            can_finalize=True,
            recommended_response_mode=ResponseMode.EXPLAIN_LIMIT,
            missing_evidence_types=["policy_evidence"],
            guardrail_flags=["rag_retry_exhausted"],
        )

    if step and step.action_type == ActionType.GENERAL_RESPONSE:
        return VerificationResult(
            can_finalize=True,
            recommended_response_mode=ResponseMode.FINALIZE,
        )

    plan = state.get("current_plan") or []
    current_index = int(state.get("current_step_index") or 0)
    for next_step in plan[current_index + 1 :]:
        if next_step.status.value in {"pending", "running", "blocked"}:
            return VerificationResult(
                should_replan=True,
                recommended_response_mode=ResponseMode.IDLE,
                guardrail_flags=["next_step_available"],
            )

    if not state.get("verified_facts") and not state.get("retrieval_evidence"):
        return VerificationResult(
            can_finalize=True,
            recommended_response_mode=ResponseMode.EXPLAIN_LIMIT,
            unsupported_answer_risk=True,
            missing_evidence_types=["tool_or_retrieval_evidence"],
            guardrail_flags=["no_evidence_for_finalize"],
        )

    return VerificationResult(
        can_finalize=True,
        recommended_response_mode=ResponseMode.FINALIZE,
        unsupported_answer_risk=False,
    )
