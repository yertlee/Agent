from __future__ import annotations

import os
import re
from enum import Enum
from typing import Annotated, Any, Dict, List, Optional, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class IntentType(str, Enum):
    SMALLTALK = "smalltalk"
    ORDER = "order"
    POLICY = "policy"
    ESCALATION = "escalation"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class PlanMode(str, Enum):
    MINIMAL = "minimal"
    FULL = "full"


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ResponseMode(str, Enum):
    IDLE = "idle"
    FINALIZE = "finalize"
    ASK_USER = "ask_user"
    EXPLAIN_LIMIT = "explain_limit"
    HANDOFF = "handoff"


class SpecialistName(str, Enum):
    GENERAL = "general"
    ORDER = "order"
    POLICY = "policy"
    ESCALATION = "escalation"


class ObservationSource(str, Enum):
    TOOL = "tool"
    RETRIEVAL = "retrieval"
    JUDGEMENT = "judgement"
    USER_CLARIFICATION = "user_clarification"
    HANDOFF = "handoff"


class ActionType(str, Enum):
    GENERAL_RESPONSE = "general_response"
    QUERY_ORDER = "query_order"
    QUERY_LOGISTICS = "query_logistics"
    CREATE_AFTERSALES = "create_aftersales"
    QUERY_AFTERSALES = "query_aftersales"
    QUERY_POLICY = "query_policy"
    HANDLE_ESCALATION = "handle_escalation"
    HANDOFF = "handoff"
    CLARIFY = "clarify"


class VerifiedFact(BaseModel):
    step_id: str
    source_type: str
    title: str
    detail: str
    reference_id: str = ""


class RetrievalEvidence(BaseModel):
    query_used: str
    rewritten_from: str = ""
    score: float
    source: str
    title: str
    chunk_id: str
    text: str
    evidence_summary: str


class PlanStep(BaseModel):
    step_id: str
    owner_agent: SpecialistName
    action_type: ActionType
    goal: str
    required_inputs: List[str] = Field(default_factory=list)
    success_condition: str
    fallback_action: str
    status: StepStatus = StepStatus.PENDING
    internal_note: str = ""


class PlanBundle(BaseModel):
    plan_mode: PlanMode
    steps: List[PlanStep] = Field(default_factory=list)
    active_specialist: SpecialistName = SpecialistName.GENERAL
    required_slots: List[str] = Field(default_factory=list)
    fallback_strategy: str = ""


class Observation(BaseModel):
    step_id: str
    source_type: ObservationSource
    source_name: str
    success: bool
    code: str
    summary: str
    structured_data: Dict[str, Any] = Field(default_factory=dict)
    evidence_refs: List[str] = Field(default_factory=list)
    missing_slots: List[str] = Field(default_factory=list)
    retryable: bool = False


class VerificationResult(BaseModel):
    can_finalize: bool = False
    must_ask_user: bool = False
    retry_same_step: bool = False
    should_replan: bool = False
    should_escalate: bool = False
    should_handoff: bool = False
    unsupported_answer_risk: bool = False
    missing_evidence_types: List[str] = Field(default_factory=list)
    recommended_response_mode: ResponseMode = ResponseMode.IDLE
    guardrail_flags: List[str] = Field(default_factory=list)
    escalation_type: str = ""
    escalation_reason: str = ""


class OrderContext(BaseModel):
    order_id: str = ""
    product_name: str = ""
    amount: Optional[float] = None
    order_status: str = ""
    pay_status: str = ""
    created_at: str = ""
    carrier_code: str = ""
    tracking_no: str = ""
    phone_last4: str = ""
    can_apply_aftersales: Optional[int] = None
    source: str = ""


class LogisticsSnapshot(BaseModel):
    carrier_code: str = ""
    tracking_no: str = ""
    delivery_state: str = ""
    delivery_state_name: str = ""
    delivery_status_code: str = ""
    last_event: str = ""
    last_event_time: str = ""
    current_location: str = ""
    route_from: str = ""
    route_to: str = ""
    is_signed: bool = False
    is_returning: bool = False
    is_abnormal: bool = False
    route_info: Dict[str, Any] = Field(default_factory=dict)
    arrival_time: str = ""
    predicted_route: List[Dict[str, Any]] = Field(default_factory=list)
    source: str = ""
    fetched_at: str = ""
    raw_payload_ref: str = ""


class AfterSalesContext(BaseModel):
    aftersales_id: str = ""
    order_id: str = ""
    service_type: str = ""
    reason: str = ""
    eligibility: str = ""
    aftersales_status: str = ""
    denial_reason: str = ""
    policy_hint: str = ""
    next_action_hint: str = ""
    created_at: str = ""
    updated_at: str = ""
    source: str = ""


class FailureRecord(BaseModel):
    step_id: str = ""
    stage: str = ""
    source_name: str = ""
    code: str = ""
    summary: str = ""
    user_input: str = ""
    retryable: bool = False


class EscalationContext(BaseModel):
    user_request: str = ""
    escalation_type: str = ""
    escalation_reason: str = ""
    order_context: Dict[str, Any] = Field(default_factory=dict)
    logistics_snapshot: Dict[str, Any] = Field(default_factory=dict)
    aftersales_context: Dict[str, Any] = Field(default_factory=dict)
    policy_evidence: List[Dict[str, Any]] = Field(default_factory=list)
    verified_facts: List[Dict[str, Any]] = Field(default_factory=list)
    failure_history: List[Dict[str, Any]] = Field(default_factory=list)
    attempted_actions: List[str] = Field(default_factory=list)
    latest_failure: Dict[str, Any] = Field(default_factory=dict)
    latest_user_request: str = ""
    last_observation_code: str = ""
    last_observation_summary: str = ""


class FinalizerInput(BaseModel):
    verified_facts: List[VerifiedFact] = Field(default_factory=list)
    retrieval_evidence: List[RetrievalEvidence] = Field(default_factory=list)
    missing_information: List[str] = Field(default_factory=list)
    allowed_response_mode: ResponseMode = ResponseMode.IDLE
    handoff_context: Dict[str, Any] = Field(default_factory=dict)
    customer_intent: IntentType = IntentType.UNKNOWN
    order_action: str = ""
    order_context: Dict[str, Any] = Field(default_factory=dict)
    logistics_snapshot: Dict[str, Any] = Field(default_factory=dict)
    aftersales_context: Dict[str, Any] = Field(default_factory=dict)
    last_observation_code: str = ""
    last_observation_summary: str = ""
    escalation_type: str = ""
    escalation_reason: str = ""
    escalation_context: Dict[str, Any] = Field(default_factory=dict)
    escalation_decision: str = ""
    handoff_case_payload: Dict[str, Any] = Field(default_factory=dict)


class SpecialistExecutionResult(BaseModel):
    observation: Observation
    state_patch: Dict[str, Any] = Field(default_factory=dict)
    recommended_next_action: str = ""
    verification_hints: Dict[str, Any] = Field(default_factory=dict)


class AgentState(TypedDict, total=False):
    session_id: str
    messages: Annotated[List[BaseMessage], add_messages]
    user_input: str
    intent_type: IntentType
    plan_mode: PlanMode
    current_plan: List[PlanStep]
    current_step_index: int
    active_agent: SpecialistName
    order_action: Optional[ActionType]
    observations: List[Observation]
    last_observation: Optional[Observation]
    verified_facts: List[VerifiedFact]
    retrieval_evidence: List[RetrievalEvidence]
    awaited_slots: List[str]
    blocked_step_id: str
    pending_question: str
    slot_values: Dict[str, Any]
    tool_retry_counts: Dict[str, int]
    rag_retry_count: int
    replan_count: int
    verification_status: Optional[VerificationResult]
    guardrail_flags: List[str]
    handoff_reason: str
    response_mode: ResponseMode
    final_response: str
    trace_tags: Dict[str, Any]
    pending_retrieval_query: str
    order_context: Optional[OrderContext]
    logistics_snapshot: Optional[LogisticsSnapshot]
    logistics_cache_meta: Dict[str, Any]
    aftersales_context: Optional[AfterSalesContext]
    escalation_type: str
    escalation_reason: str
    escalation_context: Optional[EscalationContext]
    escalation_decision: str
    handoff_case_payload: Dict[str, Any]
    failure_history: List[FailureRecord]
    retry_count_by_stage: Dict[str, int]
    mixed_order_result: Optional[Dict[str, Any]]
    mixed_policy_result: Optional[Dict[str, Any]]


def initial_state(session_id: str) -> AgentState:
    return {
        "session_id": session_id,
        "messages": [],
        "user_input": "",
        "intent_type": IntentType.UNKNOWN,
        "plan_mode": PlanMode.MINIMAL,
        "current_plan": [],
        "current_step_index": 0,
        "active_agent": SpecialistName.GENERAL,
        "order_action": None,
        "observations": [],
        "last_observation": None,
        "verified_facts": [],
        "retrieval_evidence": [],
        "awaited_slots": [],
        "blocked_step_id": "",
        "pending_question": "",
        "slot_values": {},
        "tool_retry_counts": {},
        "rag_retry_count": 0,
        "replan_count": 0,
        "verification_status": None,
        "guardrail_flags": [],
        "handoff_reason": "",
        "response_mode": ResponseMode.IDLE,
        "final_response": "",
        "order_context": None,
        "logistics_snapshot": None,
        "logistics_cache_meta": {},
        "aftersales_context": None,
        "escalation_type": "",
        "escalation_reason": "",
        "escalation_context": None,
        "escalation_decision": "",
        "handoff_case_payload": {},
        "failure_history": [],
        "retry_count_by_stage": {},
        "mixed_order_result": None,
        "mixed_policy_result": None,
        "trace_tags": {
            "current_node": "",
            "langsmith_project": os.getenv("LANGSMITH_PROJECT", ""),
            "thread_id": session_id,
        },
        "pending_retrieval_query": "",
    }


def get_current_step(state: AgentState) -> Optional[PlanStep]:
    plan = state.get("current_plan") or []
    index = int(state.get("current_step_index") or 0)
    if index < 0 or index >= len(plan):
        return None
    return plan[index]


def append_observation(state: AgentState, observation: Observation) -> Dict[str, Any]:
    return {
        "observations": list(state.get("observations") or []) + [observation],
        "last_observation": observation,
    }


def record_failure(
    state: AgentState,
    observation: Observation,
    *,
    stage: str,
    user_input: str = "",
) -> Dict[str, Any]:
    stage_key = stage or observation.source_name or observation.step_id or "unknown"
    history = list(state.get("failure_history") or [])
    history.append(
        FailureRecord(
            step_id=observation.step_id,
            stage=stage_key,
            source_name=observation.source_name,
            code=observation.code,
            summary=observation.summary,
            user_input=user_input,
            retryable=observation.retryable,
        )
    )
    history = history[-8:]
    retry_counts = dict(state.get("retry_count_by_stage") or {})
    if observation.code != "QUERY_TOO_FREQUENT":
        retry_counts[stage_key] = int(retry_counts.get(stage_key) or 0) + 1
    return {
        "failure_history": history,
        "retry_count_by_stage": retry_counts,
    }


def reset_failure_stage(state: AgentState, stage: str) -> Dict[str, Any]:
    retry_counts = dict(state.get("retry_count_by_stage") or {})
    if stage:
        retry_counts[stage] = 0
    return {"retry_count_by_stage": retry_counts}


def summarize_messages(messages: List[BaseMessage], limit: int = 6) -> str:
    recent = messages[-limit:]
    parts: List[str] = []
    for msg in recent:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        parts.append(f"{role}: {msg.content}")
    return "\n".join(parts)


def render_debug_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, list):
        return [render_debug_value(item) for item in value]
    if isinstance(value, dict):
        return {k: render_debug_value(v) for k, v in value.items()}
    if isinstance(value, BaseMessage):
        role = "user" if isinstance(value, HumanMessage) else "assistant"
        return {"role": role, "content": value.content}
    return value


def state_to_debug_dict(state: AgentState) -> Dict[str, Any]:
    return {k: render_debug_value(v) for k, v in dict(state).items()}


def mark_step_status(state: AgentState, step_id: str, status: StepStatus) -> List[PlanStep]:
    plan = list(state.get("current_plan") or [])
    new_plan: List[PlanStep] = []
    for step in plan:
        if step.step_id == step_id:
            new_plan.append(step.model_copy(update={"status": status}))
        else:
            new_plan.append(step)
    return new_plan


def next_pending_step_index(plan: List[PlanStep], start_index: int = 0) -> Optional[int]:
    for idx in range(max(start_index, 0), len(plan)):
        if plan[idx].status in {StepStatus.PENDING, StepStatus.RUNNING, StepStatus.BLOCKED}:
            return idx
    return None


REASON_PHRASES = [
    "和描述不符",
    "尺码不合适",
    "颜色不喜欢",
    "质量不好",
    "不想要了",
    "不合适",
    "买错了",
    "拍错了",
    "有瑕疵",
    "少件",
]
TRACKING_LABEL_PATTERN = r"(?:运单号|快递单号|物流单号|tracking(?:\s*no|\s*number)?)"
ORDER_LABEL_PATTERN = r"(?:订单号|订单编号|order(?:\s*id)?)"
PHONE_LABEL_PATTERN = r"(?:后四位|尾号|手机号后四位|手机后四位)"
PHONE_EXPLICIT_PATTERN = r"(?:收件人)?手机号"
ORDER_DOMAIN_HINTS = [
    "订单",
    "物流",
    "快递",
    "售后",
    "退款",
    "退货",
    "换货",
    "规则",
    "政策",
    "发货",
    "签收",
    "运单",
]
ORDER_DOMAIN_SLOT_KEYS = ["order_id", "phone_last4", "carrier_code", "tracking_no", "service_type", "reason"]


def _clean_reason_text(value: str) -> str:
    return value.strip().strip("，,。！？!；;：: ")


def _looks_like_followup_reason(raw: str) -> bool:
    text = _clean_reason_text(raw)
    if not text or len(text) > 40:
        return False
    if text.isdigit():
        return False
    if any(mark in text for mark in ["?", "？"]):
        return False
    if re.search(r"(?:订单号|手机号|后四位|运单号|快递单号|物流单号)", text):
        return False
    if re.search(r"\d{8,}", text):
        return False
    return not any(hint in text for hint in ORDER_DOMAIN_HINTS)


def _match_explicit_identifier(raw: str, label_pattern: str, *, min_len: int = 6) -> str:
    pattern = rf"{label_pattern}[^\wA-Za-z0-9]{{0,8}}([A-Za-z0-9-]{{{min_len},32}})"
    match = re.search(pattern, raw, re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _detect_service_type(raw: str) -> str:
    if "换货" in raw:
        return "换货"
    if "退货" in raw:
        return "退货"
    if any(word in raw for word in ["退款", "退钱", "退掉"]):
        return "退款"
    return ""


def extract_slots_from_text(text: str, awaited_slots: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    raw = (text or "").strip()
    awaited = set(awaited_slots or [])
    if not raw:
        return out

    carrier_match = re.search(
        r"(?:carrier[_ ]?code|快递公司编码|公司编码|物流公司编码|快递公司)[\s:：-]*([A-Za-z][A-Za-z0-9_-]{1,20})",
        raw,
        re.IGNORECASE,
    )
    if carrier_match:
        out["carrier_code"] = carrier_match.group(1).strip().lower()

    tracking_no = _match_explicit_identifier(raw, TRACKING_LABEL_PATTERN, min_len=6)
    if tracking_no:
        out["tracking_no"] = tracking_no

    order_id = _match_explicit_identifier(raw, ORDER_LABEL_PATTERN, min_len=8)
    if order_id:
        out["order_id"] = order_id

    if "order_id" not in out:
        generic_candidates = re.findall(r"(?<!\d)([A-Za-z0-9-]{8,32})(?!\d)", raw)
        tracking_hint = bool(re.search(TRACKING_LABEL_PATTERN, raw, re.IGNORECASE))
        for candidate in generic_candidates:
            if candidate == out.get("tracking_no"):
                continue
            if tracking_hint and "order_id" not in out:
                continue
            if candidate.isdigit():
                out["order_id"] = candidate
                break

    phone_match = re.search(rf"(?:{PHONE_LABEL_PATTERN}|{PHONE_EXPLICIT_PATTERN})[^\d]{{0,6}}(\d{{4}})(?!\d)", raw)
    if phone_match:
        out["phone_last4"] = phone_match.group(1)
    elif "phone_last4" in awaited:
        loose_phone_match = re.search(r"(?<!\d)(\d{4})(?!\d)", raw)
        if loose_phone_match:
            out["phone_last4"] = loose_phone_match.group(1)

    service_type = _detect_service_type(raw)
    if service_type:
        out["service_type"] = service_type

    reason_match = re.search(r"(?:因为|原因是|理由是)[：:，,\s]*(.+)", raw)
    if reason_match:
        reason = _clean_reason_text(reason_match.group(1))
        if reason:
            out["reason"] = reason

    if "reason" not in out:
        for phrase in REASON_PHRASES:
            if phrase in raw:
                out["reason"] = phrase
                break

    if "reason" not in out and "reason" in awaited and _looks_like_followup_reason(raw):
        out["reason"] = _clean_reason_text(raw)

    return out


def merge_slot_values(state: AgentState, extra_slots: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(state.get("slot_values") or {})
    previous_order_id = str(merged.get("order_id") or "")
    next_order_id = str(extra_slots.get("order_id") or "")
    previous_phone_last4 = str(merged.get("phone_last4") or "")
    next_phone_last4 = str(extra_slots.get("phone_last4") or "")
    previous_tracking_no = str(merged.get("tracking_no") or "")
    next_tracking_no = str(extra_slots.get("tracking_no") or "")

    identity_changed = bool(
        (previous_order_id and next_order_id and previous_order_id != next_order_id)
        or (previous_phone_last4 and next_phone_last4 and previous_phone_last4 != next_phone_last4)
        or (previous_tracking_no and next_tracking_no and previous_tracking_no != next_tracking_no)
    )
    if identity_changed:
        for key in ORDER_DOMAIN_SLOT_KEYS:
            merged.pop(key, None)

    for key, value in extra_slots.items():
        if value not in (None, "", []):
            merged[key] = value
    return merged


def observation_to_fact_candidates(observation: Observation) -> List[VerifiedFact]:
    facts: List[VerifiedFact] = []
    data = observation.structured_data or {}
    if observation.source_type == ObservationSource.TOOL and observation.success:
        for key, value in data.items():
            if key in {"tool_result", "user_hint", "raw", "raw_payload_ref", "_cache_meta"}:
                continue
            facts.append(
                VerifiedFact(
                    step_id=observation.step_id,
                    source_type=observation.source_type.value,
                    title=key,
                    detail=str(value),
                    reference_id=f"{observation.source_name}:{key}",
                )
            )
    return facts


def as_ai_message(text: str) -> AIMessage:
    return AIMessage(content=text)
