from __future__ import annotations

import os
import re
from enum import Enum
from typing import Annotated, Any, Dict, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class IntentType(str, Enum):
    SMALLTALK = "smalltalk"
    ORDER = "order"
    AFTERSALES = "aftersales"
    POLICY = "policy"
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
    AFTERSALES = "aftersales"
    POLICY = "policy"


class ObservationSource(str, Enum):
    TOOL = "tool"
    RETRIEVAL = "retrieval"
    JUDGEMENT = "judgement"
    USER_CLARIFICATION = "user_clarification"
    HANDOFF = "handoff"


class ActionType(str, Enum):
    GENERAL_RESPONSE = "general_response"
    QUERY_ORDER = "query_order"
    QUERY_POLICY = "query_policy"
    QUERY_AFTERSALES = "query_aftersales"
    CREATE_AFTERSALES = "create_aftersales"
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
    should_replan: bool = False
    should_handoff: bool = False
    unsupported_answer_risk: bool = False
    missing_evidence_types: List[str] = Field(default_factory=list)
    recommended_response_mode: ResponseMode = ResponseMode.IDLE
    guardrail_flags: List[str] = Field(default_factory=list)


class FinalizerInput(BaseModel):
    verified_facts: List[VerifiedFact] = Field(default_factory=list)
    retrieval_evidence: List[RetrievalEvidence] = Field(default_factory=list)
    missing_information: List[str] = Field(default_factory=list)
    allowed_response_mode: ResponseMode = ResponseMode.IDLE
    handoff_context: Dict[str, Any] = Field(default_factory=dict)
    customer_intent: IntentType = IntentType.UNKNOWN


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


def extract_slots_from_text(text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    raw = (text or "").strip()
    if not raw:
        return out

    order_match = re.search(r"(\d{8,})", raw)
    if order_match:
        out["order_id"] = order_match.group(1)

    phone_match = re.search(r"(?:后四位|尾号|手机号后四位)[^\d]{0,6}(\d{4})", raw)
    if phone_match:
        out["phone_last4"] = phone_match.group(1)
    else:
        loose_phone_match = re.search(r"\b(\d{4})\b", raw)
        if loose_phone_match and "order_id" not in out:
            out["phone_last4"] = loose_phone_match.group(1)

    if "换货" in raw:
        out["service_type"] = "换货"
    elif "退货" in raw:
        out["service_type"] = "退货"
    elif "退款" in raw:
        out["service_type"] = "退款"

    reason_match = re.search(r"(?:因为|原因是|理由是)[:：]?\s*(.+)", raw)
    if reason_match:
        reason = reason_match.group(1).strip("，。！？；:： ")
        if reason:
            out["reason"] = reason

    return out


def merge_slot_values(state: AgentState, extra_slots: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(state.get("slot_values") or {})
    for key, value in extra_slots.items():
        if value not in (None, "", []):
            merged[key] = value
    return merged


def observation_to_fact_candidates(observation: Observation) -> List[VerifiedFact]:
    facts: List[VerifiedFact] = []
    data = observation.structured_data or {}
    if observation.source_type == ObservationSource.TOOL and observation.success:
        for key, value in data.items():
            if key in {"tool_result", "user_hint", "raw"}:
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
