from __future__ import annotations

from typing import Any, Dict, Optional

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from .llm import build_chat_model
from .planner import build_initial_plan, classify_intent, replan_after_verification
from .prompts import FINALIZER_SYSTEM_PROMPT, build_finalizer_user_prompt
from .specialists import aftersales_specialist_node, build_order_subgraph, build_policy_subgraph
from .state import (
    ActionType,
    AgentState,
    FinalizerInput,
    IntentType,
    Observation,
    ObservationSource,
    PlanMode,
    ResponseMode,
    SpecialistName,
    StepStatus,
    append_observation,
    as_ai_message,
    extract_slots_from_text,
    get_current_step,
    initial_state,
    mark_step_status,
    merge_slot_values,
    next_pending_step_index,
)
from .verifier import verify_state
from .langsmith_utils import traceable
from .tool_registry import build_tool_registry


TOOL_REGISTRY = build_tool_registry()


def _with_node_tag(state: AgentState, node_name: str) -> Dict[str, Any]:
    tags = dict(state.get("trace_tags") or {})
    tags["current_node"] = node_name
    return {"trace_tags": tags}


@traceable(name="agent_v3_ingest_node")
def ingest_node(state: AgentState) -> Dict[str, Any]:
    user_input = state.get("user_input", "")
    slot_values = merge_slot_values(state, extract_slots_from_text(user_input))
    patch = _with_node_tag(state, "ingest_node")
    patch.update(
        {
            "session_id": state.get("session_id") or "",
            "slot_values": slot_values,
            "response_mode": ResponseMode.IDLE,
            "final_response": "",
            "handoff_reason": "",
            "pending_retrieval_query": "",
            "rag_retry_count": 0,
        }
    )
    if user_input:
        patch["messages"] = [HumanMessage(content=user_input)]
    return patch


@traceable(name="agent_v3_classify_node")
def classify_node(state: AgentState) -> Dict[str, Any]:
    patch = _with_node_tag(state, "classify_node")
    intent = classify_intent(state)
    patch["intent_type"] = intent
    return patch


def _apply_plan_bundle(state: AgentState, bundle, *, reset_evidence: bool, replan_count_delta: int = 0) -> Dict[str, Any]:
    plan = list(bundle.steps)
    current_index = next_pending_step_index(plan, 0) or 0
    patch = {
        "plan_mode": bundle.plan_mode,
        "current_plan": plan,
        "current_step_index": current_index,
        "active_agent": bundle.active_specialist,
        "awaited_slots": list(bundle.required_slots),
        "pending_question": state.get("pending_question", "") if bundle.required_slots else "",
        "verification_status": None,
        "replan_count": int(state.get("replan_count") or 0) + replan_count_delta,
    }
    if reset_evidence:
        patch["retrieval_evidence"] = []
    return patch


@traceable(name="agent_v3_planner_node")
def planner_node(state: AgentState) -> Dict[str, Any]:
    patch = _with_node_tag(state, "planner_node")
    current_plan = list(state.get("current_plan") or [])
    verification = state.get("verification_status")
    current_step = get_current_step(state)

    if current_plan and verification is None and current_step is not None and current_step.status in {
        StepStatus.PENDING,
        StepStatus.RUNNING,
        StepStatus.BLOCKED,
    }:
        patch.update(
            {
                "active_agent": current_step.owner_agent,
                "awaited_slots": [],
                "pending_question": "",
            }
        )
        return patch

    if current_plan and verification and "next_step_available" in verification.guardrail_flags:
        next_index = next_pending_step_index(current_plan, int(state.get("current_step_index") or 0) + 1)
        if next_index is None:
            return patch
        next_step = current_plan[next_index]
        patch.update(
            {
                "current_step_index": next_index,
                "active_agent": next_step.owner_agent,
                "awaited_slots": [],
                "pending_question": "",
                "verification_status": None,
            }
        )
        return patch

    if current_plan and verification and verification.should_replan:
        bundle = replan_after_verification(state, verification)
        patch.update(_apply_plan_bundle(state, bundle, reset_evidence=False, replan_count_delta=1))
        return patch

    bundle = build_initial_plan(state)
    patch.update(_apply_plan_bundle(state, bundle, reset_evidence=True))
    return patch


@traceable(name="agent_v3_dispatch_node")
def dispatch_node(state: AgentState) -> Dict[str, Any]:
    patch = _with_node_tag(state, "dispatch_node")
    step = get_current_step(state)
    if step is None:
        return patch

    if step.status in {StepStatus.COMPLETED, StepStatus.SKIPPED}:
        return patch

    slot_values = dict(state.get("slot_values") or {})
    missing_slots = [slot for slot in step.required_inputs if not slot_values.get(slot)]
    if missing_slots:
        question = f"为了继续处理，请补充：{'、'.join(missing_slots)}。"
        observation = Observation(
            step_id=step.step_id,
            source_type=ObservationSource.USER_CLARIFICATION,
            source_name="dispatch_node",
            success=False,
            code="MISSING_SLOTS",
            summary=question,
            structured_data={},
            missing_slots=missing_slots,
            retryable=True,
        )
        patch.update(append_observation(state, observation))
        patch.update(
            {
                "awaited_slots": missing_slots,
                "blocked_step_id": step.step_id,
                "pending_question": question,
                "current_plan": mark_step_status(state, step.step_id, StepStatus.BLOCKED),
            }
        )
        return patch

    patch["active_agent"] = step.owner_agent
    patch["blocked_step_id"] = ""
    patch["pending_question"] = ""
    patch["awaited_slots"] = []
    return patch


@traceable(name="agent_v3_general_node")
def general_node(state: AgentState) -> Dict[str, Any]:
    step = get_current_step(state)
    patch = _with_node_tag(state, "general_node")
    if step is None:
        return patch

    observation = Observation(
        step_id=step.step_id,
        source_type=ObservationSource.JUDGEMENT,
        source_name="general_node",
        success=True,
        code="GENERAL_READY",
        summary="当前对话属于寒暄或能力说明，可直接礼貌回复。",
        structured_data={"intent": (state.get("intent_type") or IntentType.SMALLTALK).value},
    )
    patch.update(append_observation(state, observation))
    patch["current_plan"] = mark_step_status(state, step.step_id, StepStatus.COMPLETED)
    return patch


@traceable(name="agent_v3_verifier_node")
def verifier_node(state: AgentState) -> Dict[str, Any]:
    patch = _with_node_tag(state, "verifier_node")
    verification = verify_state(state)
    patch["verification_status"] = verification
    patch["response_mode"] = verification.recommended_response_mode
    patch["guardrail_flags"] = verification.guardrail_flags
    return patch


@traceable(name="agent_v3_await_user_node")
def await_user_node(state: AgentState) -> Dict[str, Any]:
    patch = _with_node_tag(state, "await_user_node")
    resume_payload = interrupt(
        {
            "pending_question": state.get("pending_question") or "请补充继续处理所需的信息。",
            "awaited_slots": list(state.get("awaited_slots") or []),
            "blocked_step_id": state.get("blocked_step_id") or "",
        }
    )
    if isinstance(resume_payload, dict):
        user_input = str(resume_payload.get("user_input") or "")
    else:
        user_input = str(resume_payload or "")
    new_slots = extract_slots_from_text(user_input)
    merged_slots = merge_slot_values(state, new_slots)
    blocked_step_id = state.get("blocked_step_id") or ""
    patch.update(
        {
            "user_input": user_input,
            "messages": [HumanMessage(content=user_input)],
            "slot_values": merged_slots,
            "awaited_slots": [],
            "pending_question": "",
            "blocked_step_id": "",
            "verification_status": None,
        }
    )
    if blocked_step_id:
        patch["current_plan"] = mark_step_status(state, blocked_step_id, StepStatus.PENDING)
    return patch


def _build_finalizer_input(state: AgentState) -> FinalizerInput:
    verification = state.get("verification_status")
    return FinalizerInput(
        verified_facts=list(state.get("verified_facts") or []),
        retrieval_evidence=list(state.get("retrieval_evidence") or []),
        missing_information=list(state.get("awaited_slots") or []),
        allowed_response_mode=(verification.recommended_response_mode if verification else ResponseMode.IDLE),
        handoff_context={
            "reason": state.get("handoff_reason") or "",
            "pending_question": state.get("pending_question") or "",
        },
        customer_intent=state.get("intent_type") or IntentType.UNKNOWN,
    )


def _finalizer_fallback(data: FinalizerInput) -> str:
    if data.allowed_response_mode == ResponseMode.ASK_USER:
        missing = "、".join(data.missing_information) or "必要信息"
        return f"为了继续处理，请补充：{missing}。"
    if data.allowed_response_mode == ResponseMode.EXPLAIN_LIMIT:
        return "目前基于已有工具结果和规则检索证据，还不足以给出完全确定的结论。您可以补充更具体的信息，我再继续为您判断。"
    if data.allowed_response_mode == ResponseMode.HANDOFF:
        reason = data.handoff_context.get("reason") or "当前问题需要人工进一步处理"
        return f"当前问题我已为您转人工处理，原因是：{reason}。请稍候。"
    if data.customer_intent == IntentType.SMALLTALK:
        return "您好，我可以协助您查询订单、处理售后、解释规则，也可以在需要时帮您转人工客服。"

    parts = []
    if data.verified_facts:
        fact_lines = [f"- {fact.title}: {fact.detail}" for fact in data.verified_facts[:4]]
        parts.append("基于系统查询：\n" + "\n".join(fact_lines))
    if data.retrieval_evidence:
        policy_lines = [f"- {item.title}: {item.evidence_summary}" for item in data.retrieval_evidence[:2]]
        parts.append("基于规则检索：\n" + "\n".join(policy_lines))
    if not parts:
        parts.append("目前没有足够证据支撑明确结论。")
    parts.append("如需我继续处理，请告诉我下一步需求。")
    return "\n\n".join(parts)


@traceable(name="agent_v3_finalizer_node")
def finalizer_node(state: AgentState) -> Dict[str, Any]:
    patch = _with_node_tag(state, "finalizer_node")
    data = _build_finalizer_input(state)

    llm = build_chat_model(temperature=0.1, max_tokens=500, tags=["finalizer"])
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", FINALIZER_SYSTEM_PROMPT),
            ("human", "{user_prompt}"),
        ]
    )
    chain = prompt | llm
    try:
        result = chain.invoke({"user_prompt": build_finalizer_user_prompt(data)})
        final_text = (result.content or "").strip() or _finalizer_fallback(data)
    except Exception:
        final_text = _finalizer_fallback(data)

    patch["final_response"] = final_text
    patch["messages"] = [as_ai_message(final_text)]
    return patch


@traceable(name="agent_v3_handoff_node")
def handoff_node(state: AgentState) -> Dict[str, Any]:
    step = get_current_step(state)
    patch = _with_node_tag(state, "handoff_node")
    reason = state.get("handoff_reason") or "当前问题需要人工介入"
    summary = state.get("user_input") or "用户请求人工处理"
    result = TOOL_REGISTRY["handoff_to_human_tool"].callable(summary=summary, reason=reason)
    observation = Observation(
        step_id=step.step_id if step else "handoff_step",
        source_type=ObservationSource.HANDOFF,
        source_name="handoff_to_human_tool",
        success=bool(result.get("success")),
        code=str(result.get("code") or ""),
        summary=str(result.get("message") or ""),
        structured_data=(result.get("data") or {}) if isinstance(result.get("data"), dict) else {},
        retryable=False,
    )
    patch.update(append_observation(state, observation))
    if step is not None:
        patch["current_plan"] = mark_step_status(state, step.step_id, StepStatus.COMPLETED)
    final_text = f"当前问题我已为您转人工处理，原因是：{reason}。请稍候。"
    patch["final_response"] = final_text
    patch["messages"] = [as_ai_message(final_text)]
    return patch


def _route_after_dispatch(state: AgentState) -> str:
    if state.get("awaited_slots"):
        return "verifier_node"
    step = get_current_step(state)
    if step is None:
        return "finalizer_node"
    if step.owner_agent == SpecialistName.GENERAL:
        return "general_node"
    if step.owner_agent == SpecialistName.ORDER:
        return "order_subgraph"
    if step.owner_agent == SpecialistName.POLICY:
        return "policy_subgraph"
    return "aftersales_node"


def _route_after_verifier(state: AgentState) -> str:
    verification = state.get("verification_status")
    if verification is None:
        return "planner_node"
    if verification.must_ask_user:
        return "await_user_node"
    if verification.should_handoff:
        return "handoff_node"
    if verification.can_finalize:
        return "finalizer_node"
    if verification.should_replan:
        return "planner_node"
    return "finalizer_node"


def build_agent_graph():
    order_subgraph = build_order_subgraph()
    policy_subgraph = build_policy_subgraph()

    graph = StateGraph(AgentState)
    graph.add_node("ingest_node", ingest_node)
    graph.add_node("classify_node", classify_node)
    graph.add_node("planner_node", planner_node)
    graph.add_node("dispatch_node", dispatch_node)
    graph.add_node("general_node", general_node)
    graph.add_node("order_subgraph", order_subgraph)
    graph.add_node("policy_subgraph", policy_subgraph)
    graph.add_node("aftersales_node", aftersales_specialist_node)
    graph.add_node("verifier_node", verifier_node)
    graph.add_node("await_user_node", await_user_node)
    graph.add_node("finalizer_node", finalizer_node)
    graph.add_node("handoff_node", handoff_node)

    graph.add_edge(START, "ingest_node")
    graph.add_edge("ingest_node", "classify_node")
    graph.add_edge("classify_node", "planner_node")
    graph.add_edge("planner_node", "dispatch_node")
    graph.add_conditional_edges(
        "dispatch_node",
        _route_after_dispatch,
        {
            "general_node": "general_node",
            "order_subgraph": "order_subgraph",
            "policy_subgraph": "policy_subgraph",
            "aftersales_node": "aftersales_node",
            "verifier_node": "verifier_node",
            "finalizer_node": "finalizer_node",
        },
    )
    graph.add_edge("general_node", "verifier_node")
    graph.add_edge("order_subgraph", "verifier_node")
    graph.add_edge("policy_subgraph", "verifier_node")
    graph.add_edge("aftersales_node", "verifier_node")
    graph.add_conditional_edges(
        "verifier_node",
        _route_after_verifier,
        {
            "planner_node": "planner_node",
            "await_user_node": "await_user_node",
            "handoff_node": "handoff_node",
            "finalizer_node": "finalizer_node",
        },
    )
    graph.add_edge("await_user_node", "planner_node")
    graph.add_edge("handoff_node", END)
    graph.add_edge("finalizer_node", END)
    return graph


def seed_state(session_id: str, user_input: str) -> AgentState:
    seeded = initial_state(session_id)
    seeded["session_id"] = session_id
    seeded["user_input"] = user_input
    return seeded
