from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from .langsmith_utils import traceable
from .planner import build_initial_plan, classify_intent, replan_after_verification
from .specialists import build_escalation_subgraph, build_order_subgraph, build_policy_subgraph
from .state import (
    ActionType,
    AgentState,
    FinalizerInput,
    IntentType,
    Observation,
    ObservationSource,
    ORDER_DOMAIN_SLOT_KEYS,
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
    record_failure,
    reset_failure_stage,
)
from .tool_registry import build_tool_registry
from .verifier import verify_state


TOOL_REGISTRY = build_tool_registry()
MIXED_ORDER_SUBGRAPH = build_order_subgraph()
MIXED_POLICY_SUBGRAPH = build_policy_subgraph()
NEW_ORDER_HINTS = [
    "另一个订单",
    "另外一个订单",
    "另外一单",
    "另一单",
    "其他订单",
    "别的订单",
    "换个订单",
    "换一个订单",
    "新订单",
    "不是这个订单",
]


def _with_node_tag(state: AgentState, node_name: str) -> Dict[str, Any]:
    tags = dict(state.get("trace_tags") or {})
    tags["current_node"] = node_name
    return {"trace_tags": tags}


def _fresh_trace_tags(state: AgentState, session_id: str) -> Dict[str, Any]:
    existing = dict(state.get("trace_tags") or {})
    return {
        "current_node": "",
        "langsmith_project": existing.get("langsmith_project", ""),
        "thread_id": session_id,
    }


def _mentions_new_order_context(user_input: str) -> bool:
    text = (user_input or "").strip()
    return any(hint in text for hint in NEW_ORDER_HINTS)


def _clear_order_slot_values(state: AgentState) -> Dict[str, Any]:
    slot_values = dict(state.get("slot_values") or {})
    for key in ORDER_DOMAIN_SLOT_KEYS:
        slot_values.pop(key, None)
    return slot_values


def _should_reset_order_context(state: AgentState, user_input: str, new_slots: Dict[str, Any]) -> bool:
    current_context = state.get("order_context")
    current_order_id = ""
    current_phone_last4 = ""
    current_tracking_no = ""
    if current_context is not None:
        current_order_id = str(getattr(current_context, "order_id", "") or "")
        current_phone_last4 = str(getattr(current_context, "phone_last4", "") or "")
        current_tracking_no = str(getattr(current_context, "tracking_no", "") or "")

    previous_order_id = str((state.get("slot_values") or {}).get("order_id") or "")
    previous_phone_last4 = str((state.get("slot_values") or {}).get("phone_last4") or "")
    previous_tracking_no = str((state.get("slot_values") or {}).get("tracking_no") or "")
    explicit_order_id = str(new_slots.get("order_id") or "")
    explicit_phone_last4 = str(new_slots.get("phone_last4") or "")
    explicit_tracking_no = str(new_slots.get("tracking_no") or "")
    return bool(
        _mentions_new_order_context(user_input)
        or (explicit_order_id and explicit_order_id not in {current_order_id, previous_order_id})
        or (explicit_phone_last4 and explicit_phone_last4 not in {current_phone_last4, previous_phone_last4})
        or (explicit_tracking_no and explicit_tracking_no not in {current_tracking_no, previous_tracking_no})
    )


def _policy_text_from_evidence(evidence: list) -> str:
    if not evidence:
        return ""
    summaries = [getattr(item, "evidence_summary", "") or getattr(item, "title", "") for item in evidence[:2]]
    text = "我先帮你整理一下相关规则：" + "；".join(filter(None, summaries))
    return text.rstrip("；") + "。"


def _merge_unique_model_list(*lists):
    merged = []
    seen = set()
    for items in lists:
        for item in items or []:
            key = repr(item)
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
    return merged


def _mixed_branch_state(state: AgentState, owner: SpecialistName) -> Dict[str, Any]:
    branch_state = deepcopy(dict(state))
    branch_steps = [step.model_copy() for step in list(state.get("current_plan") or []) if step.owner_agent == owner]
    branch_state["current_plan"] = branch_steps
    branch_state["current_step_index"] = 0
    branch_state["active_agent"] = owner
    branch_state["awaited_slots"] = []
    branch_state["blocked_step_id"] = ""
    branch_state["pending_question"] = ""
    branch_state["verification_status"] = None
    branch_state["response_mode"] = ResponseMode.IDLE
    branch_state["final_response"] = ""
    branch_state["mixed_order_result"] = None
    branch_state["mixed_policy_result"] = None
    return branch_state


def _has_pending_owner_step(state: AgentState, owner: SpecialistName) -> bool:
    for step in list(state.get("current_plan") or []):
        if step.owner_agent == owner and step.status in {StepStatus.PENDING, StepStatus.RUNNING, StepStatus.BLOCKED}:
            return True
    return False


@traceable(name="agent_v3_ingest_node")
def ingest_node(state: AgentState) -> Dict[str, Any]:
    user_input = state.get("user_input", "")
    extra_slots = extract_slots_from_text(user_input)
    reset_order_context = _should_reset_order_context(state, user_input, extra_slots)
    slot_seed = _clear_order_slot_values(state) if reset_order_context else dict(state.get("slot_values") or {})
    slot_state = dict(state)
    slot_state["slot_values"] = slot_seed
    slot_values = merge_slot_values(slot_state, extra_slots)
    session_id = state.get("session_id") or ""

    patch = _with_node_tag(state, "ingest_node")
    patch.update(
        {
            "session_id": session_id,
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
            "slot_values": slot_values,
            "tool_retry_counts": {},
            "rag_retry_count": 0,
            "replan_count": 0,
            "verification_status": None,
            "guardrail_flags": [],
            "handoff_reason": "",
            "response_mode": ResponseMode.IDLE,
            "final_response": "",
            "pending_retrieval_query": "",
            "escalation_type": "",
            "escalation_reason": "",
            "escalation_context": None,
            "escalation_decision": "",
            "handoff_case_payload": {},
            "mixed_order_result": None,
            "mixed_policy_result": None,
            "failure_history": [] if reset_order_context else list(state.get("failure_history") or []),
            "retry_count_by_stage": {} if reset_order_context else dict(state.get("retry_count_by_stage") or {}),
            "trace_tags": _fresh_trace_tags(state, session_id),
        }
    )

    if reset_order_context:
        patch.update(
            {
                "order_context": None,
                "logistics_snapshot": None,
                "logistics_cache_meta": {},
                "aftersales_context": None,
            }
        )
    else:
        patch.update(
            {
                "order_context": state.get("order_context"),
                "logistics_snapshot": state.get("logistics_snapshot"),
                "logistics_cache_meta": dict(state.get("logistics_cache_meta") or {}),
                "aftersales_context": state.get("aftersales_context"),
            }
        )

    if user_input:
        patch["messages"] = [HumanMessage(content=user_input)]
    return patch


@traceable(name="agent_v3_classify_node")
def classify_node(state: AgentState) -> Dict[str, Any]:
    patch = _with_node_tag(state, "classify_node")
    patch["intent_type"] = classify_intent(state)
    return patch


def _apply_plan_bundle(state: AgentState, bundle, *, reset_evidence: bool, replan_count_delta: int = 0) -> Dict[str, Any]:
    plan = list(bundle.steps)
    current_index = next_pending_step_index(plan, 0) or 0
    current_step = plan[current_index] if plan else None
    patch: Dict[str, Any] = {
        "plan_mode": bundle.plan_mode,
        "current_plan": plan,
        "current_step_index": current_index,
        "active_agent": bundle.active_specialist,
        "awaited_slots": list(bundle.required_slots),
        "pending_question": state.get("pending_question", "") if bundle.required_slots else "",
        "verification_status": None,
        "replan_count": int(state.get("replan_count") or 0) + replan_count_delta,
        "order_action": current_step.action_type if current_step and current_step.owner_agent == SpecialistName.ORDER else None,
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
                "order_action": current_step.action_type if current_step.owner_agent == SpecialistName.ORDER else None,
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
                "order_action": next_step.action_type if next_step.owner_agent == SpecialistName.ORDER else None,
            }
        )
        return patch

    if current_plan and verification and verification.retry_same_step and current_step is not None:
        patch.update(
            {
                "current_plan": mark_step_status(state, current_step.step_id, StepStatus.PENDING),
                "current_step_index": int(state.get("current_step_index") or 0),
                "active_agent": current_step.owner_agent,
                "awaited_slots": [],
                "pending_question": "",
                "verification_status": None,
                "response_mode": ResponseMode.IDLE,
            }
        )
        return patch

    if current_plan and verification and verification.should_escalate:
        bundle = replan_after_verification(state, verification)
        patch.update(_apply_plan_bundle(state, bundle, reset_evidence=False, replan_count_delta=1))
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

    if (
        state.get("intent_type") == IntentType.MIXED
        and _has_pending_owner_step(state, SpecialistName.ORDER)
        and _has_pending_owner_step(state, SpecialistName.POLICY)
    ):
        patch.update(
            {
                "active_agent": SpecialistName.GENERAL,
                "blocked_step_id": "",
                "pending_question": "",
                "awaited_slots": [],
            }
        )
        return patch

    if step.owner_agent == SpecialistName.ORDER:
        patch.update(
            {
                "active_agent": step.owner_agent,
                "blocked_step_id": "",
                "pending_question": "",
                "awaited_slots": [],
                "order_action": step.action_type,
            }
        )
        return patch

    if step.owner_agent == SpecialistName.ESCALATION:
        patch.update(
            {
                "active_agent": step.owner_agent,
                "blocked_step_id": "",
                "pending_question": "",
                "awaited_slots": [],
            }
        )
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
                "messages": [as_ai_message(question)],
                "current_plan": mark_step_status(state, step.step_id, StepStatus.BLOCKED),
            }
        )
        return patch

    patch["active_agent"] = step.owner_agent
    patch["blocked_step_id"] = ""
    patch["pending_question"] = ""
    patch["awaited_slots"] = []
    return patch


@traceable(name="agent_v3_mixed_parallel_entry")
def mixed_parallel_entry_node(state: AgentState) -> Dict[str, Any]:
    return _with_node_tag(state, "mixed_parallel_entry")


@traceable(name="agent_v3_mixed_order_runner")
def mixed_order_runner_node(state: AgentState) -> Dict[str, Any]:
    branch_state = _mixed_branch_state(state, SpecialistName.ORDER)
    branch_result = MIXED_ORDER_SUBGRAPH.invoke(branch_state)
    return {"mixed_order_result": branch_result}


@traceable(name="agent_v3_mixed_policy_runner")
def mixed_policy_runner_node(state: AgentState) -> Dict[str, Any]:
    branch_state = _mixed_branch_state(state, SpecialistName.POLICY)
    branch_result = MIXED_POLICY_SUBGRAPH.invoke(branch_state)
    return {"mixed_policy_result": branch_result}


@traceable(name="agent_v3_mixed_join")
def mixed_join_node(state: AgentState) -> Dict[str, Any]:
    patch = _with_node_tag(state, "mixed_join")
    order_result = dict(state.get("mixed_order_result") or {})
    policy_result = dict(state.get("mixed_policy_result") or {})

    order_plan = list(order_result.get("current_plan") or [])
    policy_plan = list(policy_result.get("current_plan") or [])
    order_step_status = order_plan[0].status if order_plan else StepStatus.PENDING
    policy_step_status = policy_plan[0].status if policy_plan else StepStatus.PENDING

    combined_plan = list(state.get("current_plan") or [])
    if len(combined_plan) >= 1:
        combined_plan[0] = combined_plan[0].model_copy(update={"status": order_step_status})
    if len(combined_plan) >= 2:
        combined_plan[1] = combined_plan[1].model_copy(update={"status": policy_step_status})

    awaited_slots = list(order_result.get("awaited_slots") or [])
    pending_question = str(order_result.get("pending_question") or "")
    retrieval_evidence = _merge_unique_model_list(policy_result.get("retrieval_evidence") or [])
    policy_prefix = _policy_text_from_evidence(retrieval_evidence)
    if awaited_slots and pending_question and policy_prefix:
        pending_question = f"{policy_prefix}\n\n{pending_question}"

    observations = _merge_unique_model_list(
        order_result.get("observations") or [],
        policy_result.get("observations") or [],
    )
    summary = "订单域与规则域并行处理完成。"
    if awaited_slots:
        summary = "规则部分已整理完成，订单部分仍需用户补充信息。"
    join_observation = Observation(
        step_id="mixed_parallel_handle",
        source_type=ObservationSource.JUDGEMENT,
        source_name="mixed_join",
        success=True,
        code="MIXED_RESULTS_READY",
        summary=summary,
        structured_data={
            "order_branch_status": order_step_status.value,
            "policy_branch_status": policy_step_status.value,
            "policy_hit_count": len(retrieval_evidence),
        },
    )
    observations = observations + [join_observation]

    current_index = next_pending_step_index(combined_plan, 0)
    if current_index is None:
        current_index = max(len(combined_plan) - 1, 0)

    patch.update(
        {
            "current_plan": combined_plan,
            "current_step_index": current_index,
            "active_agent": combined_plan[current_index].owner_agent if combined_plan else SpecialistName.GENERAL,
            "observations": observations,
            "last_observation": join_observation,
            "verified_facts": _merge_unique_model_list(
                order_result.get("verified_facts") or [],
                policy_result.get("verified_facts") or [],
            ),
            "retrieval_evidence": retrieval_evidence,
            "awaited_slots": awaited_slots,
            "blocked_step_id": str(order_result.get("blocked_step_id") or ""),
            "pending_question": pending_question,
            "slot_values": dict(order_result.get("slot_values") or state.get("slot_values") or {}),
            "tool_retry_counts": dict(order_result.get("tool_retry_counts") or state.get("tool_retry_counts") or {}),
            "order_action": order_result.get("order_action") or state.get("order_action"),
            "order_context": order_result.get("order_context") or state.get("order_context"),
            "logistics_snapshot": order_result.get("logistics_snapshot") or state.get("logistics_snapshot"),
            "logistics_cache_meta": dict(order_result.get("logistics_cache_meta") or state.get("logistics_cache_meta") or {}),
            "aftersales_context": order_result.get("aftersales_context") or state.get("aftersales_context"),
            "handoff_reason": str(order_result.get("handoff_reason") or policy_result.get("handoff_reason") or ""),
            "messages": [as_ai_message(pending_question)] if awaited_slots and pending_question else [],
            "mixed_order_result": None,
            "mixed_policy_result": None,
        }
    )
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
        summary="当前对话属于寒暄或能力说明，可直接自然回复。",
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
    if verification.should_escalate:
        patch["escalation_type"] = verification.escalation_type
        patch["escalation_reason"] = verification.escalation_reason

    observation = state.get("last_observation")
    if isinstance(observation, Observation):
        stage = f"{observation.step_id}:{observation.source_name or 'unknown'}"
        if observation.success:
            patch.update(reset_failure_stage(state, stage))
        elif observation.source_type != ObservationSource.USER_CLARIFICATION:
            patch.update(
                record_failure(
                    state,
                    observation,
                    stage=stage,
                    user_input=str(state.get("user_input") or ""),
                )
            )
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

    new_slots = extract_slots_from_text(user_input, awaited_slots=state.get("awaited_slots") or [])
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
    last_observation = state.get("last_observation")
    order_context = state.get("order_context")
    logistics_snapshot = state.get("logistics_snapshot")
    aftersales_context = state.get("aftersales_context")
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
        order_action=(state.get("order_action").value if state.get("order_action") else ""),
        order_context=order_context.model_dump() if order_context else {},
        logistics_snapshot=logistics_snapshot.model_dump() if logistics_snapshot else {},
        aftersales_context=aftersales_context.model_dump() if aftersales_context else {},
        last_observation_code=str(getattr(last_observation, "code", "") or ""),
        last_observation_summary=str(getattr(last_observation, "summary", "") or ""),
        escalation_type=str(state.get("escalation_type") or ""),
        escalation_reason=str(state.get("escalation_reason") or ""),
        escalation_context=(
            state.get("escalation_context").model_dump()
            if getattr(state.get("escalation_context"), "model_dump", None)
            else dict(state.get("escalation_context") or {})
        ),
        escalation_decision=str(state.get("escalation_decision") or ""),
        handoff_case_payload=dict(state.get("handoff_case_payload") or {}),
    )


def _policy_reply(data: FinalizerInput) -> str:
    if not data.retrieval_evidence:
        return "我先按现有规则库查了，但这次还没有拿到足够明确的规则依据。"
    summaries = [item.evidence_summary or item.title for item in data.retrieval_evidence[:2]]
    return "我帮你整理了一下相关规则：" + "；".join(filter(None, summaries)) + "。"


def _order_reply(data: FinalizerInput) -> str:
    action = data.order_action
    order_context = data.order_context or {}
    logistics = data.logistics_snapshot or {}
    aftersales = data.aftersales_context or {}

    if action == ActionType.QUERY_ORDER.value:
        if order_context:
            order_status = order_context.get("order_status") or "未知"
            pay_status = order_context.get("pay_status") or "未知"
            product_name = order_context.get("product_name") or ""
            product_text = f"，商品是{product_name}" if product_name else ""
            return (
                f"已帮你查到订单。当前订单状态为{order_status}，支付状态为{pay_status}"
                f"{product_text}。如需我继续帮你看物流或处理售后，也可以继续说。"
            )
        return data.last_observation_summary or "我已经开始帮你查订单了，但当前还缺少稳定的订单结果。"

    if action == ActionType.QUERY_LOGISTICS.value:
        if data.last_observation_code == "QUERY_TOO_FREQUENT":
            if logistics:
                status = logistics.get("delivery_state_name") or logistics.get("delivery_state") or "未知"
                last_event = logistics.get("last_event") or "暂无最新轨迹"
                last_event_time = logistics.get("last_event_time") or "暂无更新时间"
                location = logistics.get("current_location") or ""
                suffix = f"，位置在{location}" if location else ""
                return (
                    "该物流单号最近 30 分钟内已查询过，我先按最近一次缓存结果告诉你："
                    f"物流目前为{status}，最新一条轨迹是{last_event}，时间是{last_event_time}{suffix}。"
                )
            return "该物流单号最近 30 分钟内已查询过，为避免接口锁定，暂时不能重复调用。请稍后再试。"
        if logistics:
            status = logistics.get("delivery_state_name") or logistics.get("delivery_state") or "未知"
            last_event = logistics.get("last_event") or "暂无最新轨迹"
            last_event_time = logistics.get("last_event_time") or "暂无更新时间"
            location = logistics.get("current_location") or ""
            suffix = f"，位置在{location}" if location else ""
            return f"我查到这个订单的物流目前为{status}，最新一条轨迹是{last_event}，时间是{last_event_time}{suffix}。"
        if data.last_observation_code == "LOGISTICS_IDENTIFIERS_MISSING":
            return data.last_observation_summary or "我先帮你定位到了订单，但这笔订单暂时没有可用的物流单号或承运商信息。"
        if data.allowed_response_mode == ResponseMode.EXPLAIN_LIMIT:
            return data.last_observation_summary or "我已经定位到订单了，但这次物流服务暂时没返回稳定结果，建议稍后再试。"
        return data.last_observation_summary or "我先帮你定位到订单了，但暂时还没有拿到可用的物流结果。"

    if action == ActionType.CREATE_AFTERSALES.value:
        service_type = aftersales.get("service_type") or "售后"
        aftersales_id = aftersales.get("aftersales_id") or ""
        status = aftersales.get("aftersales_status") or ""
        denial_reason = aftersales.get("denial_reason") or ""
        next_action_hint = aftersales.get("next_action_hint") or ""
        if aftersales_id:
            return f"已为你提交{service_type}申请，售后单号为{aftersales_id}，当前状态为{status or '处理中'}。"
        if denial_reason:
            tail = f" 下一步建议是{next_action_hint}。" if next_action_hint else ""
            return f"这笔订单暂时还不能自动提交{service_type}申请，原因是{denial_reason}。{tail}".strip()
        if status:
            tail = f" 下一步建议是{next_action_hint}。" if next_action_hint else ""
            return f"这笔订单的{service_type}处理结果为{status}。{tail}".strip()
        return data.last_observation_summary or "售后信息我已经帮你整理好了，如需继续处理可以接着告诉我。"

    if action == ActionType.QUERY_AFTERSALES.value:
        aftersales_id = aftersales.get("aftersales_id") or ""
        status = aftersales.get("aftersales_status") or ""
        next_action_hint = aftersales.get("next_action_hint") or ""
        if aftersales_id or status:
            prefix = f"你的售后单 {aftersales_id} " if aftersales_id else "你的售后单目前"
            suffix = f"，下一步建议是{next_action_hint}" if next_action_hint else ""
            return f"{prefix}状态为{status or '处理中'}{suffix}。"
        return data.last_observation_summary or "当前这笔订单还没有查到售后记录。"

    return data.last_observation_summary or "我已经完成当前订单域处理。"


def _finalizer_fallback(data: FinalizerInput) -> str:
    pending_question = str(data.handoff_context.get("pending_question") or "")
    if data.allowed_response_mode == ResponseMode.ASK_USER:
        if pending_question:
            return pending_question
        missing = "、".join(data.missing_information) or "必要信息"
        return f"为了继续处理，请补充：{missing}。"

    if data.allowed_response_mode == ResponseMode.HANDOFF:
        reason = data.handoff_context.get("reason") or "当前问题需要人工进一步处理"
        return f"当前问题我已经为你转人工处理，原因是：{reason}。请稍候。"

    if data.customer_intent == IntentType.SMALLTALK:
        return "你好，我可以帮你查订单、看物流、处理售后，也可以帮你解释平台规则。"

    if data.customer_intent == IntentType.POLICY:
        return _policy_reply(data)

    if data.customer_intent == IntentType.MIXED:
        policy_part = _policy_reply(data)
        order_part = _order_reply(data)
        if policy_part and order_part:
            return f"{policy_part}\n\n{order_part}"
        return order_part or policy_part

    if data.customer_intent == IntentType.ORDER:
        return _order_reply(data)

    if data.allowed_response_mode == ResponseMode.EXPLAIN_LIMIT:
        return data.last_observation_summary or "我已经查到一部分结果，但当前证据还不足以给出完全确定的结论。"

    if data.retrieval_evidence:
        return _policy_reply(data)
    if data.order_action:
        return _order_reply(data)
    return data.last_observation_summary or "目前还没有足够证据支持明确结论。"


def _escalation_reply_v2(data: FinalizerInput) -> str:
    decision = data.escalation_decision
    if decision == "HANDOFF_HUMAN":
        return "我已经为你转入人工处理，并整理了当前问题的关键信息，后续客服会继续跟进。"
    if decision == "CREATE_CASE":
        return "我已为你提交升级处理记录，当前问题会按异常工单继续跟进。"
    if decision == "RETRY_AUTOMATION":
        return "我已经记录当前异常，并建议继续自动核查一次；如果你希望，我也可以继续帮你整理人工处理所需信息。"
    if decision == "FRIENDLY_FALLBACK":
        return "这个问题暂时无法自动完成处理，我建议你联系人工客服继续协助，我也可以帮你整理目前已确认的信息。"
    if data.escalation_reason:
        return f"当前问题已经进入异常升级处理，我已记录升级原因：{data.escalation_reason}"
    return "当前问题已经进入异常升级处理，我会继续按已确认的信息为你整理后续处理方向。"


def _finalizer_fallback_v2(data: FinalizerInput) -> str:
    pending_question = str(data.handoff_context.get("pending_question") or "")
    if data.allowed_response_mode == ResponseMode.ASK_USER:
        if pending_question:
            return pending_question
        missing = "、".join(data.missing_information) or "必要信息"
        return f"为了继续处理，请补充：{missing}。"

    if data.allowed_response_mode == ResponseMode.HANDOFF and not data.escalation_decision:
        reason = data.handoff_context.get("reason") or "当前问题需要人工进一步处理"
        return f"当前问题我已经为你转人工处理，原因是：{reason}。请稍候。"

    if data.escalation_decision or data.customer_intent == IntentType.ESCALATION:
        return _escalation_reply_v2(data)

    if data.customer_intent == IntentType.MIXED:
        policy_part = _policy_reply(data) if data.retrieval_evidence else ""
        order_part = _order_reply(data) if data.order_action or data.order_context else ""
        if policy_part and order_part:
            return f"{policy_part}\n\n{order_part}"
        return order_part or policy_part or (data.last_observation_summary or "")

    if data.customer_intent == IntentType.POLICY:
        return _policy_reply(data)
    if data.customer_intent == IntentType.ORDER:
        return _order_reply(data)
    if data.allowed_response_mode == ResponseMode.EXPLAIN_LIMIT:
        return data.last_observation_summary or "我已经查到一部分结果，但当前证据还不足以给出完全确定的结论。"
    if data.retrieval_evidence:
        return _policy_reply(data)
    if data.order_action:
        return _order_reply(data)
    return data.last_observation_summary or "目前还没有足够证据支持明确结论。"


@traceable(name="agent_v3_finalizer_node")
def finalizer_node(state: AgentState) -> Dict[str, Any]:
    patch = _with_node_tag(state, "finalizer_node")
    data = _build_finalizer_input(state)
    final_text = _finalizer_fallback_v2(data)
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
    final_text = f"当前问题我已经为你转人工处理，原因是：{reason}。请稍候。"
    patch["final_response"] = final_text
    patch["messages"] = [as_ai_message(final_text)]
    return patch


def _route_after_dispatch(state: AgentState) -> str:
    if state.get("awaited_slots"):
        return "verifier_node"
    if (
        state.get("intent_type") == IntentType.MIXED
        and _has_pending_owner_step(state, SpecialistName.ORDER)
        and _has_pending_owner_step(state, SpecialistName.POLICY)
    ):
        return "mixed_parallel_entry"
    step = get_current_step(state)
    if step is None:
        return "finalizer_node"
    if step.owner_agent == SpecialistName.GENERAL:
        return "general_node"
    if step.owner_agent == SpecialistName.ORDER:
        return "order_subgraph"
    if step.owner_agent == SpecialistName.POLICY:
        return "policy_subgraph"
    if step.owner_agent == SpecialistName.ESCALATION:
        return "escalation_subgraph"
    return "finalizer_node"


def _route_after_verifier(state: AgentState) -> str:
    verification = state.get("verification_status")
    if verification is None:
        return "planner_node"
    if verification.must_ask_user:
        return "await_user_node"
    if verification.should_escalate:
        return "planner_node"
    if verification.should_handoff:
        return "handoff_node"
    if verification.can_finalize:
        return "finalizer_node"
    if verification.retry_same_step:
        return "planner_node"
    if verification.should_replan:
        return "planner_node"
    return "finalizer_node"


def build_agent_graph():
    order_subgraph = build_order_subgraph()
    policy_subgraph = build_policy_subgraph()
    escalation_subgraph = build_escalation_subgraph()

    graph = StateGraph(AgentState)
    graph.add_node("ingest_node", ingest_node)
    graph.add_node("classify_node", classify_node)
    graph.add_node("planner_node", planner_node)
    graph.add_node("dispatch_node", dispatch_node)
    graph.add_node("general_node", general_node)
    graph.add_node("mixed_parallel_entry", mixed_parallel_entry_node)
    graph.add_node("mixed_order_runner", mixed_order_runner_node)
    graph.add_node("mixed_policy_runner", mixed_policy_runner_node)
    graph.add_node("mixed_join", mixed_join_node)
    graph.add_node("order_subgraph", order_subgraph)
    graph.add_node("policy_subgraph", policy_subgraph)
    graph.add_node("escalation_subgraph", escalation_subgraph)
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
            "mixed_parallel_entry": "mixed_parallel_entry",
            "order_subgraph": "order_subgraph",
            "policy_subgraph": "policy_subgraph",
            "escalation_subgraph": "escalation_subgraph",
            "verifier_node": "verifier_node",
            "finalizer_node": "finalizer_node",
        },
    )
    graph.add_edge("general_node", "verifier_node")
    graph.add_edge("mixed_parallel_entry", "mixed_order_runner")
    graph.add_edge("mixed_parallel_entry", "mixed_policy_runner")
    graph.add_edge("mixed_order_runner", "mixed_join")
    graph.add_edge("mixed_policy_runner", "mixed_join")
    graph.add_edge("mixed_join", "verifier_node")
    graph.add_edge("order_subgraph", "verifier_node")
    graph.add_edge("policy_subgraph", "verifier_node")
    graph.add_edge("escalation_subgraph", "verifier_node")
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
