from __future__ import annotations

import os
import uuid
from typing import Any, Dict, List

from langgraph.graph import END, START, StateGraph

from .langsmith_utils import traceable
from .rag_retriever import retrieve_policy_evidence, rewrite_query
from .state import (
    ActionType,
    AfterSalesContext,
    AgentState,
    EscalationContext,
    LogisticsSnapshot,
    Observation,
    ObservationSource,
    OrderContext,
    RetrievalEvidence,
    SpecialistName,
    StepStatus,
    VerifiedFact,
    append_observation,
    as_ai_message,
    extract_slots_from_text,
    get_current_step,
    mark_step_status,
    merge_slot_values,
    observation_to_fact_candidates,
    summarize_messages,
)
from .tool_registry import ToolSpec, build_tool_registry


TOOL_REGISTRY = build_tool_registry()
ACCEPTABLE_AFTERSALES_CODES = {"AFTERSALES_ALREADY_EXISTS", "AFTERSALES_NOT_ALLOWED", "AFTERSALES_NOT_FOUND"}
LOGISTICS_KEYWORDS = [
    "物流",
    "快递",
    "到哪",
    "签收",
    "派件",
    "在途",
    "拒签",
    "退回",
    "物流异常",
]


def _contains_any(text: str, keywords: List[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _with_node_tag(state: AgentState, node_name: str) -> Dict[str, Any]:
    tags = dict(state.get("trace_tags") or {})
    tags["current_node"] = node_name
    return {"trace_tags": tags}


def _with_trace_updates(state: AgentState, **kwargs: Any) -> Dict[str, Any]:
    tags = dict(state.get("trace_tags") or {})
    tags.update(kwargs)
    return {"trace_tags": tags}


def _make_clarification_patch(
    state: AgentState,
    step_id: str,
    missing_slots: List[str],
    source_name: str,
    *,
    question: str | None = None,
) -> Dict[str, Any]:
    prompt = question or f"为了继续处理，请补充：{'、'.join(missing_slots)}。"
    observation = Observation(
        step_id=step_id,
        source_type=ObservationSource.USER_CLARIFICATION,
        source_name=source_name,
        success=False,
        code="MISSING_SLOTS",
        summary=prompt,
        missing_slots=missing_slots,
        retryable=True,
    )
    patch = append_observation(state, observation)
    patch.update(
        {
            "awaited_slots": missing_slots,
            "blocked_step_id": step_id,
            "pending_question": prompt,
            "messages": [as_ai_message(prompt)],
            "current_plan": mark_step_status(state, step_id, StepStatus.BLOCKED),
        }
    )
    return patch


def _increment_tool_retry_counts(state: AgentState, tool_name: str) -> Dict[str, int]:
    counts = dict(state.get("tool_retry_counts") or {})
    counts[tool_name] = counts.get(tool_name, 0) + 1
    return counts


def _tool_observation(step_id: str, tool_name: str, result: Dict[str, Any], retryable: bool) -> Observation:
    data = (result or {}).get("data") or {}
    missing_slots = list((result or {}).get("missing_slots") or data.get("missing_slots") or [])
    return Observation(
        step_id=step_id,
        source_type=ObservationSource.TOOL,
        source_name=tool_name,
        success=bool((result or {}).get("success")),
        code=str((result or {}).get("code") or ""),
        summary=str((result or {}).get("message") or ""),
        structured_data=data if isinstance(data, dict) else {},
        evidence_refs=[],
        missing_slots=missing_slots,
        retryable=retryable,
    )


def _fact_list_from_retrieval(step_id: str, evidence: List[RetrievalEvidence]) -> List[VerifiedFact]:
    return [
        VerifiedFact(
            step_id=step_id,
            source_type=ObservationSource.RETRIEVAL.value,
            title=item.title,
            detail=item.evidence_summary,
            reference_id=item.chunk_id,
        )
        for item in evidence[:2]
    ]


def _context_to_facts(step_id: str, source_name: str, payload: Dict[str, Any]) -> List[VerifiedFact]:
    facts: List[VerifiedFact] = []
    for key, value in payload.items():
        if value in (None, "", [], {}):
            continue
        facts.append(
            VerifiedFact(
                step_id=step_id,
                source_type=ObservationSource.JUDGEMENT.value,
                title=key,
                detail=str(value),
                reference_id=f"{source_name}:{key}",
            )
        )
    return facts


def _build_order_context(data: Dict[str, Any]) -> OrderContext:
    amount = data.get("amount")
    try:
        normalized_amount = float(amount) if amount not in (None, "") else None
    except Exception:
        normalized_amount = None

    can_apply = data.get("can_apply_aftersales")
    try:
        normalized_can_apply = int(can_apply) if can_apply not in (None, "") else None
    except Exception:
        normalized_can_apply = None

    return OrderContext(
        order_id=str(data.get("order_id") or ""),
        product_name=str(data.get("product_name") or ""),
        amount=normalized_amount,
        order_status=str(data.get("order_status") or ""),
        pay_status=str(data.get("pay_status") or ""),
        created_at=str(data.get("created_at") or ""),
        carrier_code=str(data.get("carrier_code") or ""),
        tracking_no=str(data.get("tracking_no") or ""),
        phone_last4=str(data.get("phone_last4") or ""),
        can_apply_aftersales=normalized_can_apply,
        source=str(data.get("source") or "order_tool"),
    )


def _build_logistics_snapshot(data: Dict[str, Any]) -> LogisticsSnapshot:
    return LogisticsSnapshot(
        carrier_code=str(data.get("carrier_code") or ""),
        tracking_no=str(data.get("tracking_no") or ""),
        delivery_state=str(data.get("delivery_state") or ""),
        delivery_state_name=str(data.get("delivery_state_name") or ""),
        delivery_status_code=str(data.get("delivery_status_code") or ""),
        last_event=str(data.get("last_event") or ""),
        last_event_time=str(data.get("last_event_time") or ""),
        current_location=str(data.get("current_location") or ""),
        route_from=str(data.get("route_from") or ""),
        route_to=str(data.get("route_to") or ""),
        is_signed=bool(data.get("is_signed")),
        is_returning=bool(data.get("is_returning")),
        is_abnormal=bool(data.get("is_abnormal")),
        route_info=data.get("route_info") if isinstance(data.get("route_info"), dict) else {},
        arrival_time=str(data.get("arrival_time") or ""),
        predicted_route=data.get("predicted_route") if isinstance(data.get("predicted_route"), list) else [],
        source=str(data.get("source") or "logistics_tool"),
        fetched_at=str(data.get("fetched_at") or ""),
        raw_payload_ref=str(data.get("raw_payload_ref") or ""),
    )


def _build_aftersales_context_from_tool(
    *,
    data: Dict[str, Any],
    slot_values: Dict[str, Any],
    trace_tags: Dict[str, Any],
    fallback_source: str,
) -> AfterSalesContext:
    return AfterSalesContext(
        aftersales_id=str(data.get("ticket_id") or data.get("aftersales_id") or ""),
        order_id=str(data.get("order_id") or slot_values.get("order_id") or ""),
        service_type=str(data.get("service_type") or slot_values.get("service_type") or ""),
        reason=str(data.get("reason") or slot_values.get("reason") or ""),
        eligibility=str(trace_tags.get("aftersales_eligibility") or ""),
        aftersales_status=str(data.get("ticket_status") or data.get("aftersales_status") or ""),
        denial_reason=str(trace_tags.get("aftersales_denial_reason") or ""),
        policy_hint=str(data.get("policy_hint") or trace_tags.get("aftersales_policy_hint") or ""),
        next_action_hint=str(data.get("next_action_hint") or trace_tags.get("aftersales_next_action_hint") or ""),
        created_at=str(data.get("created_at") or ""),
        updated_at=str(data.get("updated_at") or ""),
        source=str(data.get("source") or fallback_source),
    )


def _normalize_service_type(value: str) -> str:
    raw = (value or "").strip().lower()
    if raw in {"refund", "退款", "退钱"}:
        return "退款"
    if raw in {"return", "退货"}:
        return "退货"
    if raw in {"exchange", "换货"}:
        return "换货"
    return value.strip()


def _detect_order_action(state: AgentState) -> ActionType:
    step = get_current_step(state)
    if step and step.action_type in {
        ActionType.QUERY_ORDER,
        ActionType.QUERY_LOGISTICS,
        ActionType.CREATE_AFTERSALES,
        ActionType.QUERY_AFTERSALES,
    }:
        return step.action_type

    text = state.get("user_input") or ""
    slot_values = dict(state.get("slot_values") or {})
    if any(word in text for word in LOGISTICS_KEYWORDS):
        return ActionType.QUERY_LOGISTICS
    if any(word in text for word in ["退款", "退货", "换货", "售后", "退钱"]) or slot_values.get("service_type"):
        if any(word in text for word in ["进度", "状态", "售后单", "处理到哪", "查询售后"]):
            return ActionType.QUERY_AFTERSALES
        return ActionType.CREATE_AFTERSALES
    return ActionType.QUERY_ORDER


def _sync_slot_values_with_order_context(slot_values: Dict[str, Any], context: OrderContext) -> Dict[str, Any]:
    merged = dict(slot_values)
    if context.order_id:
        merged["order_id"] = context.order_id
    if context.phone_last4:
        merged["phone_last4"] = context.phone_last4
    if context.carrier_code:
        merged["carrier_code"] = context.carrier_code
    if context.tracking_no:
        merged["tracking_no"] = context.tracking_no
    return merged


def _order_context_data(state: AgentState) -> Dict[str, Any]:
    context = state.get("order_context")
    return context.model_dump() if isinstance(context, OrderContext) else {}


def _logistics_snapshot_data(state: AgentState) -> Dict[str, Any]:
    snapshot = state.get("logistics_snapshot")
    return snapshot.model_dump() if isinstance(snapshot, LogisticsSnapshot) else {}


def _matching_logistics_snapshot(
    state: AgentState,
    *,
    carrier_code: str,
    tracking_no: str,
) -> LogisticsSnapshot | None:
    snapshot = state.get("logistics_snapshot")
    if not isinstance(snapshot, LogisticsSnapshot):
        return None
    if carrier_code and snapshot.carrier_code and snapshot.carrier_code != carrier_code:
        return None
    if tracking_no and snapshot.tracking_no and snapshot.tracking_no != tracking_no:
        return None
    return snapshot


def _order_context_matches(context: OrderContext | None, order_id: str, phone_last4: str) -> bool:
    if not isinstance(context, OrderContext):
        return False
    return context.order_id == order_id and context.phone_last4 == phone_last4


def _missing_identity_slots(slot_values: Dict[str, Any]) -> List[str]:
    missing: List[str] = []
    if not slot_values.get("order_id"):
        missing.append("order_id")
    if not slot_values.get("phone_last4"):
        missing.append("phone_last4")
    return missing


@traceable(name="agent_v3_order_intent_understand")
def order_intent_understand_node(state: AgentState) -> Dict[str, Any]:
    step = get_current_step(state)
    patch = _with_node_tag(state, "order_intent_understand")
    if step is None:
        return patch

    action = _detect_order_action(state)
    observation = Observation(
        step_id=step.step_id,
        source_type=ObservationSource.JUDGEMENT,
        source_name="order_intent_understand",
        success=True,
        code="ORDER_ACTION_READY",
        summary=f"订单域动作已识别为 {action.value}。",
        structured_data={"order_action": action.value},
    )
    patch.update(append_observation(state, observation))
    patch.update(
        {
            "active_agent": SpecialistName.ORDER,
            "order_action": action,
            "current_plan": mark_step_status(state, step.step_id, StepStatus.RUNNING),
        }
    )
    patch.update(_with_trace_updates(state, order_action=action.value))
    return patch


@traceable(name="agent_v3_order_identity_gate")
def order_identity_gate_node(state: AgentState) -> Dict[str, Any]:
    step = get_current_step(state)
    patch = _with_node_tag(state, "order_identity_gate")
    if step is None:
        return patch

    slot_values = dict(state.get("slot_values") or {})
    context = state.get("order_context")
    if isinstance(context, OrderContext):
        if not slot_values.get("order_id") and context.order_id:
            slot_values["order_id"] = context.order_id
        if not slot_values.get("phone_last4") and context.phone_last4:
            slot_values["phone_last4"] = context.phone_last4

    missing_slots = _missing_identity_slots(slot_values)
    if missing_slots:
        patch.update(
            _make_clarification_patch(
                state,
                step.step_id,
                missing_slots,
                "order_identity_gate",
                question="请提供订单号和收件人手机号后四位，我先帮你定位订单。",
            )
        )
        return patch

    patch.update(
        {
            "slot_values": slot_values,
            "awaited_slots": [],
            "blocked_step_id": "",
            "pending_question": "",
        }
    )
    return patch


@traceable(name="agent_v3_order_profile_lookup")
def order_profile_lookup_node(state: AgentState) -> Dict[str, Any]:
    step = get_current_step(state)
    patch = _with_node_tag(state, "order_profile_lookup")
    if step is None:
        return patch

    slot_values = dict(state.get("slot_values") or {})
    order_id = str(slot_values.get("order_id") or "")
    phone_last4 = str(slot_values.get("phone_last4") or "")
    cached_context = state.get("order_context")

    if _order_context_matches(cached_context, order_id, phone_last4):
        context = cached_context
        observation = Observation(
            step_id=step.step_id,
            source_type=ObservationSource.JUDGEMENT,
            source_name="order_profile_lookup",
            success=True,
            code="ORDER_PROFILE_REUSED",
            summary="已复用当前订单上下文。",
            structured_data=context.model_dump(),
        )
        patch.update(append_observation(state, observation))
        patch["slot_values"] = _sync_slot_values_with_order_context(slot_values, context)
        patch["verified_facts"] = list(state.get("verified_facts") or []) + _context_to_facts(
            step.step_id,
            "order_profile_lookup",
            context.model_dump(),
        )
        return patch

    spec: ToolSpec = TOOL_REGISTRY["get_order_info_tool"]
    result = spec.callable(order_id=order_id, phone_last4=phone_last4)
    observation = _tool_observation(step.step_id, spec.name, result, spec.retryable)
    patch.update(append_observation(state, observation))
    patch["tool_retry_counts"] = _increment_tool_retry_counts(state, spec.name)

    if observation.success:
        context = _build_order_context(observation.structured_data)
        patch["order_context"] = context
        patch["slot_values"] = _sync_slot_values_with_order_context(slot_values, context)
        patch["verified_facts"] = list(state.get("verified_facts") or []) + observation_to_fact_candidates(observation)
        return patch

    if observation.code in {"PHONE_MISMATCH", "ORDER_NOT_FOUND"}:
        patch.update(
            {
                "awaited_slots": ["order_id", "phone_last4"],
                "blocked_step_id": step.step_id,
                "pending_question": "订单号或手机号后四位和系统记录不一致，请重新提供，我再帮你核对。",
                "messages": [as_ai_message("订单号或手机号后四位和系统记录不一致，请重新提供，我再帮你核对。")],
                "current_plan": mark_step_status(state, step.step_id, StepStatus.BLOCKED),
            }
        )
    else:
        patch["current_plan"] = mark_step_status(state, step.step_id, StepStatus.FAILED)
    return patch


@traceable(name="agent_v3_order_action_router")
def order_action_router_node(state: AgentState) -> Dict[str, Any]:
    patch = _with_node_tag(state, "order_action_router")
    action = state.get("order_action")
    if action is None:
        action = _detect_order_action(state)
        patch["order_action"] = action
    patch.update(
        _with_trace_updates(
            state,
            aftersales_requires_logistics=bool(action == ActionType.CREATE_AFTERSALES),
        )
    )
    return patch


@traceable(name="agent_v3_order_summary_interpret")
def order_summary_interpret_node(state: AgentState) -> Dict[str, Any]:
    step = get_current_step(state)
    patch = _with_node_tag(state, "order_summary_interpret")
    context = state.get("order_context")
    if step is None or not isinstance(context, OrderContext):
        return patch

    summary = f"已定位订单，当前订单状态为 {context.order_status or '未知'}，支付状态为 {context.pay_status or '未知'}。"
    observation = Observation(
        step_id=step.step_id,
        source_type=ObservationSource.JUDGEMENT,
        source_name="order_summary_interpret",
        success=True,
        code="ORDER_SUMMARY_READY",
        summary=summary,
        structured_data={
            "order_id": context.order_id,
            "product_name": context.product_name,
            "order_status": context.order_status,
            "pay_status": context.pay_status,
            "created_at": context.created_at,
        },
    )
    patch.update(append_observation(state, observation))
    patch["verified_facts"] = list(state.get("verified_facts") or []) + _context_to_facts(
        step.step_id,
        "order_summary_interpret",
        observation.structured_data,
    )
    patch["current_plan"] = mark_step_status(state, step.step_id, StepStatus.COMPLETED)
    return patch


@traceable(name="agent_v3_logistics_snapshot_lookup")
def logistics_snapshot_lookup_node(state: AgentState) -> Dict[str, Any]:
    step = get_current_step(state)
    patch = _with_node_tag(state, "logistics_snapshot_lookup")
    if step is None:
        return patch

    slot_values = dict(state.get("slot_values") or {})
    context_data = _order_context_data(state)
    carrier_code = str(slot_values.get("carrier_code") or context_data.get("carrier_code") or "").strip()
    tracking_no = str(slot_values.get("tracking_no") or context_data.get("tracking_no") or "").strip()
    phone_last4 = str(slot_values.get("phone_last4") or context_data.get("phone_last4") or "").strip()

    if not carrier_code or not tracking_no:
        observation = Observation(
            step_id=step.step_id,
            source_type=ObservationSource.JUDGEMENT,
            source_name="logistics_snapshot_lookup",
            success=True,
            code="LOGISTICS_IDENTIFIERS_MISSING",
            summary="我已经帮你定位到订单，但这笔订单暂时没有可用的物流单号或承运商信息。",
            structured_data={
                "carrier_code": carrier_code,
                "tracking_no": tracking_no,
                "phone_last4": phone_last4,
            },
        )
        patch.update(append_observation(state, observation))
        patch["logistics_snapshot"] = None
        patch["logistics_cache_meta"] = {}
        return patch

    spec: ToolSpec = TOOL_REGISTRY["query_logistics_snapshot_tool"]
    result = spec.callable(
        carrier_code=carrier_code,
        tracking_no=tracking_no,
        phone_last4=phone_last4,
    )
    observation = _tool_observation(step.step_id, spec.name, result, spec.retryable)
    patch.update(append_observation(state, observation))
    patch["tool_retry_counts"] = _increment_tool_retry_counts(state, spec.name)

    if observation.success:
        snapshot = _build_logistics_snapshot(observation.structured_data)
        cache_meta = (
            observation.structured_data.get("_cache_meta")
            if isinstance(observation.structured_data.get("_cache_meta"), dict)
            else {}
        )
        patch["logistics_snapshot"] = snapshot
        patch["logistics_cache_meta"] = cache_meta or {
            "cache_key": f"{snapshot.carrier_code}:{snapshot.tracking_no}",
            "cache_hit": False,
            "last_query_at": snapshot.fetched_at,
            "ttl_minutes": 30,
        }
        patch["verified_facts"] = list(state.get("verified_facts") or []) + observation_to_fact_candidates(observation)
        return patch

    cache_meta = (
        observation.structured_data.get("_cache_meta")
        if isinstance(observation.structured_data.get("_cache_meta"), dict)
        else {}
    )
    preserved_snapshot = None
    if observation.code == "QUERY_TOO_FREQUENT":
        preserved_snapshot = _matching_logistics_snapshot(
            state,
            carrier_code=carrier_code,
            tracking_no=tracking_no,
        )
    patch["logistics_snapshot"] = preserved_snapshot
    patch["logistics_cache_meta"] = cache_meta
    if observation.code == "408" or observation.missing_slots:
        missing_slots = observation.missing_slots or ["phone_last4"]
        patch.update(
            {
                "awaited_slots": missing_slots,
                "blocked_step_id": step.step_id,
                "pending_question": "为了继续查询物流，请再提供一下收件人手机号后四位。",
                "messages": [as_ai_message("为了继续查询物流，请再提供一下收件人手机号后四位。")],
                "current_plan": mark_step_status(state, step.step_id, StepStatus.BLOCKED),
            }
        )
    else:
        patch["current_plan"] = mark_step_status(state, step.step_id, StepStatus.FAILED)
    return patch


@traceable(name="agent_v3_logistics_interpret")
def logistics_interpret_node(state: AgentState) -> Dict[str, Any]:
    step = get_current_step(state)
    patch = _with_node_tag(state, "logistics_interpret")
    if step is None:
        return patch

    snapshot = state.get("logistics_snapshot")
    last_observation = state.get("last_observation")
    if isinstance(snapshot, LogisticsSnapshot):
        summary = (
            f"物流当前状态为 {snapshot.delivery_state_name or snapshot.delivery_state or '未知'}，"
            f"最新轨迹为 {snapshot.last_event or '暂无'}，时间 {snapshot.last_event_time or '暂无'}。"
        )
        structured_data = {
            "delivery_state": snapshot.delivery_state,
            "delivery_state_name": snapshot.delivery_state_name,
            "last_event": snapshot.last_event,
            "last_event_time": snapshot.last_event_time,
            "current_location": snapshot.current_location,
            "is_signed": snapshot.is_signed,
            "is_abnormal": snapshot.is_abnormal,
        }
    else:
        summary = (
            last_observation.summary
            if isinstance(last_observation, Observation)
            else "这笔订单暂时还没有可用的物流信息。"
        )
        structured_data = {"delivery_state_name": "", "last_event": "", "last_event_time": ""}

    observation = Observation(
        step_id=step.step_id,
        source_type=ObservationSource.JUDGEMENT,
        source_name="logistics_interpret",
        success=True,
        code="LOGISTICS_RESULT_READY",
        summary=summary,
        structured_data=structured_data,
    )
    patch.update(append_observation(state, observation))
    patch["verified_facts"] = list(state.get("verified_facts") or []) + _context_to_facts(
        step.step_id,
        "logistics_interpret",
        structured_data,
    )
    patch["current_plan"] = mark_step_status(state, step.step_id, StepStatus.COMPLETED)
    return patch


def _aftersales_question(service_type: str, missing_slots: List[str]) -> str:
    if missing_slots == ["reason"]:
        prefix = service_type or "售后"
        return f"请告诉我{prefix}原因，比如不想要了、不合适、买错了等，我再继续帮你提交。"
    if missing_slots == ["service_type"]:
        return "请告诉我是想退款、退货还是换货，我再继续帮你处理。"
    return "请告诉我是想退款、退货还是换货，以及具体原因，我再继续帮你处理。"


@traceable(name="agent_v3_aftersales_prepare")
def aftersales_prepare_node(state: AgentState) -> Dict[str, Any]:
    step = get_current_step(state)
    patch = _with_node_tag(state, "aftersales_prepare")
    if step is None:
        return patch

    extracted = extract_slots_from_text(state.get("user_input", ""), awaited_slots=state.get("awaited_slots") or [])
    slot_values = merge_slot_values(state, extracted)

    service_type = _normalize_service_type(str(slot_values.get("service_type") or ""))
    if service_type:
        slot_values["service_type"] = service_type
    reason = str(slot_values.get("reason") or "").strip()
    if reason:
        slot_values["reason"] = reason

    missing_slots: List[str] = []
    if not slot_values.get("service_type"):
        missing_slots.append("service_type")
    if not slot_values.get("reason"):
        missing_slots.append("reason")

    patch["slot_values"] = slot_values
    if missing_slots:
        patch.update(
            _make_clarification_patch(
                state,
                step.step_id,
                missing_slots,
                "aftersales_prepare",
                question=_aftersales_question(service_type, missing_slots),
            )
        )
        return patch

    observation = Observation(
        step_id=step.step_id,
        source_type=ObservationSource.JUDGEMENT,
        source_name="aftersales_prepare",
        success=True,
        code="AFTERSALES_PREPARED",
        summary=f"已补齐售后申请信息：{slot_values['service_type']}，原因：{slot_values['reason']}。",
        structured_data={
            "service_type": slot_values["service_type"],
            "reason": slot_values["reason"],
        },
    )
    patch.update(append_observation(state, observation))
    return patch


@traceable(name="agent_v3_aftersales_eligibility_check")
def eligibility_check_node(state: AgentState) -> Dict[str, Any]:
    step = get_current_step(state)
    patch = _with_node_tag(state, "aftersales_eligibility_check")
    if step is None:
        return patch

    trace_tags = dict(state.get("trace_tags") or {})
    slot_values = dict(state.get("slot_values") or {})
    order_data = _order_context_data(state)
    logistics_data = _logistics_snapshot_data(state)
    service_type = str(slot_values.get("service_type") or "")
    last_observation = state.get("last_observation")

    eligibility = "ALLOWED"
    denial_reason = ""
    policy_hint = ""
    next_action_hint = ""

    if int(order_data.get("can_apply_aftersales") or 0) != 1:
        eligibility = "NOT_ALLOWED"
        denial_reason = "这笔订单当前不满足自动售后条件。"
        policy_hint = "订单主数据已标记为不可申请售后。"
        next_action_hint = "如果你希望我继续协助，可以联系人工客服进一步核实。"
    elif logistics_data:
        delivery_state = str(logistics_data.get("delivery_state") or "unknown").strip().lower()
        if delivery_state in {"signed", "received"} or bool(logistics_data.get("is_signed")):
            eligibility = "ALLOWED"
        elif delivery_state in {"in_transit", "delivering"}:
            eligibility = "NOT_ALLOWED"
            denial_reason = "\u5546\u54c1\u76ee\u524d\u4ecd\u5728\u8fd0\u8f93\u6216\u6d3e\u9001\u4e2d\uff0c\u6682\u4e0d\u652f\u6301\u81ea\u52a8\u63d0\u4ea4\u9000\u8d27/\u9000\u6b3e\u7533\u8bf7\u3002"
            policy_hint = "\u5728\u9014\u6216\u6d3e\u9001\u4e2d\u7684\u8ba2\u5355\u9700\u7b49\u5f85\u7269\u6d41\u72b6\u6001\u66f4\u660e\u786e\u540e\u518d\u5904\u7406\u552e\u540e\u3002"
            next_action_hint = "\u5efa\u8bae\u7b49\u7269\u6d41\u7b7e\u6536\u540e\u518d\u8bd5\uff0c\u6216\u8054\u7cfb\u4eba\u5de5\u5ba2\u670d\u7ee7\u7eed\u6838\u5b9e\u3002"
        elif delivery_state in {"returning", "reject", "abnormal", "not_found", "unknown"} or bool(
            logistics_data.get("is_returning") or logistics_data.get("is_abnormal")
        ):
            eligibility = "NEED_MANUAL"
            denial_reason = "\u5f53\u524d\u7269\u6d41\u72b6\u6001\u5f02\u5e38\u6216\u65e0\u6cd5\u786e\u8ba4\uff0c\u6682\u4e0d\u5efa\u8bae\u76f4\u63a5\u81ea\u52a8\u63d0\u4ea4\u552e\u540e\u7533\u8bf7\u3002"
            policy_hint = "\u5f02\u5e38\u3001\u9000\u56de\u6216\u65e0\u6cd5\u786e\u8ba4\u7684\u7269\u6d41\u72b6\u6001\u4e0d\u9002\u5408\u76f4\u63a5\u81ea\u52a8\u5224\u5b9a\u3002"
            next_action_hint = "\u5efa\u8bae\u8054\u7cfb\u4eba\u5de5\u5ba2\u670d\u7ee7\u7eed\u5904\u7406\u3002"
    elif isinstance(last_observation, Observation) and last_observation.code == "LOGISTICS_IDENTIFIERS_MISSING":
        eligibility = "NEED_MANUAL"
        denial_reason = "\u8fd9\u7b14\u8ba2\u5355\u6682\u65f6\u6ca1\u6709\u53ef\u7528\u7684\u7269\u6d41\u5355\u53f7\u6216\u627f\u8fd0\u5546\u4fe1\u606f\uff0c\u6682\u4e0d\u80fd\u81ea\u52a8\u63d0\u4ea4\u552e\u540e\u7533\u8bf7\u3002"
        policy_hint = "\u7f3a\u5c11\u53ef\u7528\u7684\u7269\u6d41\u4e3b\u6570\u636e\u65f6\uff0c\u9700\u8981\u4eba\u5de5\u518d\u8fdb\u4e00\u6b65\u6838\u5b9e\u3002"
        next_action_hint = "\u5efa\u8bae\u8054\u7cfb\u4eba\u5de5\u5ba2\u670d\u534f\u52a9\u786e\u8ba4\u540e\u518d\u5904\u7406\u3002"

    observation = Observation(
        step_id=step.step_id,
        source_type=ObservationSource.JUDGEMENT,
        source_name="aftersales_eligibility_check",
        success=True,
        code="AFTERSALES_ELIGIBILITY_READY",
        summary=f"售后资格判断结果为 {eligibility}。",
        structured_data={
            "eligibility": eligibility,
            "denial_reason": denial_reason,
            "policy_hint": policy_hint,
            "next_action_hint": next_action_hint,
            "service_type": service_type,
        },
    )
    patch.update(append_observation(state, observation))
    patch.update(
        _with_trace_updates(
            state,
            aftersales_eligibility=eligibility,
            aftersales_denial_reason=denial_reason,
            aftersales_policy_hint=policy_hint,
            aftersales_next_action_hint=next_action_hint,
        )
    )
    return patch


def _acceptable_aftersales_context(
    code: str,
    slot_values: Dict[str, Any],
    trace_tags: Dict[str, Any],
    message: str,
) -> AfterSalesContext:
    if code == "AFTERSALES_ALREADY_EXISTS":
        return AfterSalesContext(
            order_id=str(slot_values.get("order_id") or ""),
            service_type=str(slot_values.get("service_type") or ""),
            reason=str(slot_values.get("reason") or ""),
            eligibility="ALLOWED",
            aftersales_status="已有进行中的售后申请",
            denial_reason="",
            next_action_hint="如需了解当前处理进度，我也可以继续帮你查询。",
            source="create_aftersales_tool",
        )
    if code == "AFTERSALES_NOT_FOUND":
        return AfterSalesContext(
            order_id=str(slot_values.get("order_id") or ""),
            service_type="",
            reason="",
            eligibility="QUERY_ONLY",
            aftersales_status="未找到售后记录",
            denial_reason="",
            next_action_hint="如果你想发起退款、退货或换货，我也可以继续帮你处理。",
            source="query_aftersales_tool",
        )
    return AfterSalesContext(
        order_id=str(slot_values.get("order_id") or ""),
        service_type=str(slot_values.get("service_type") or ""),
        reason=str(slot_values.get("reason") or ""),
        eligibility=str(trace_tags.get("aftersales_eligibility") or "NOT_ALLOWED"),
        aftersales_status="不可创建",
        denial_reason=message or str(trace_tags.get("aftersales_denial_reason") or ""),
        policy_hint=str(trace_tags.get("aftersales_policy_hint") or ""),
        next_action_hint=str(trace_tags.get("aftersales_next_action_hint") or ""),
        source="aftersales_tool",
    )


@traceable(name="agent_v3_aftersales_create_exec")
def aftersales_create_exec_node(state: AgentState) -> Dict[str, Any]:
    step = get_current_step(state)
    patch = _with_node_tag(state, "aftersales_create_exec")
    if step is None:
        return patch

    slot_values = dict(state.get("slot_values") or {})
    trace_tags = dict(state.get("trace_tags") or {})
    eligibility = str(trace_tags.get("aftersales_eligibility") or "")
    denial_reason = str(trace_tags.get("aftersales_denial_reason") or "")
    policy_hint = str(trace_tags.get("aftersales_policy_hint") or "")
    next_action_hint = str(trace_tags.get("aftersales_next_action_hint") or "")

    if eligibility == "ALLOWED":
        spec = TOOL_REGISTRY["create_aftersales_tool"]
        result = spec.callable(
            order_id=slot_values.get("order_id"),
            phone_last4=slot_values.get("phone_last4"),
            service_type=slot_values.get("service_type"),
            reason=slot_values.get("reason"),
        )
        observation = _tool_observation(step.step_id, spec.name, result, spec.retryable)
        patch.update(append_observation(state, observation))
        patch["tool_retry_counts"] = _increment_tool_retry_counts(state, spec.name)
        if observation.success:
            patch["aftersales_context"] = _build_aftersales_context_from_tool(
                data=observation.structured_data,
                slot_values=slot_values,
                trace_tags=trace_tags,
                fallback_source="create_aftersales_tool",
            )
        elif observation.code in ACCEPTABLE_AFTERSALES_CODES:
            patch["aftersales_context"] = _acceptable_aftersales_context(
                observation.code,
                slot_values,
                trace_tags,
                observation.summary,
            )
        elif observation.code in {"DB_ERROR", "DB_NOT_FOUND", "TOOL_ERROR"}:
            patch["handoff_reason"] = "售后创建工具执行失败，需要人工介入。"
            patch["current_plan"] = mark_step_status(state, step.step_id, StepStatus.FAILED)
        else:
            patch["current_plan"] = mark_step_status(state, step.step_id, StepStatus.FAILED)
        return patch

    context = AfterSalesContext(
        order_id=str(slot_values.get("order_id") or ""),
        service_type=str(slot_values.get("service_type") or ""),
        reason=str(slot_values.get("reason") or ""),
        eligibility=eligibility or "NOT_ALLOWED",
        aftersales_status="manual_review" if eligibility == "NEED_MANUAL" else "not_created",
        denial_reason=denial_reason,
        policy_hint=policy_hint,
        next_action_hint=next_action_hint,
        source="aftersales_eligibility_check",
    )
    blocked_code = "AFTERSALES_NOT_ALLOWED" if eligibility == "NOT_ALLOWED" else "AFTERSALES_NEED_MANUAL"
    observation = Observation(
        step_id=step.step_id,
        source_type=ObservationSource.JUDGEMENT,
        source_name="aftersales_create_exec",
        success=True,
        code=blocked_code,
        summary=denial_reason or "当前暂不满足自动创建售后的条件。",
        structured_data=context.model_dump(),
    )
    patch.update(append_observation(state, observation))
    patch["aftersales_context"] = context
    return patch


@traceable(name="agent_v3_aftersales_query_exec")
def aftersales_query_exec_node(state: AgentState) -> Dict[str, Any]:
    step = get_current_step(state)
    patch = _with_node_tag(state, "aftersales_query_exec")
    if step is None:
        return patch

    slot_values = dict(state.get("slot_values") or {})
    trace_tags = dict(state.get("trace_tags") or {})
    spec = TOOL_REGISTRY["query_aftersales_tool"]
    result = spec.callable(order_id=slot_values.get("order_id"), phone_last4=slot_values.get("phone_last4"))
    observation = _tool_observation(step.step_id, spec.name, result, spec.retryable)
    patch.update(append_observation(state, observation))
    patch["tool_retry_counts"] = _increment_tool_retry_counts(state, spec.name)

    if observation.success:
        patch["aftersales_context"] = _build_aftersales_context_from_tool(
            data=observation.structured_data,
            slot_values=slot_values,
            trace_tags=trace_tags,
            fallback_source="query_aftersales_tool",
        )
    elif observation.code in ACCEPTABLE_AFTERSALES_CODES:
        patch["aftersales_context"] = _acceptable_aftersales_context(
            observation.code,
            slot_values,
            trace_tags,
            observation.summary,
        )
    elif observation.code in {"DB_ERROR", "DB_NOT_FOUND", "TOOL_ERROR"}:
        patch["handoff_reason"] = "售后查询工具执行失败，需要人工介入。"
        patch["current_plan"] = mark_step_status(state, step.step_id, StepStatus.FAILED)
    else:
        patch["current_plan"] = mark_step_status(state, step.step_id, StepStatus.FAILED)
    return patch


def _aftersales_summary(context: AfterSalesContext, action: ActionType) -> str:
    if action == ActionType.CREATE_AFTERSALES:
        if context.aftersales_id:
            return f"已为用户提交 {context.service_type or '售后'} 申请，售后单号为 {context.aftersales_id}，当前状态为 {context.aftersales_status or '处理中'}。"
        if context.denial_reason:
            if context.next_action_hint:
                return f"{context.denial_reason} 下一步建议：{context.next_action_hint}"
            return context.denial_reason
        if context.aftersales_status:
            return f"这笔订单当前的售后处理结果为：{context.aftersales_status}。"
        return "售后信息已更新。"

    if context.aftersales_status == "未找到售后记录" and not context.aftersales_id:
        return "当前这笔订单还没有查到售后记录。"
    if context.aftersales_id:
        base = f"售后单 {context.aftersales_id} 当前状态为 {context.aftersales_status or '处理中'}。"
    else:
        base = f"售后当前状态为 {context.aftersales_status or '处理中'}。"
    if context.next_action_hint:
        return f"{base} 下一步建议：{context.next_action_hint}"
    return base


@traceable(name="agent_v3_aftersales_interpret")
def aftersales_interpret_node(state: AgentState) -> Dict[str, Any]:
    step = get_current_step(state)
    patch = _with_node_tag(state, "aftersales_interpret")
    if step is None:
        return patch

    slot_values = dict(state.get("slot_values") or {})
    trace_tags = dict(state.get("trace_tags") or {})
    last_observation = state.get("last_observation")
    context = state.get("aftersales_context")

    if not isinstance(context, AfterSalesContext) and isinstance(last_observation, Observation):
        if last_observation.source_type == ObservationSource.TOOL:
            context = _build_aftersales_context_from_tool(
                data=last_observation.structured_data,
                slot_values=slot_values,
                trace_tags=trace_tags,
                fallback_source=last_observation.source_name,
            )
        else:
            context = AfterSalesContext(**(last_observation.structured_data or {}))

    if not isinstance(context, AfterSalesContext):
        return patch

    summary = _aftersales_summary(context, step.action_type)
    observation = Observation(
        step_id=step.step_id,
        source_type=ObservationSource.JUDGEMENT,
        source_name="aftersales_interpret",
        success=True,
        code="AFTERSALES_RESULT_READY",
        summary=summary,
        structured_data=context.model_dump(),
    )
    patch.update(append_observation(state, observation))
    patch["aftersales_context"] = context
    patch["verified_facts"] = list(state.get("verified_facts") or []) + _context_to_facts(
        step.step_id,
        "aftersales_interpret",
        context.model_dump(),
    )
    patch["current_plan"] = mark_step_status(state, step.step_id, StepStatus.COMPLETED)
    return patch


def _route_order_after_identity_gate(state: AgentState) -> str:
    if state.get("awaited_slots"):
        return END
    return "order_profile_lookup"


def _route_order_after_profile_lookup(state: AgentState) -> str:
    observation = state.get("last_observation")
    if observation is not None and observation.success:
        return "order_action_router"
    return END


def _route_order_after_action_router(state: AgentState) -> str:
    action = state.get("order_action")
    if action == ActionType.QUERY_ORDER:
        return "order_summary_interpret"
    if action == ActionType.QUERY_LOGISTICS:
        return "logistics_snapshot_lookup"
    if action == ActionType.CREATE_AFTERSALES:
        return "aftersales_prepare"
    if action == ActionType.QUERY_AFTERSALES:
        return "aftersales_query_exec"
    return END


def _route_order_after_logistics_lookup(state: AgentState) -> str:
    if state.get("awaited_slots"):
        return END
    observation = state.get("last_observation")
    if state.get("order_action") == ActionType.CREATE_AFTERSALES:
        if observation is not None and observation.success:
            return "aftersales_eligibility_check"
        return END
    if observation is not None and observation.success:
        return "logistics_interpret"
    return END


def _route_order_after_aftersales_prepare(state: AgentState) -> str:
    if state.get("awaited_slots"):
        return END
    return "logistics_snapshot_lookup"


def _route_order_after_aftersales_exec(state: AgentState) -> str:
    observation = state.get("last_observation")
    if observation is None:
        return END
    if observation.source_type == ObservationSource.JUDGEMENT and observation.success:
        return "aftersales_interpret"
    if observation.source_type == ObservationSource.TOOL and (
        observation.success or observation.code in ACCEPTABLE_AFTERSALES_CODES
    ):
        return "aftersales_interpret"
    return END


def build_order_subgraph():
    graph = StateGraph(AgentState)
    graph.add_node("order_intent_understand", order_intent_understand_node)
    graph.add_node("order_identity_gate", order_identity_gate_node)
    graph.add_node("order_profile_lookup", order_profile_lookup_node)
    graph.add_node("order_action_router", order_action_router_node)
    graph.add_node("order_summary_interpret", order_summary_interpret_node)
    graph.add_node("logistics_snapshot_lookup", logistics_snapshot_lookup_node)
    graph.add_node("logistics_interpret", logistics_interpret_node)
    graph.add_node("aftersales_prepare", aftersales_prepare_node)
    graph.add_node("aftersales_eligibility_check", eligibility_check_node)
    graph.add_node("aftersales_create_exec", aftersales_create_exec_node)
    graph.add_node("aftersales_query_exec", aftersales_query_exec_node)
    graph.add_node("aftersales_interpret", aftersales_interpret_node)

    graph.add_edge(START, "order_intent_understand")
    graph.add_edge("order_intent_understand", "order_identity_gate")
    graph.add_conditional_edges(
        "order_identity_gate",
        _route_order_after_identity_gate,
        {END: END, "order_profile_lookup": "order_profile_lookup"},
    )
    graph.add_conditional_edges(
        "order_profile_lookup",
        _route_order_after_profile_lookup,
        {END: END, "order_action_router": "order_action_router"},
    )
    graph.add_conditional_edges(
        "order_action_router",
        _route_order_after_action_router,
        {
            END: END,
            "order_summary_interpret": "order_summary_interpret",
            "logistics_snapshot_lookup": "logistics_snapshot_lookup",
            "aftersales_prepare": "aftersales_prepare",
            "aftersales_query_exec": "aftersales_query_exec",
        },
    )
    graph.add_edge("order_summary_interpret", END)
    graph.add_conditional_edges(
        "logistics_snapshot_lookup",
        _route_order_after_logistics_lookup,
        {
            END: END,
            "logistics_interpret": "logistics_interpret",
            "aftersales_eligibility_check": "aftersales_eligibility_check",
        },
    )
    graph.add_edge("logistics_interpret", END)
    graph.add_conditional_edges(
        "aftersales_prepare",
        _route_order_after_aftersales_prepare,
        {END: END, "logistics_snapshot_lookup": "logistics_snapshot_lookup"},
    )
    graph.add_edge("aftersales_eligibility_check", "aftersales_create_exec")
    graph.add_conditional_edges(
        "aftersales_create_exec",
        _route_order_after_aftersales_exec,
        {END: END, "aftersales_interpret": "aftersales_interpret"},
    )
    graph.add_conditional_edges(
        "aftersales_query_exec",
        _route_order_after_aftersales_exec,
        {END: END, "aftersales_interpret": "aftersales_interpret"},
    )
    graph.add_edge("aftersales_interpret", END)
    return graph.compile()


@traceable(name="agent_v3_policy_prepare_query")
def policy_prepare_query_node(state: AgentState) -> Dict[str, Any]:
    step = get_current_step(state)
    user_input = state.get("user_input", "")
    query = state.get("pending_retrieval_query") or user_input
    tags = dict(state.get("trace_tags") or {})
    tags["current_node"] = "policy_prepare_query"
    tags.setdefault("policy_original_query", user_input)
    return {
        "active_agent": SpecialistName.POLICY,
        "pending_retrieval_query": query,
        "trace_tags": tags,
        "current_plan": mark_step_status(state, step.step_id, StepStatus.RUNNING) if step else state.get("current_plan") or [],
    }


@traceable(name="agent_v3_policy_retrieve")
def policy_retrieve_node(state: AgentState) -> Dict[str, Any]:
    step = get_current_step(state)
    query = (state.get("pending_retrieval_query") or state.get("user_input") or "").strip()
    rewritten_from = str((state.get("trace_tags") or {}).get("policy_rewritten_from") or "")
    evidence = retrieve_policy_evidence(query, rewritten_from=rewritten_from)
    success = bool(evidence)
    observation = Observation(
        step_id=step.step_id if step else "policy_lookup",
        source_type=ObservationSource.RETRIEVAL,
        source_name="policy_retrieve",
        success=success,
        code="OK" if success else "NO_HITS",
        summary=f"Retrieved {len(evidence)} policy evidence items." if success else "No policy evidence found.",
        structured_data={
            "query_used": query,
            "rewritten_from": rewritten_from,
            "hit_count": len(evidence),
        },
        evidence_refs=[item.chunk_id for item in evidence],
        retryable=not success,
    )
    patch = _with_node_tag(state, "policy_retrieve")
    patch.update(append_observation(state, observation))
    patch["retrieval_evidence"] = list(state.get("retrieval_evidence") or []) + evidence
    return patch


@traceable(name="agent_v3_policy_assess")
def policy_assess_node(state: AgentState) -> Dict[str, Any]:
    step = get_current_step(state)
    last_obs = state.get("last_observation")
    patch = _with_node_tag(state, "policy_assess")
    tags = dict(state.get("trace_tags") or {})
    tags["policy_retry_needed"] = False
    patch["trace_tags"] = tags

    if step is None or last_obs is None:
        return patch

    if last_obs.success:
        evidence = list(state.get("retrieval_evidence") or [])
        assessment = Observation(
            step_id=step.step_id,
            source_type=ObservationSource.JUDGEMENT,
            source_name="policy_assess",
            success=True,
            code="POLICY_EVIDENCE_READY",
            summary="Policy evidence is ready for verifier review.",
            structured_data={"evidence_count": len(evidence)},
            evidence_refs=[item.chunk_id for item in evidence[:2]],
        )
        patch.update(append_observation(state, assessment))
        patch["verified_facts"] = list(state.get("verified_facts") or []) + _fact_list_from_retrieval(step.step_id, evidence)
        patch["current_plan"] = mark_step_status(state, step.step_id, StepStatus.COMPLETED)
        patch["pending_retrieval_query"] = ""
        return patch

    if (state.get("rag_retry_count") or 0) < 1:
        if os.getenv("AGENT_V3_DISABLE_QUERY_REWRITE", "0") == "1":
            final_observation = Observation(
                step_id=step.step_id,
                source_type=ObservationSource.JUDGEMENT,
                source_name="policy_assess",
                success=False,
                code="NO_HITS",
                summary="No policy hit and query rewrite is disabled.",
                structured_data={},
                retryable=False,
            )
            patch.update(append_observation(state, final_observation))
            patch["current_plan"] = mark_step_status(state, step.step_id, StepStatus.FAILED)
            return patch

        original_query = state.get("pending_retrieval_query") or state.get("user_input") or ""
        rewritten = rewrite_query(original_query, state.get("user_input") or original_query)
        if rewritten != original_query:
            tags["policy_retry_needed"] = True
            tags["policy_rewritten_from"] = original_query
            patch["trace_tags"] = tags
            retry_observation = Observation(
                step_id=step.step_id,
                source_type=ObservationSource.JUDGEMENT,
                source_name="policy_assess",
                success=False,
                code="POLICY_RETRY_REWRITE",
                summary="Retrying policy retrieval with a rewritten query.",
                structured_data={"next_query": rewritten},
                retryable=True,
            )
            patch.update(append_observation(state, retry_observation))
            patch["pending_retrieval_query"] = rewritten
            return patch

    final_observation = Observation(
        step_id=step.step_id,
        source_type=ObservationSource.JUDGEMENT,
        source_name="policy_assess",
        success=False,
        code="NO_HITS",
        summary="Policy evidence is still unavailable after retry.",
        structured_data={},
        retryable=False,
    )
    patch.update(append_observation(state, final_observation))
    patch["current_plan"] = mark_step_status(state, step.step_id, StepStatus.FAILED)
    return patch


@traceable(name="agent_v3_policy_retry_or_finish")
def policy_retry_or_finish_node(state: AgentState) -> Dict[str, Any]:
    patch = _with_node_tag(state, "policy_retry_or_finish")
    tags = dict(state.get("trace_tags") or {})
    if tags.get("policy_retry_needed"):
        patch["rag_retry_count"] = int(state.get("rag_retry_count") or 0) + 1
    return patch


def _route_policy_after_retry(state: AgentState) -> str:
    tags = dict(state.get("trace_tags") or {})
    if tags.get("policy_retry_needed"):
        return "policy_retrieve"
    return END


def build_policy_subgraph():
    graph = StateGraph(AgentState)
    graph.add_node("policy_prepare_query", policy_prepare_query_node)
    graph.add_node("policy_retrieve", policy_retrieve_node)
    graph.add_node("policy_assess", policy_assess_node)
    graph.add_node("policy_retry_or_finish", policy_retry_or_finish_node)
    graph.add_edge(START, "policy_prepare_query")
    graph.add_edge("policy_prepare_query", "policy_retrieve")
    graph.add_edge("policy_retrieve", "policy_assess")
    graph.add_edge("policy_assess", "policy_retry_or_finish")
    graph.add_conditional_edges(
        "policy_retry_or_finish",
        _route_policy_after_retry,
        {END: END, "policy_retrieve": "policy_retrieve"},
    )
    return graph.compile()


ESCALATION_HUMAN_REQUEST_KEYWORDS = [
    "转人工",
    "人工客服",
    "找人工",
    "找客服主管",
    "找主管",
    "你处理不了",
]

ESCALATION_COMPLAINT_KEYWORDS = [
    "我要投诉",
    "投诉",
    "我要举报",
    "举报",
]

ESCALATION_COMPENSATION_KEYWORDS = [
    "赔偿",
    "赔付",
    "补偿",
]

ESCALATION_DELIVERY_DISPUTE_KEYWORDS = [
    "丢件",
    "丢包",
    "没收到却显示签收",
    "显示签收但没收到",
    "签收争议",
    "物流一直不更新",
    "物流不更新",
]

ESCALATION_RULE_CONFLICT_KEYWORDS = [
    "规则说可以退",
    "规则说可以",
    "政策说可以退",
    "系统又不给我退",
    "系统不给退",
    "规则和系统对不上",
]


def _safe_model_dump(value: Any) -> Dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return dict(value)
    return {}


def _detect_escalation_type(state: AgentState) -> tuple[str, str]:
    text = str(state.get("user_input") or "")
    if _contains_any(text, ESCALATION_HUMAN_REQUEST_KEYWORDS):
        return ("HUMAN_REQUEST", "用户明确要求人工接管。")
    if _contains_any(text, ESCALATION_COMPENSATION_KEYWORDS):
        return ("COMPENSATION_DISPUTE", "用户提出赔偿或赔付争议。")
    if _contains_any(text, ESCALATION_DELIVERY_DISPUTE_KEYWORDS):
        return ("DELIVERY_DISPUTE", "用户提出物流或签收争议。")
    if _contains_any(text, ESCALATION_COMPLAINT_KEYWORDS):
        return ("COMPLAINT", "用户明确提出投诉或升级诉求。")
    if _contains_any(text, ESCALATION_RULE_CONFLICT_KEYWORDS):
        return ("RULE_FACT_CONFLICT", "用户描述规则与系统事实存在冲突。")
    if state.get("escalation_type") or state.get("escalation_reason"):
        return (
            str(state.get("escalation_type") or "SYSTEM_FAILURE_ESCALATION"),
            str(state.get("escalation_reason") or "当前问题需要异常升级处理。"),
        )
    retry_counts = dict(state.get("retry_count_by_stage") or {})
    if any(int(count or 0) >= 2 for count in retry_counts.values()):
        return ("SYSTEM_FAILURE_ESCALATION", "同一处理阶段已连续失败，建议升级人工继续处理。")
    return ("SYSTEM_FAILURE_ESCALATION", "当前问题需要异常升级处理。")


def _collect_attempted_actions(state: AgentState) -> List[str]:
    actions: List[str] = []
    for observation in list(state.get("observations") or [])[-6:]:
        if not isinstance(observation, Observation):
            continue
        label = f"{observation.source_name}:{observation.code}"
        if label not in actions:
            actions.append(label)
    return actions


def _build_escalation_summary(context: EscalationContext) -> str:
    order_context = context.order_context or {}
    logistics = context.logistics_snapshot or {}
    aftersales = context.aftersales_context or {}
    parts = [
        f"用户诉求：{context.user_request or context.latest_user_request or '异常升级处理'}",
        f"升级类型：{context.escalation_type or 'SYSTEM_FAILURE_ESCALATION'}",
        f"升级原因：{context.escalation_reason or '需要人工进一步处理'}",
    ]
    if order_context:
        parts.append(
            "订单信息："
            f"订单号 {order_context.get('order_id') or '未知'}，"
            f"订单状态 {order_context.get('order_status') or '未知'}，"
            f"支付状态 {order_context.get('pay_status') or '未知'}"
        )
    if logistics:
        parts.append(
            "物流信息："
            f"状态 {logistics.get('delivery_state_name') or logistics.get('delivery_state') or '未知'}，"
            f"最新轨迹 {logistics.get('last_event') or '暂无'}"
        )
    if aftersales:
        parts.append(
            "售后信息："
            f"状态 {aftersales.get('aftersales_status') or '暂无'}，"
            f"原因 {aftersales.get('denial_reason') or aftersales.get('reason') or '暂无'}"
        )
    if context.attempted_actions:
        parts.append("已尝试动作：" + "；".join(context.attempted_actions))
    return "\n".join(parts)


def _stable_escalation_case_id(
    session_id: str,
    decision: str,
    escalation_reason: str,
    context: EscalationContext,
) -> str:
    seed = "|".join(
        [
            session_id or "session",
            decision or "FRIENDLY_FALLBACK",
            escalation_reason or "",
            context.latest_user_request or context.user_request or "",
            str(context.latest_failure.get("code") or ""),
            str(context.order_context.get("order_id") or ""),
        ]
    )
    return "ESC-" + uuid.uuid5(uuid.NAMESPACE_URL, seed).hex[:12].upper()


@traceable(name="agent_v3_escalation_intent_understand")
def escalation_intent_understand_node(state: AgentState) -> Dict[str, Any]:
    step = get_current_step(state)
    patch = _with_node_tag(state, "escalation_intent_understand")
    if step is None:
        return patch

    escalation_type, escalation_reason = _detect_escalation_type(state)
    observation = Observation(
        step_id=step.step_id,
        source_type=ObservationSource.JUDGEMENT,
        source_name="escalation_intent_understand",
        success=True,
        code="ESCALATION_INTENT_READY",
        summary=f"升级原因已识别为 {escalation_type}。",
        structured_data={
            "escalation_type": escalation_type,
            "escalation_reason": escalation_reason,
        },
    )
    patch.update(append_observation(state, observation))
    patch.update(
        {
            "escalation_type": escalation_type,
            "escalation_reason": escalation_reason,
            "active_agent": SpecialistName.ESCALATION,
            "current_plan": mark_step_status(state, step.step_id, StepStatus.RUNNING),
        }
    )
    return patch


@traceable(name="agent_v3_escalation_context_collect")
def context_collect_node(state: AgentState) -> Dict[str, Any]:
    step = get_current_step(state)
    patch = _with_node_tag(state, "escalation_context_collect")
    if step is None:
        return patch

    retrieval_evidence = [
        {
            "title": item.title,
            "summary": item.evidence_summary,
            "reference": item.chunk_id,
        }
        for item in list(state.get("retrieval_evidence") or [])[:3]
    ]
    verified_facts = [
        {
            "title": item.title,
            "detail": item.detail,
            "reference": item.reference_id,
        }
        for item in list(state.get("verified_facts") or [])[-8:]
    ]
    failure_history = [
        item.model_dump() if hasattr(item, "model_dump") else dict(item)
        for item in list(state.get("failure_history") or [])
    ]
    latest_failure = failure_history[-1] if failure_history else {}
    context = EscalationContext(
        user_request=str(state.get("user_input") or ""),
        escalation_type=str(state.get("escalation_type") or ""),
        escalation_reason=str(state.get("escalation_reason") or ""),
        order_context=_safe_model_dump(state.get("order_context")),
        logistics_snapshot=_safe_model_dump(state.get("logistics_snapshot")),
        aftersales_context=_safe_model_dump(state.get("aftersales_context")),
        policy_evidence=retrieval_evidence,
        verified_facts=verified_facts,
        failure_history=failure_history,
        attempted_actions=_collect_attempted_actions(state),
        latest_failure=latest_failure,
        latest_user_request=str(state.get("user_input") or summarize_messages(list(state.get("messages") or []), limit=4)),
        last_observation_code=str(getattr(state.get("last_observation"), "code", "") or ""),
        last_observation_summary=str(getattr(state.get("last_observation"), "summary", "") or ""),
    )
    observation = Observation(
        step_id=step.step_id,
        source_type=ObservationSource.JUDGEMENT,
        source_name="context_collect",
        success=True,
        code="ESCALATION_CONTEXT_READY",
        summary="已汇总升级处理所需的上下文信息。",
        structured_data={
            "has_order_context": bool(context.order_context),
            "has_logistics_snapshot": bool(context.logistics_snapshot),
            "has_aftersales_context": bool(context.aftersales_context),
            "failure_count": len(context.failure_history),
        },
    )
    patch.update(append_observation(state, observation))
    patch["escalation_context"] = context
    return patch


@traceable(name="agent_v3_escalation_risk_decide")
def risk_and_route_decide_node(state: AgentState) -> Dict[str, Any]:
    step = get_current_step(state)
    patch = _with_node_tag(state, "risk_and_route_decide")
    if step is None:
        return patch

    context = state.get("escalation_context")
    context_data = _safe_model_dump(context)
    escalation_type = str(state.get("escalation_type") or context_data.get("escalation_type") or "")
    user_text = str(state.get("user_input") or "")
    max_retry = max([int(v or 0) for v in dict(state.get("retry_count_by_stage") or {}).values()] or [0])

    decision = "FRIENDLY_FALLBACK"
    decision_reason = "当前场景更适合给出友好兜底说明。"
    if escalation_type == "HUMAN_REQUEST":
        decision = "HANDOFF_HUMAN"
        decision_reason = "用户明确要求人工接管。"
    elif escalation_type in {"COMPENSATION_DISPUTE", "COMPLAINT"}:
        decision = "CREATE_CASE"
        decision_reason = "当前问题属于高争议或赔付场景，优先进入升级工单。"
    elif escalation_type == "DELIVERY_DISPUTE":
        decision = "HANDOFF_HUMAN" if _contains_any(user_text, ESCALATION_HUMAN_REQUEST_KEYWORDS + ESCALATION_COMPLAINT_KEYWORDS) else "CREATE_CASE"
        decision_reason = "物流或签收争议需要人工继续核实。"
    elif escalation_type == "RULE_FACT_CONFLICT":
        decision = "CREATE_CASE"
        decision_reason = "规则证据与业务事实冲突，需人工核实。"
    elif escalation_type == "SYSTEM_FAILURE_ESCALATION":
        if max_retry >= 2:
            decision = "CREATE_CASE"
            decision_reason = "同一处理阶段已连续失败，进入异常工单处理。"
        else:
            decision = "RETRY_AUTOMATION"
            decision_reason = "当前仅出现单次异常，仍可继续自动重试。"

    observation = Observation(
        step_id=step.step_id,
        source_type=ObservationSource.JUDGEMENT,
        source_name="risk_and_route_decide",
        success=True,
        code="ESCALATION_DECISION_READY",
        summary=f"升级处理决策为 {decision}。",
        structured_data={
            "escalation_decision": decision,
            "decision_reason": decision_reason,
            "escalation_type": escalation_type,
        },
    )
    patch.update(append_observation(state, observation))
    patch["escalation_decision"] = decision
    return patch


@traceable(name="agent_v3_escalation_handoff_or_case")
def handoff_or_case_create_node(state: AgentState) -> Dict[str, Any]:
    step = get_current_step(state)
    patch = _with_node_tag(state, "handoff_or_case_create")
    if step is None:
        return patch

    context = state.get("escalation_context")
    context_model = context if isinstance(context, EscalationContext) else EscalationContext(**_safe_model_dump(context))
    decision = str(state.get("escalation_decision") or "FRIENDLY_FALLBACK")
    escalation_reason = str(state.get("escalation_reason") or context_model.escalation_reason or "")
    summary = _build_escalation_summary(context_model)
    session_id = str(state.get("session_id") or "session")
    case_id = _stable_escalation_case_id(session_id, decision, escalation_reason, context_model)

    if decision == "HANDOFF_HUMAN":
        result = TOOL_REGISTRY["handoff_to_human_tool"].callable(summary=summary, reason=escalation_reason or "异常升级处理")
        observation = _tool_observation(step.step_id, "handoff_to_human_tool", result, retryable=False)
        patch.update(append_observation(state, observation))
        if bool(result.get("success")):
            patch["handoff_case_payload"] = {
                "decision": decision,
                "summary": summary,
                "escalation_type": context_model.escalation_type,
                "escalation_reason": escalation_reason,
                "handoff_result": result.get("data") if isinstance(result.get("data"), dict) else {},
            }
            return patch

        fallback_payload = {
            "case_id": case_id,
            "decision": "CREATE_CASE",
            "summary": summary,
            "escalation_type": context_model.escalation_type,
            "escalation_reason": escalation_reason,
            "latest_failure": context_model.latest_failure,
            "latest_user_request": context_model.latest_user_request,
            "handoff_failed": True,
            "handoff_error_code": str(result.get("code") or ""),
            "handoff_error_message": str(result.get("message") or ""),
        }
        fallback_observation = Observation(
            step_id=step.step_id,
            source_type=ObservationSource.JUDGEMENT,
            source_name="handoff_or_case_create",
            success=True,
            code="ESCALATION_HANDOFF_FALLBACK_CASE",
            summary="人工转接失败，已改为生成升级工单记录。",
            structured_data=fallback_payload,
        )
        patch.update(append_observation(state, fallback_observation))
        patch["escalation_decision"] = "CREATE_CASE"
        patch["handoff_case_payload"] = fallback_payload
        return patch

    payload = {
        "case_id": case_id,
        "decision": decision,
        "summary": summary,
        "escalation_type": context_model.escalation_type,
        "escalation_reason": escalation_reason,
        "latest_failure": context_model.latest_failure,
        "latest_user_request": context_model.latest_user_request,
    }
    code = "ESCALATION_CASE_CREATED" if decision == "CREATE_CASE" else "ESCALATION_FALLBACK_READY"
    summary_text = "已生成升级工单摘要。" if decision == "CREATE_CASE" else "已整理当前异常信息并准备友好兜底。"
    observation = Observation(
        step_id=step.step_id,
        source_type=ObservationSource.JUDGEMENT,
        source_name="handoff_or_case_create",
        success=True,
        code=code,
        summary=summary_text,
        structured_data=payload,
    )
    patch.update(append_observation(state, observation))
    patch["handoff_case_payload"] = payload
    return patch


@traceable(name="agent_v3_escalation_interpret")
def escalation_interpret_node(state: AgentState) -> Dict[str, Any]:
    step = get_current_step(state)
    patch = _with_node_tag(state, "escalation_interpret")
    if step is None:
        return patch

    decision = str(state.get("escalation_decision") or "")
    visible_text = {
        "HANDOFF_HUMAN": "已转入人工处理，并整理了当前问题的关键信息。",
        "CREATE_CASE": "已提交升级处理记录，当前问题会按异常工单继续跟进。",
        "RETRY_AUTOMATION": "已记录当前异常，仍建议继续自动重试一次。",
        "FRIENDLY_FALLBACK": "当前问题暂时无法自动完成处理，建议联系人工客服继续协助。",
    }.get(decision, "当前问题已经进入异常升级处理。")
    observation = Observation(
        step_id=step.step_id,
        source_type=ObservationSource.JUDGEMENT,
        source_name="escalation_interpret",
        success=True,
        code="ESCALATION_RESULT_READY",
        summary=visible_text,
        structured_data={
            "escalation_decision": decision,
            "escalation_type": state.get("escalation_type") or "",
        },
    )
    patch.update(append_observation(state, observation))
    patch["current_plan"] = mark_step_status(state, step.step_id, StepStatus.COMPLETED)
    return patch


def build_escalation_subgraph():
    graph = StateGraph(AgentState)
    graph.add_node("escalation_intent_understand", escalation_intent_understand_node)
    graph.add_node("context_collect", context_collect_node)
    graph.add_node("risk_and_route_decide", risk_and_route_decide_node)
    graph.add_node("handoff_or_case_create", handoff_or_case_create_node)
    graph.add_node("escalation_interpret", escalation_interpret_node)

    graph.add_edge(START, "escalation_intent_understand")
    graph.add_edge("escalation_intent_understand", "context_collect")
    graph.add_edge("context_collect", "risk_and_route_decide")
    graph.add_edge("risk_and_route_decide", "handoff_or_case_create")
    graph.add_edge("handoff_or_case_create", "escalation_interpret")
    graph.add_edge("escalation_interpret", END)
    return graph.compile()
