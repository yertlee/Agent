from __future__ import annotations

import os
from typing import Any, Dict, List

from langgraph.graph import END, START, StateGraph

from .state import (
    ActionType,
    AgentState,
    Observation,
    ObservationSource,
    RetrievalEvidence,
    SpecialistName,
    StepStatus,
    VerifiedFact,
    append_observation,
    get_current_step,
    mark_step_status,
    observation_to_fact_candidates,
)
from .langsmith_utils import traceable
from .rag_retriever import retrieve_policy_evidence, rewrite_query
from .tool_registry import ToolSpec, build_tool_registry


TOOL_REGISTRY = build_tool_registry()


def _with_node_tag(state: AgentState, node_name: str) -> Dict[str, Any]:
    tags = dict(state.get("trace_tags") or {})
    tags["current_node"] = node_name
    return {"trace_tags": tags}


def _missing_required_slots(state: AgentState, required_inputs: List[str]) -> List[str]:
    slot_values = state.get("slot_values") or {}
    return [slot for slot in required_inputs if not slot_values.get(slot)]


def _make_clarification_patch(state: AgentState, step_id: str, missing_slots: List[str], source_name: str) -> Dict[str, Any]:
    question = f"为了继续处理，请补充：{'、'.join(missing_slots)}。"
    observation = Observation(
        step_id=step_id,
        source_type=ObservationSource.USER_CLARIFICATION,
        source_name=source_name,
        success=False,
        code="MISSING_SLOTS",
        summary=question,
        missing_slots=missing_slots,
        retryable=True,
    )
    patch = append_observation(state, observation)
    patch.update(
        {
            "awaited_slots": missing_slots,
            "blocked_step_id": step_id,
            "pending_question": question,
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
    return Observation(
        step_id=step_id,
        source_type=ObservationSource.TOOL,
        source_name=tool_name,
        success=bool((result or {}).get("success")),
        code=str((result or {}).get("code") or ""),
        summary=str((result or {}).get("message") or ""),
        structured_data=data if isinstance(data, dict) else {},
        evidence_refs=[],
        missing_slots=[],
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


@traceable(name="agent_v3_order_slot_check")
def order_slot_check_node(state: AgentState) -> Dict[str, Any]:
    step = get_current_step(state)
    if step is None:
        return _with_node_tag(state, "order_slot_check")

    patch = _with_node_tag(state, "order_slot_check")
    missing_slots = _missing_required_slots(state, step.required_inputs)
    if missing_slots:
        patch.update(_make_clarification_patch(state, step.step_id, missing_slots, "order_slot_check"))
        return patch

    patch.update(
        {
            "awaited_slots": [],
            "blocked_step_id": "",
            "pending_question": "",
            "active_agent": SpecialistName.ORDER,
            "current_plan": mark_step_status(state, step.step_id, StepStatus.RUNNING),
        }
    )
    return patch


@traceable(name="agent_v3_order_tool_exec")
def order_tool_exec_node(state: AgentState) -> Dict[str, Any]:
    step = get_current_step(state)
    if step is None:
        return _with_node_tag(state, "order_tool_exec")

    spec: ToolSpec = TOOL_REGISTRY["get_order_info_tool"]
    slot_values = dict(state.get("slot_values") or {})
    args = {name: slot_values.get(name) for name in spec.required_slots}
    result = spec.callable(**args)
    observation = _tool_observation(step.step_id, spec.name, result, spec.retryable)

    patch = _with_node_tag(state, "order_tool_exec")
    patch.update(append_observation(state, observation))
    patch["tool_retry_counts"] = _increment_tool_retry_counts(state, spec.name)
    if observation.success:
        patch["verified_facts"] = list(state.get("verified_facts") or []) + observation_to_fact_candidates(observation)
    else:
        next_status = StepStatus.BLOCKED if observation.code in {"PHONE_MISMATCH", "ORDER_NOT_FOUND"} else StepStatus.FAILED
        patch["current_plan"] = mark_step_status(state, step.step_id, next_status)
    return patch


@traceable(name="agent_v3_order_interpret")
def order_interpret_node(state: AgentState) -> Dict[str, Any]:
    step = get_current_step(state)
    observation = state.get("last_observation")
    patch = _with_node_tag(state, "order_interpret")
    if step is None or observation is None or not observation.success:
        return patch

    data = observation.structured_data or {}
    summary = (
        f"订单 {data.get('order_id', '')} 当前状态为 {data.get('order_status', '未知')}，"
        f"支付状态为 {data.get('pay_status', '未知')}。"
    )
    judgment = Observation(
        step_id=step.step_id,
        source_type=ObservationSource.JUDGEMENT,
        source_name="order_interpret",
        success=True,
        code="ORDER_CONTEXT_READY",
        summary=summary,
        structured_data={
            "order_status": data.get("order_status"),
            "pay_status": data.get("pay_status"),
            "can_apply_aftersales": data.get("can_apply_aftersales"),
        },
        evidence_refs=[f"{observation.source_name}:order_status", f"{observation.source_name}:pay_status"],
    )
    patch.update(append_observation(state, judgment))
    patch["verified_facts"] = list(state.get("verified_facts") or []) + observation_to_fact_candidates(judgment)
    patch["current_plan"] = mark_step_status(state, step.step_id, StepStatus.COMPLETED)
    return patch


def _route_order_after_slot_check(state: AgentState) -> str:
    if state.get("awaited_slots"):
        return END
    return "order_tool_exec"


def _route_order_after_tool_exec(state: AgentState) -> str:
    observation = state.get("last_observation")
    if observation is not None and observation.success:
        return "order_interpret"
    return END


def build_order_subgraph():
    graph = StateGraph(AgentState)
    graph.add_node("order_slot_check", order_slot_check_node)
    graph.add_node("order_tool_exec", order_tool_exec_node)
    graph.add_node("order_interpret", order_interpret_node)
    graph.add_edge(START, "order_slot_check")
    graph.add_conditional_edges("order_slot_check", _route_order_after_slot_check, {END: END, "order_tool_exec": "order_tool_exec"})
    graph.add_conditional_edges("order_tool_exec", _route_order_after_tool_exec, {END: END, "order_interpret": "order_interpret"})
    graph.add_edge("order_interpret", END)
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
        summary=f"检索到 {len(evidence)} 条规则证据" if success else "未检索到高质量规则证据",
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
            summary="规则证据已准备好，可交由 verifier 判断是否收口",
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
                summary="检索未命中，且当前实验关闭了 query rewrite。",
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
                summary="首次检索未命中，准备改写 query 后重试。",
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
        summary="检索两次后仍未命中高质量规则证据。",
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


@traceable(name="agent_v3_aftersales_specialist")
def aftersales_specialist_node(state: AgentState) -> Dict[str, Any]:
    step = get_current_step(state)
    if step is None:
        return _with_node_tag(state, "aftersales_specialist")

    patch = _with_node_tag(state, "aftersales_specialist")
    missing_slots = _missing_required_slots(state, step.required_inputs)
    if missing_slots:
        patch.update(_make_clarification_patch(state, step.step_id, missing_slots, "aftersales_specialist"))
        return patch

    spec_name = "create_aftersales_tool" if step.action_type == ActionType.CREATE_AFTERSALES else "query_aftersales_tool"
    spec = TOOL_REGISTRY[spec_name]
    slot_values = dict(state.get("slot_values") or {})
    args = {name: slot_values.get(name) for name in spec.required_slots}
    result = spec.callable(**args)
    observation = _tool_observation(step.step_id, spec.name, result, spec.retryable)
    patch.update(append_observation(state, observation))
    patch["tool_retry_counts"] = _increment_tool_retry_counts(state, spec.name)

    if observation.success or observation.code in {"AFTERSALES_ALREADY_EXISTS", "AFTERSALES_NOT_ALLOWED", "AFTERSALES_NOT_FOUND"}:
        patch["verified_facts"] = list(state.get("verified_facts") or []) + observation_to_fact_candidates(observation)
        patch["current_plan"] = mark_step_status(state, step.step_id, StepStatus.COMPLETED)
        if step.action_type == ActionType.CREATE_AFTERSALES and observation.code == "AFTERSALES_NOT_ALLOWED":
            patch["handoff_reason"] = "自动售后创建未通过，建议人工进一步判断处理。"
    elif observation.code in {"DB_ERROR", "DB_NOT_FOUND", "TOOL_ERROR"}:
        patch["handoff_reason"] = "售后工具执行失败，需要人工介入。"
        patch["current_plan"] = mark_step_status(state, step.step_id, StepStatus.FAILED)
    else:
        patch["current_plan"] = mark_step_status(state, step.step_id, StepStatus.FAILED)
    return patch
