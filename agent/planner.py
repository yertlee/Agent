from __future__ import annotations

import re
from typing import Any, Dict, List

from .langsmith_utils import traceable
from .state import (
    ActionType,
    AgentState,
    IntentType,
    PlanBundle,
    PlanMode,
    PlanStep,
    SpecialistName,
    VerificationResult,
    extract_slots_from_text,
    get_current_step,
    merge_slot_values,
)


POLICY_KEYWORDS = [
    "规则",
    "政策",
    "七天无理由",
    "7天无理由",
    "运费规则",
    "邮费",
    "运费",
    "平台规则",
    "平台政策",
    "退款规则",
    "退货条件",
]

EXPLICIT_POLICY_KEYWORDS = [
    "规则",
    "政策",
    "七天无理由",
    "7天无理由",
    "运费规则",
    "平台规则",
    "平台政策",
    "退款规则",
    "退货条件",
]

ORDER_KEYWORDS = [
    "订单",
    "订单号",
    "订单状态",
    "订单信息",
    "订单详情",
    "物流",
    "快递",
    "发货",
    "签收",
    "派件",
    "在途",
    "退货",
    "退款",
    "换货",
    "售后",
    "售后单",
    "售后进度",
]

ORDER_STATUS_KEYWORDS = [
    "订单状态",
    "支付",
    "什么时候发货",
    "何时发货",
    "查订单",
    "订单信息",
    "订单详情",
]

EXPLICIT_ORDER_STATUS_KEYWORDS = [
    "查订单",
    "订单状态",
    "订单信息",
    "订单详情",
    "什么时候发货",
    "何时发货",
    "支付状态",
]

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

AFTERSALES_KEYWORDS = [
    "退款",
    "退货",
    "换货",
    "售后",
    "退钱",
    "申请售后",
    "帮我退",
]

AFTERSALES_PROGRESS_KEYWORDS = [
    "进度",
    "状态",
    "售后单",
    "处理到哪",
    "查询售后",
    "售后进展",
]

EXPLICIT_AFTERSALES_REQUEST_KEYWORDS = [
    "我要退货",
    "我要退款",
    "我要换货",
    "申请售后",
    "申请退货",
    "申请退款",
    "帮我退",
    "帮我退款",
    "售后进度",
    "售后状态",
    "处理到哪",
]

ESCALATION_STRONG_KEYWORDS = [
    "转人工",
    "人工客服",
    "找人工",
    "找客服主管",
    "找主管",
    "你处理不了",
    "我要投诉",
    "我要举报",
    "我要赔偿",
    "我要赔付",
]

ESCALATION_DISPUTE_KEYWORDS = [
    "丢件",
    "丢包",
    "没收到却显示签收",
    "显示签收但没收到",
    "签收争议",
    "赔付争议",
    "赔偿争议",
    "系统结果错误",
    "不接受当前结果",
    "不接受这个结果",
]

ORDER_EXECUTION_PATTERNS = [
    re.compile(pattern)
    for pattern in [
        r"我(?:想要|想|要).{0,3}(退货|退款|换货|申请售后|售后)",
        r"(?:帮我|请帮我).{0,3}(退货|退款|换货|申请售后|查订单|查物流|查快递)",
        r"申请.{0,3}(退货|退款|换货|售后)",
        r"查(?:一下)?(?:这个|该|这笔|这单)?(?:订单|物流|快递)",
        r"(?:退货|退款|换货|售后).{0,3}(进度|状态|处理到哪|售后单)",
    ]
]

ORDER_CONTEXT_HINTS = [
    "订单号",
    "订单状态",
    "订单信息",
    "订单详情",
    "这个订单",
    "该订单",
    "这笔订单",
    "这单",
    "物流",
    "快递",
    "发货",
    "签收",
    "派件",
]

EXPLICIT_ORDER_KEYWORDS = [
    "订单",
    "订单号",
    "订单状态",
    "订单信息",
    "物流",
    "快递",
    "发货",
    "签收",
]


def _contains_any(text: str, words: List[str]) -> bool:
    return any(word in text for word in words)


def _strong_escalation_hit(text: str) -> bool:
    return _contains_any(text, ESCALATION_STRONG_KEYWORDS) or _contains_any(text, ESCALATION_DISPUTE_KEYWORDS)


def _policy_signal_hit(text: str) -> bool:
    return _contains_any(text, POLICY_KEYWORDS) or _contains_any(text, EXPLICIT_POLICY_KEYWORDS)


def _order_signal_hit(text: str, slot_values: Dict[str, Any]) -> bool:
    if slot_values.get("order_id") or slot_values.get("tracking_no"):
        return True
    if _contains_any(text, ORDER_CONTEXT_HINTS):
        return True
    if _contains_any(text, ORDER_KEYWORDS) or _contains_any(text, EXPLICIT_ORDER_KEYWORDS):
        return True
    return any(pattern.search(text) for pattern in ORDER_EXECUTION_PATTERNS)


def _concrete_order_context_hit(text: str, slot_values: Dict[str, Any]) -> bool:
    if slot_values.get("order_id") or slot_values.get("tracking_no"):
        return True
    if _contains_any(text, ORDER_CONTEXT_HINTS) or _contains_any(text, EXPLICIT_ORDER_KEYWORDS):
        return True
    return any(pattern.search(text) for pattern in ORDER_EXECUTION_PATTERNS)


def _state_needs_escalation(state: AgentState) -> bool:
    if state.get("escalation_reason") or state.get("escalation_type"):
        return True
    if state.get("handoff_reason"):
        return True
    return False


def _guess_intent_rule_based(state: AgentState, user_input: str, slot_values: Dict[str, Any]) -> IntentType:
    text = (user_input or "").strip()
    if not text:
        return IntentType.UNKNOWN

    if _strong_escalation_hit(text) or _state_needs_escalation(state):
        return IntentType.ESCALATION

    order_hit = _order_signal_hit(text, slot_values)
    policy_hit = _policy_signal_hit(text)
    concrete_order_hit = _concrete_order_context_hit(text, slot_values)

    if policy_hit and not concrete_order_hit:
        return IntentType.POLICY

    if order_hit and policy_hit:
        return IntentType.MIXED

    if policy_hit:
        return IntentType.POLICY

    if order_hit:
        return IntentType.ORDER

    return IntentType.UNKNOWN


@traceable(name="agent_v3_classify_intent")
def classify_intent(state: AgentState) -> IntentType:
    user_input = state.get("user_input", "")
    slot_values = merge_slot_values(
        state,
        extract_slots_from_text(user_input, awaited_slots=state.get("awaited_slots") or []),
    )
    return _guess_intent_rule_based(state, user_input, slot_values)


def _detect_order_domain_action(user_input: str, slot_values: Dict[str, Any], state: AgentState) -> ActionType:
    text = user_input or ""
    policy_hit = _policy_signal_hit(text)
    order_query_hit = _contains_any(text, ORDER_STATUS_KEYWORDS) or _contains_any(text, EXPLICIT_ORDER_STATUS_KEYWORDS)
    explicit_aftersales_request = _contains_any(text, EXPLICIT_AFTERSALES_REQUEST_KEYWORDS)

    if _contains_any(text, LOGISTICS_KEYWORDS):
        return ActionType.QUERY_LOGISTICS

    if policy_hit and order_query_hit and not explicit_aftersales_request:
        return ActionType.QUERY_ORDER

    if _contains_any(text, AFTERSALES_KEYWORDS) or slot_values.get("service_type"):
        if policy_hit and not explicit_aftersales_request:
            return ActionType.QUERY_ORDER if order_query_hit or slot_values.get("order_id") else ActionType.CREATE_AFTERSALES
        if _contains_any(text, AFTERSALES_PROGRESS_KEYWORDS):
            return ActionType.QUERY_AFTERSALES
        return ActionType.CREATE_AFTERSALES

    if order_query_hit:
        return ActionType.QUERY_ORDER

    previous_action = state.get("order_action")
    if previous_action in {
        ActionType.QUERY_ORDER,
        ActionType.QUERY_LOGISTICS,
        ActionType.CREATE_AFTERSALES,
        ActionType.QUERY_AFTERSALES,
    }:
        return previous_action

    return ActionType.QUERY_ORDER


def _required_slots_for_action(action_type: ActionType) -> List[str]:
    if action_type in {
        ActionType.QUERY_ORDER,
        ActionType.QUERY_LOGISTICS,
        ActionType.CREATE_AFTERSALES,
        ActionType.QUERY_AFTERSALES,
    }:
        return ["order_id", "phone_last4"]
    return []


def _make_step(
    step_id: str,
    owner_agent: SpecialistName,
    action_type: ActionType,
    goal: str,
    *,
    required_inputs: List[str] | None = None,
    success_condition: str = "",
    fallback_action: str = "",
    internal_note: str = "",
) -> PlanStep:
    return PlanStep(
        step_id=step_id,
        owner_agent=owner_agent,
        action_type=action_type,
        goal=goal,
        required_inputs=required_inputs or _required_slots_for_action(action_type),
        success_condition=success_condition or "observation 成功，或 verifier 允许进入下一步",
        fallback_action=fallback_action or "ask_user",
        internal_note=internal_note,
    )


def _missing_slots(slots: Dict[str, Any], required: List[str]) -> List[str]:
    return [slot for slot in required if not slots.get(slot)]


def _order_step_for_input(state: AgentState, slot_values: Dict[str, Any]) -> PlanStep:
    action = _detect_order_domain_action(state.get("user_input", ""), slot_values, state)
    goals = {
        ActionType.QUERY_ORDER: "确认订单身份并解释订单状态",
        ActionType.QUERY_LOGISTICS: "确认订单身份并查询订单物流快照",
        ActionType.CREATE_AFTERSALES: "确认订单身份并创建售后申请",
        ActionType.QUERY_AFTERSALES: "确认订单身份并查询售后进度",
    }
    return _make_step(
        "order_domain_handle",
        SpecialistName.ORDER,
        action,
        goals.get(action, "处理订单域诉求"),
        fallback_action="ask_user",
    )


def _escalation_step(reason: str = "", escalation_type: str = "") -> PlanStep:
    return _make_step(
        "escalation_handle",
        SpecialistName.ESCALATION,
        ActionType.HANDLE_ESCALATION,
        "整理异常上下文并决定升级处理方式",
        required_inputs=[],
        fallback_action="finalize",
        internal_note=f"{escalation_type}|{reason}".strip("|"),
    )


def _build_rule_based_plan(state: AgentState, intent_type: IntentType) -> PlanBundle:
    user_input = state.get("user_input", "")
    slot_values = merge_slot_values(
        state,
        extract_slots_from_text(user_input, awaited_slots=state.get("awaited_slots") or []),
    )

    if intent_type == IntentType.ORDER:
        steps = [_order_step_for_input(state, slot_values)]
    elif intent_type == IntentType.MIXED:
        steps = [
            _order_step_for_input(state, slot_values),
            _make_step(
                "policy_lookup",
                SpecialistName.POLICY,
                ActionType.QUERY_POLICY,
                "妫€绱㈠苟鎬荤粨鐩稿叧骞冲彴瑙勫垯",
                required_inputs=[],
                fallback_action="explain_limit",
            ),
        ]
    elif intent_type == IntentType.POLICY:
        steps = [
            _make_step(
                "policy_lookup",
                SpecialistName.POLICY,
                ActionType.QUERY_POLICY,
                "检索并总结相关平台规则",
                required_inputs=[],
                fallback_action="explain_limit",
            )
        ]
    elif intent_type == IntentType.ESCALATION:
        steps = [_escalation_step(str(state.get("escalation_reason") or ""), str(state.get("escalation_type") or ""))]
    else:
        steps = [
            _make_step(
                "general_fallback",
                SpecialistName.GENERAL,
                ActionType.GENERAL_RESPONSE,
                "给出友好的澄清或能力说明",
                required_inputs=[],
                fallback_action="finalize",
            )
        ]

    plan_mode = PlanMode.FULL if intent_type == IntentType.MIXED else PlanMode.MINIMAL
    required_slots = [] if intent_type == IntentType.MIXED else (_missing_slots(slot_values, steps[0].required_inputs) if steps else [])
    active_specialist = SpecialistName.GENERAL if intent_type == IntentType.MIXED else (steps[0].owner_agent if steps else SpecialistName.GENERAL)
    fallback_strategy = steps[0].fallback_action if steps else "ask_user"
    return PlanBundle(
        plan_mode=plan_mode,
        steps=steps,
        active_specialist=active_specialist,
        required_slots=required_slots,
        fallback_strategy=fallback_strategy,
    )


def _validate_plan(bundle: PlanBundle, intent_type: IntentType) -> bool:
    allowed_slots = {"order_id", "phone_last4", "service_type", "reason", "summary", "query", "tracking_no"}
    if not bundle.steps:
        return False
    if bundle.plan_mode == PlanMode.MINIMAL and len(bundle.steps) > 1:
        return False
    if len(bundle.steps) > 4:
        return False

    seen_ids = set()
    for step in bundle.steps:
        if step.step_id in seen_ids:
            return False
        seen_ids.add(step.step_id)
        if any(slot not in allowed_slots for slot in step.required_inputs):
            return False
        expected_required = set(_required_slots_for_action(step.action_type))
        if expected_required and not expected_required.issubset(set(step.required_inputs)):
            return False

    if intent_type == IntentType.ORDER and bundle.steps[0].owner_agent != SpecialistName.ORDER:
        return False
    if intent_type == IntentType.MIXED:
        if len(bundle.steps) != 2:
            return False
        owners = [step.owner_agent for step in bundle.steps]
        if owners != [SpecialistName.ORDER, SpecialistName.POLICY]:
            return False
    if intent_type == IntentType.POLICY and bundle.steps[0].owner_agent != SpecialistName.POLICY:
        return False
    if intent_type == IntentType.ESCALATION and bundle.steps[0].owner_agent != SpecialistName.ESCALATION:
        return False
    return True


@traceable(name="agent_v3_build_initial_plan")
def build_initial_plan(state: AgentState) -> PlanBundle:
    intent_type = state.get("intent_type") or classify_intent(state)
    bundle = _build_rule_based_plan(state, intent_type)
    if not _validate_plan(bundle, intent_type):
        return _build_rule_based_plan(state, IntentType.UNKNOWN)
    return bundle


def _fallback_replan(state: AgentState, verification_result: VerificationResult) -> PlanBundle:
    plan = list(state.get("current_plan") or [])
    current_step = get_current_step(state)
    current_owner = current_step.owner_agent if current_step else SpecialistName.GENERAL

    if verification_result.should_escalate:
        return PlanBundle(
            plan_mode=PlanMode.MINIMAL,
            steps=[_escalation_step(verification_result.escalation_reason, verification_result.escalation_type)],
            active_specialist=SpecialistName.ESCALATION,
            required_slots=[],
            fallback_strategy="finalize",
        )

    if verification_result.must_ask_user:
        return PlanBundle(
            plan_mode=state.get("plan_mode") or PlanMode.MINIMAL,
            steps=plan,
            active_specialist=current_owner,
            required_slots=list(state.get("awaited_slots") or []),
            fallback_strategy="ask_user",
        )

    if verification_result.retry_same_step:
        return PlanBundle(
            plan_mode=state.get("plan_mode") or PlanMode.MINIMAL,
            steps=plan,
            active_specialist=current_owner,
            required_slots=current_step.required_inputs if current_step else [],
            fallback_strategy=current_step.fallback_action if current_step else "finalize",
        )

    if verification_result.should_replan:
        return PlanBundle(
            plan_mode=PlanMode.FULL,
            steps=plan,
            active_specialist=current_owner,
            required_slots=current_step.required_inputs if current_step else [],
            fallback_strategy=current_step.fallback_action if current_step else "finalize",
        )

    return PlanBundle(
        plan_mode=state.get("plan_mode") or PlanMode.MINIMAL,
        steps=plan,
        active_specialist=current_owner,
        required_slots=[],
        fallback_strategy="finalize",
    )


@traceable(name="agent_v3_replan_after_verification")
def replan_after_verification(state: AgentState, verification_result: VerificationResult) -> PlanBundle:
    return _fallback_replan(state, verification_result)
