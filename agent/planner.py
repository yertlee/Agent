from __future__ import annotations

import os
from typing import Any, Dict, List

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from .llm import build_chat_model
from .prompts import (
    CLASSIFIER_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    REPLAN_SYSTEM_PROMPT,
    build_classifier_user_prompt,
    build_planner_user_prompt,
    build_replanner_user_prompt,
)
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
    summarize_messages,
)
from .langsmith_utils import traceable


SMALLTALK_KEYWORDS = [
    "你好",
    "您好",
    "在吗",
    "谢谢",
    "多谢",
    "拜拜",
    "再见",
    "你能做什么",
    "你会什么",
]

ORDER_KEYWORDS = [
    "订单",
    "发货",
    "物流",
    "支付",
    "未发货",
    "订单状态",
]

AFTERSALES_KEYWORDS = [
    "退款",
    "退货",
    "换货",
    "售后",
    "退钱",
]

POLICY_KEYWORDS = [
    "规则",
    "政策",
    "七天无理由",
    "运费",
    "邮费",
    "时效",
    "平台",
    "支持吗",
]


class ClassificationOutput(BaseModel):
    intent_type: IntentType
    confidence: float = 0.0
    rationale: str = ""


def _contains_any(text: str, words: List[str]) -> bool:
    return any(word in text for word in words)


def _planner_system_prompt() -> str:
    variant = os.getenv("AGENT_V3_PLANNER_PROMPT_VARIANT", "default").strip().lower()
    if variant == "compact":
        return PLANNER_SYSTEM_PROMPT + "\n补充要求：优先生成最短可执行计划，避免冗余步骤。"
    return PLANNER_SYSTEM_PROMPT


def _replan_system_prompt() -> str:
    variant = os.getenv("AGENT_V3_PLANNER_PROMPT_VARIANT", "default").strip().lower()
    if variant == "compact":
        return REPLAN_SYSTEM_PROMPT + "\n补充要求：如无必要，不新增步骤。"
    return REPLAN_SYSTEM_PROMPT


def _guess_intent_rule_based(user_input: str) -> IntentType:
    text = (user_input or "").strip()
    if not text:
        return IntentType.UNKNOWN

    smalltalk_hit = _contains_any(text, SMALLTALK_KEYWORDS) and not (
        _contains_any(text, ORDER_KEYWORDS)
        or _contains_any(text, AFTERSALES_KEYWORDS)
        or _contains_any(text, POLICY_KEYWORDS)
    )
    if smalltalk_hit and len(text) <= 40:
        return IntentType.SMALLTALK

    order_hit = _contains_any(text, ORDER_KEYWORDS)
    aftersales_hit = _contains_any(text, AFTERSALES_KEYWORDS)
    policy_hit = _contains_any(text, POLICY_KEYWORDS)
    action_request_hit = any(
        word in text
        for word in ["帮我申请", "发起", "帮我退", "帮我换", "帮我退款", "帮我退货", "查询售后", "售后进度"]
    )

    if policy_hit and aftersales_hit and not action_request_hit and not order_hit:
        return IntentType.POLICY

    business_hits = int(order_hit) + int(aftersales_hit) + int(policy_hit)
    if business_hits >= 2:
        return IntentType.MIXED
    if aftersales_hit:
        return IntentType.AFTERSALES
    if order_hit:
        return IntentType.ORDER
    if policy_hit:
        return IntentType.POLICY
    if len(text) <= 24 and not any(mark in text for mark in ["?", "？"]):
        return IntentType.SMALLTALK
    return IntentType.UNKNOWN


@traceable(name="agent_v3_classify_intent")
def classify_intent(state: AgentState) -> IntentType:
    user_input = state.get("user_input", "")
    rule_guess = _guess_intent_rule_based(user_input)
    if rule_guess in {
        IntentType.SMALLTALK,
        IntentType.ORDER,
        IntentType.AFTERSALES,
        IntentType.POLICY,
        IntentType.MIXED,
    }:
        return rule_guess

    llm = build_chat_model(temperature=0.0, max_tokens=200, tags=["classifier"])
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CLASSIFIER_SYSTEM_PROMPT),
            ("human", "{user_prompt}"),
        ]
    )
    chain = prompt | llm.with_structured_output(ClassificationOutput)
    recent_messages = summarize_messages(state.get("messages") or [])
    try:
        result = chain.invoke({"user_prompt": build_classifier_user_prompt(user_input, recent_messages)})
        return result.intent_type
    except Exception:
        return IntentType.UNKNOWN


def _detect_aftersales_action(user_input: str, slot_values: Dict[str, Any]) -> ActionType:
    text = user_input or ""
    if any(word in text for word in ["进度", "状态", "售后单", "售后进展", "查询售后"]):
        return ActionType.QUERY_AFTERSALES
    if slot_values.get("service_type") or any(word in text for word in ["退款", "退货", "换货", "申请售后"]):
        return ActionType.CREATE_AFTERSALES
    return ActionType.QUERY_AFTERSALES


def _required_slots_for_action(action_type: ActionType) -> List[str]:
    if action_type == ActionType.QUERY_ORDER:
        return ["order_id", "phone_last4"]
    if action_type == ActionType.QUERY_AFTERSALES:
        return ["order_id", "phone_last4"]
    if action_type == ActionType.CREATE_AFTERSALES:
        return ["order_id", "phone_last4", "service_type", "reason"]
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


def _build_rule_based_plan(state: AgentState, intent_type: IntentType) -> PlanBundle:
    user_input = state.get("user_input", "")
    slot_values = merge_slot_values(state, extract_slots_from_text(user_input))
    steps: List[PlanStep] = []

    if intent_type == IntentType.SMALLTALK:
        steps = [
            _make_step(
                "general_smalltalk",
                SpecialistName.GENERAL,
                ActionType.GENERAL_RESPONSE,
                "礼貌回应用户，并说明可支持的客服能力",
                required_inputs=[],
                fallback_action="finalize",
            )
        ]
    elif intent_type == IntentType.ORDER:
        steps = [
            _make_step(
                "order_lookup",
                SpecialistName.ORDER,
                ActionType.QUERY_ORDER,
                "查询并解释订单状态",
                fallback_action="ask_user",
            )
        ]
    elif intent_type == IntentType.POLICY:
        steps = [
            _make_step(
                "policy_lookup",
                SpecialistName.POLICY,
                ActionType.QUERY_POLICY,
                "检索并解释相关平台规则",
                required_inputs=[],
                fallback_action="explain_limit",
            )
        ]
    elif intent_type == IntentType.AFTERSALES:
        aftersales_action = _detect_aftersales_action(user_input, slot_values)
        steps = [
            _make_step(
                "aftersales_handle",
                SpecialistName.AFTERSALES,
                aftersales_action,
                "处理售后查询或售后申请",
                fallback_action="ask_user" if aftersales_action == ActionType.CREATE_AFTERSALES else "handoff",
            )
        ]
    else:
        order_hit = _contains_any(user_input, ORDER_KEYWORDS)
        policy_hit = _contains_any(user_input, POLICY_KEYWORDS)
        aftersales_hit = _contains_any(user_input, AFTERSALES_KEYWORDS)
        if order_hit:
            steps.append(
                _make_step(
                    "order_lookup",
                    SpecialistName.ORDER,
                    ActionType.QUERY_ORDER,
                    "先查询订单与身份校验信息",
                    fallback_action="ask_user",
                )
            )
        if policy_hit:
            steps.append(
                _make_step(
                    "policy_lookup",
                    SpecialistName.POLICY,
                    ActionType.QUERY_POLICY,
                    "检索并解释相关规则",
                    required_inputs=[],
                    fallback_action="explain_limit",
                )
            )
        if aftersales_hit:
            aftersales_action = _detect_aftersales_action(user_input, slot_values)
            steps.append(
                _make_step(
                    "aftersales_handle",
                    SpecialistName.AFTERSALES,
                    aftersales_action,
                    "处理售后任务",
                    fallback_action="ask_user" if aftersales_action == ActionType.CREATE_AFTERSALES else "handoff",
                )
            )
        if not steps:
            steps = [
                _make_step(
                    "policy_lookup",
                    SpecialistName.POLICY,
                    ActionType.QUERY_POLICY,
                    "尝试以规则答疑方式理解用户诉求",
                    required_inputs=[],
                    fallback_action="explain_limit",
                )
            ]

    plan_mode = PlanMode.MINIMAL if len(steps) == 1 else PlanMode.FULL
    required_slots = _missing_slots(slot_values, steps[0].required_inputs) if steps else []
    active_specialist = steps[0].owner_agent if steps else SpecialistName.GENERAL
    fallback_strategy = steps[0].fallback_action if steps else "ask_user"
    return PlanBundle(
        plan_mode=plan_mode,
        steps=steps,
        active_specialist=active_specialist,
        required_slots=required_slots,
        fallback_strategy=fallback_strategy,
    )


def _validate_plan(bundle: PlanBundle, intent_type: IntentType) -> bool:
    allowed_slots = {"order_id", "phone_last4", "service_type", "reason", "summary", "query"}
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

    if intent_type == IntentType.SMALLTALK:
        step = bundle.steps[0]
        if step.owner_agent != SpecialistName.GENERAL or step.action_type != ActionType.GENERAL_RESPONSE:
            return False
    if intent_type in {IntentType.ORDER, IntentType.AFTERSALES, IntentType.POLICY, IntentType.MIXED}:
        if bundle.steps[0].owner_agent == SpecialistName.GENERAL:
            return False
    return True


@traceable(name="agent_v3_build_initial_plan")
def build_initial_plan(state: AgentState) -> PlanBundle:
    intent_type = state.get("intent_type") or classify_intent(state)
    fallback_bundle = _build_rule_based_plan(state, intent_type)

    if intent_type == IntentType.SMALLTALK:
        return fallback_bundle

    llm = build_chat_model(temperature=0.0, max_tokens=700, tags=["planner"])
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _planner_system_prompt()),
            ("human", "{user_prompt}"),
        ]
    )
    chain = prompt | llm.with_structured_output(PlanBundle)
    slot_values = merge_slot_values(state, extract_slots_from_text(state.get("user_input", "")))
    try:
        bundle = chain.invoke(
            {
                "user_prompt": build_planner_user_prompt(
                    user_input=state.get("user_input", ""),
                    intent_type=intent_type,
                    known_slots=slot_values,
                    current_plan=[],
                    last_observation=state.get("last_observation"),
                    retrieved_count=len(state.get("retrieval_evidence") or []),
                )
            }
        )
        if _validate_plan(bundle, intent_type):
            return bundle
    except Exception:
        pass
    return fallback_bundle


def _fallback_replan(state: AgentState, verification_result: VerificationResult) -> PlanBundle:
    plan = list(state.get("current_plan") or [])
    current_step = get_current_step(state)

    if verification_result.should_handoff:
        handoff_step = _make_step(
            "handoff_step",
            SpecialistName.AFTERSALES,
            ActionType.HANDOFF,
            "将当前问题转交人工客服处理",
            required_inputs=[],
            fallback_action="handoff",
        )
        return PlanBundle(
            plan_mode=PlanMode.MINIMAL,
            steps=[handoff_step],
            active_specialist=SpecialistName.AFTERSALES,
            required_slots=[],
            fallback_strategy="handoff",
        )

    if verification_result.must_ask_user:
        return PlanBundle(
            plan_mode=state.get("plan_mode") or PlanMode.MINIMAL,
            steps=plan,
            active_specialist=current_step.owner_agent if current_step else SpecialistName.GENERAL,
            required_slots=list(state.get("awaited_slots") or []),
            fallback_strategy="ask_user",
        )

    if verification_result.should_replan:
        return PlanBundle(
            plan_mode=PlanMode.FULL,
            steps=plan,
            active_specialist=current_step.owner_agent if current_step else SpecialistName.GENERAL,
            required_slots=current_step.required_inputs if current_step else [],
            fallback_strategy=current_step.fallback_action if current_step else "finalize",
        )

    return PlanBundle(
        plan_mode=state.get("plan_mode") or PlanMode.MINIMAL,
        steps=plan,
        active_specialist=current_step.owner_agent if current_step else SpecialistName.GENERAL,
        required_slots=[],
        fallback_strategy="finalize",
    )


@traceable(name="agent_v3_replan_after_verification")
def replan_after_verification(state: AgentState, verification_result: VerificationResult) -> PlanBundle:
    fallback_bundle = _fallback_replan(state, verification_result)
    if not verification_result.should_replan:
        return fallback_bundle

    llm = build_chat_model(temperature=0.0, max_tokens=700, tags=["replanner"])
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _replan_system_prompt()),
            ("human", "{user_prompt}"),
        ]
    )
    chain = prompt | llm.with_structured_output(PlanBundle)
    slot_values = merge_slot_values(state, extract_slots_from_text(state.get("user_input", "")))
    try:
        bundle = chain.invoke(
            {
                "user_prompt": build_replanner_user_prompt(
                    user_input=state.get("user_input", ""),
                    current_plan=list(state.get("current_plan") or []),
                    current_step_index=int(state.get("current_step_index") or 0),
                    last_observation=state.get("last_observation"),
                    verification_result=verification_result,
                    known_slots=slot_values,
                )
            }
        )
        if _validate_plan(bundle, state.get("intent_type") or IntentType.UNKNOWN):
            return bundle
    except Exception:
        pass
    return fallback_bundle
