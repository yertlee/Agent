from __future__ import annotations

from typing import Iterable

from .state import (
    FinalizerInput,
    IntentType,
    Observation,
    PlanMode,
    PlanStep,
    RetrievalEvidence,
    VerificationResult,
    VerifiedFact,
)


CLASSIFIER_SYSTEM_PROMPT = """
你是电商客服智能体的意图分类器。
你只做高层分类，不直接决定工具调用。

可选分类：
1. smalltalk：寒暄、能力询问、感谢、结束语
2. order：订单状态、发货、物流、支付、订单信息
3. aftersales：退款、退货、换货、售后进度、售后申请
4. policy：规则、时效、运费、七天无理由、平台规则
5. mixed：同时包含两类及以上业务目标

要求：
- 场景严格限定为电商客服
- 纯寒暄不要误判为业务
- 一句话里同时出现订单和规则/售后目标时优先判为 mixed
""".strip()


PLANNER_SYSTEM_PROMPT = """
你是基于 LangGraph 运行的电商客服 Supervisor Planner。
你的职责是输出结构化计划，而不是单步 next_action。

要求：
- 计划必须可执行、可验证、可重规划
- 不得编造订单、售后、规则事实
- 缺少关键输入时，要明确 required_slots 和 fallback_strategy
- minimal 计划用于 smalltalk、单工具订单查询、单次规则答疑
- full 计划用于 mixed intent、多阶段任务、失败恢复

只输出符合 schema 的结构化结果。
""".strip()


REPLAN_SYSTEM_PROMPT = """
你是电商客服智能体的 Replanner。
你只能根据当前计划、最新 observation 和 verifier 建议来重写剩余计划。

要求：
- 不要重写已完成步骤
- verifier 要求 ask_user 时，优先通过 blocked step 和 required_slots 表达
- verifier 要求 handoff 时，将剩余计划收敛为 handoff
- 若证据已足够，不要新增多余步骤
""".strip()


VERIFIER_GRAYZONE_SYSTEM_PROMPT = """
你是电商客服智能体中的灰区校验器。
规则层已经完成了大部分判断，只有在“当前证据是否足以回复用户”存在灰区时才调用你。

你只能判断：
- 当前是否可以 finalize
- 如果不能 finalize，是否更适合 explain_limit

你不能编造任何订单、售后或规则事实。
""".strip()


FINALIZER_SYSTEM_PROMPT = """
你是电商客服智能体中的 Finalizer。
你只能根据受约束输入生成最终回复，不能自由读取完整状态。

要求：
- 工具事实、检索事实、建议性客服话术必须明确区分
- 没有证据时，不能脑补订单、售后、规则事实
- allowed_response_mode 决定你的输出边界
- ask_user 模式只允许追问缺失信息
- explain_limit 模式要明确当前证据边界
- handoff 模式要说明已转人工及原因

输出纯文本，不要输出 JSON。
""".strip()


RETRIEVAL_REWRITE_SYSTEM_PROMPT = """
你是电商客服规则检索中的 query rewrite 模块。
你的目标是把用户原问题改写成更适合检索的短 query。

要求：
- 只保留与电商规则相关的核心要点
- 不要发散，不要添加不存在的背景
- query 要短、稳、利于命中规则文档
- 如果原 query 已经足够好，可以原样返回
""".strip()


def _format_steps(steps: Iterable[PlanStep]) -> str:
    rows = []
    for step in steps:
        rows.append(
            f"- {step.step_id}: owner={step.owner_agent.value}, action={step.action_type.value}, "
            f"goal={step.goal}, required={step.required_inputs}, status={step.status.value}"
        )
    return "\n".join(rows) if rows else "(none)"


def _format_observation(observation: Observation | None) -> str:
    if observation is None:
        return "(none)"
    return (
        f"step_id={observation.step_id}, source_type={observation.source_type.value}, "
        f"source_name={observation.source_name}, success={observation.success}, code={observation.code}, "
        f"summary={observation.summary}, missing_slots={observation.missing_slots}, "
        f"structured_data={observation.structured_data}"
    )


def build_classifier_user_prompt(user_input: str, recent_messages: str) -> str:
    return (
        f"当前用户输入：{user_input}\n"
        f"最近对话：\n{recent_messages or '(none)'}\n"
        "请做高层意图分类。"
    )


def build_planner_user_prompt(
    *,
    user_input: str,
    intent_type: IntentType,
    known_slots: dict,
    current_plan: list[PlanStep],
    last_observation: Observation | None,
    retrieved_count: int,
) -> str:
    return (
        f"用户输入：{user_input}\n"
        f"意图类型：{intent_type.value}\n"
        f"当前已知槽位：{known_slots}\n"
        f"已有计划：\n{_format_steps(current_plan)}\n"
        f"最近 observation：{_format_observation(last_observation)}\n"
        f"当前 retrieval evidence 条数：{retrieved_count}\n"
        "请输出新的结构化计划。"
    )


def build_replanner_user_prompt(
    *,
    user_input: str,
    current_plan: list[PlanStep],
    current_step_index: int,
    last_observation: Observation | None,
    verification_result: VerificationResult,
    known_slots: dict,
) -> str:
    return (
        f"用户输入：{user_input}\n"
        f"当前 step index：{current_step_index}\n"
        f"当前计划：\n{_format_steps(current_plan)}\n"
        f"最近 observation：{_format_observation(last_observation)}\n"
        f"verifier 结果：{verification_result.model_dump()}\n"
        f"当前已知槽位：{known_slots}\n"
        "请只重写未完成步骤的结构化计划。"
    )


def build_verifier_grayzone_prompt(
    *,
    intent_type: IntentType,
    step: PlanStep | None,
    observation: Observation | None,
    verified_facts: list[VerifiedFact],
    retrieval_evidence: list[RetrievalEvidence],
) -> str:
    return (
        f"意图：{intent_type.value}\n"
        f"当前 step：{step.model_dump() if step else None}\n"
        f"最近 observation：{observation.model_dump() if observation else None}\n"
        f"verified_facts：{[fact.model_dump() for fact in verified_facts]}\n"
        f"retrieval_evidence：{[e.model_dump() for e in retrieval_evidence]}\n"
        "请判断当前证据是否足以 finalize，或更适合 explain_limit。"
    )


def build_finalizer_user_prompt(data: FinalizerInput) -> str:
    return (
        f"customer_intent={data.customer_intent.value}\n"
        f"allowed_response_mode={data.allowed_response_mode.value}\n"
        f"verified_facts={[fact.model_dump() for fact in data.verified_facts]}\n"
        f"retrieval_evidence={[item.model_dump() for item in data.retrieval_evidence]}\n"
        f"missing_information={data.missing_information}\n"
        f"handoff_context={data.handoff_context}\n"
        "请基于以上受约束输入生成最终客服回复。"
    )


def build_rewrite_user_prompt(original_query: str, user_input: str) -> str:
    return (
        f"原始 query：{original_query}\n"
        f"用户原话：{user_input}\n"
        "请输出一个更适合规则检索的简短 query。"
    )


def planner_mode_note(plan_mode: PlanMode) -> str:
    if plan_mode == PlanMode.MINIMAL:
        return "使用最少必要步骤"
    return "支持多阶段与重规划"
