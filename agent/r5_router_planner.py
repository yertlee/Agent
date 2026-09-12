"""R5 Router/Planner boundary: separated responsibilities and fixed budgets.

Router output (intents, entities, missing slots, business goal) and Planner
output (the single authoritative R5PlanV1) are separate structures even when
produced by one model call, so each layer is measurable on its own.  The
boundary owns schema validation, registered-capability admission, bounded
schema repair and root-cause retention; the model never receives gold intent,
a preselected topology or an execution spec.

Budgets default to the R5 pre-registration (model attempts <= 2, no replan
here; tool/replan budgets live in the runtime).
"""
from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .r5_capability_args import CAPABILITY_ARGS, DERIVED, DERIVED_SOURCE_CAPABILITIES, ENTITY
from .r5_entity_candidates import ambiguous_kinds, extract_candidates, single_values
from .r5_plan_contracts import R5_ALL_CAPABILITIES, R5PlanV1


R5_INTENTS = frozenset(
    {
        "PRODUCT_QUERY",
        "ORDER_QUERY",
        "LOGISTICS_QUERY",
        "POLICY_QA",
        "AFTERSALES_STATUS",
        "AFTERSALES_CREATE",
        "AFTERSALES_CANCEL",
        "AFTERSALES_MODIFY",
        "COMPLAINT",
        "CHITCHAT",
        "UNKNOWN",
        "MULTI_INTENT",
    }
)
R5_ENTITY_KEYS = frozenset(
    {"order_id", "phone_last4", "carrier_code", "tracking_no", "sku", "product_name", "case_id", "service", "reason", "amount", "query"}
)
# Arguments the runtime always supplies (trusted session context or upstream
# derivation).  A model that asks the user for these is over-clarifying.
RUNTIME_SUPPLIED_SLOTS = frozenset({"phone_last4", "carrier_code", "tracking_no"})

# Canonical after-sales service values and their user-facing aliases.
SERVICE_ALIASES = {
    "refund": "refund", "return": "return", "exchange": "exchange",
    "退款": "refund", "退货": "return", "换货": "exchange",
}


def _over_clarified(router: "R5RouterDecisionV1") -> bool:
    if not router.needs_clarification:
        return False
    missing = set(router.missing_slots)
    return bool(missing) and missing.issubset(RUNTIME_SUPPLIED_SLOTS)


class R5RouterDecisionV1(BaseModel):
    """Router half: what the user wants and which facts are missing."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    intents: tuple[str, ...] = ()
    entities: dict[str, str] = Field(default_factory=dict)
    missing_slots: tuple[str, ...] = ()
    needs_clarification: bool = False
    clarification_question: str | None = None
    business_goal: str = Field(min_length=1)

    @model_validator(mode="after")
    def valid_decision(self) -> "R5RouterDecisionV1":
        unknown = set(self.intents) - R5_INTENTS
        if unknown:
            raise ValueError(f"unknown intents: {sorted(unknown)}")
        if len(set(self.intents)) != len(self.intents):
            raise ValueError("intents must be unique")
        bad_entities = set(map(str, self.entities)) - R5_ENTITY_KEYS
        if bad_entities:
            raise ValueError(f"unsupported entity keys: {sorted(bad_entities)}")
        bad_missing = set(self.missing_slots) - R5_ENTITY_KEYS
        if bad_missing:
            raise ValueError(f"unsupported missing slots: {sorted(bad_missing)}")
        if self.needs_clarification and not (self.clarification_question or "").strip():
            raise ValueError("clarification question is required")
        return self


class R5RouterPlannerProvider(Protocol):
    """Provider boundary; it receives only the prompt and schemas."""

    def __call__(self, prompt: str, router_schema: type[R5RouterDecisionV1], plan_schema: type[R5PlanV1]) -> Any: ...


class R5RouterPlannerAttemptV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    attempt_no: int
    router_schema_valid: bool
    plan_schema_valid: bool
    error_code: str | None = None
    error_detail: str | None = None


class R5RouterPlannerResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    router: R5RouterDecisionV1
    plan: R5PlanV1
    attempts: tuple[R5RouterPlannerAttemptV1, ...]
    first_attempt_schema_valid: bool
    eventual_schema_valid: bool
    first_root_cause: str | None
    provider_called: bool
    provider_returned: bool

    @property
    def exact_handoff(self) -> bool:
        return self.plan.derived_capabilities() == frozenset(_capabilities_for_intents(self.router.intents))


def _capabilities_for_intents(intents: Sequence[str]) -> tuple[str, ...]:
    """Canonical intent→capability mapping used only for scoring handoff."""
    mapping = {
        "PRODUCT_QUERY": ("product/read@v1",),
        "ORDER_QUERY": ("order/read@v1",),
        "LOGISTICS_QUERY": ("logistics/read@v1",),
        "POLICY_QA": ("policy/read@v1",),
        "AFTERSALES_STATUS": ("aftersales/read@v1",),
        "AFTERSALES_CREATE": ("aftersales/eligibility@v1",),
        "AFTERSALES_CANCEL": ("aftersales/eligibility@v1",),
        "AFTERSALES_MODIFY": ("aftersales/eligibility@v1",),
    }
    result: list[str] = []
    for intent in intents:
        result.extend(mapping.get(intent, ()))
    return tuple(sorted(set(result)))


def _capability_contract_prompt(allowed_capabilities: frozenset[str]) -> str:
    """Render the same argument contract used by runtime normalization."""
    source_labels = {
        "entity": "用户实体或节点 args 中的同值显式参数",
        "entity_or_context": "用户实体优先，否则可信上下文",
        "request_text": "原始用户消息",
        "context": "可信上下文",
        "derived": "上游 order/read@v1 结果，必须有对应 edge；无上游则需用户显式提供",
    }
    lines = ["能力参数合同（只约束参数来源，不会自动增加能力节点）："]
    for capability in sorted(allowed_capabilities):
        contract = CAPABILITY_ARGS.get(capability)
        if not contract:
            continue
        fields = ", ".join(f"{argument}<-{source_labels.get(source, source)}" for argument, source in contract.items())
        lines.append(f"  - {capability}: {fields}")
    source_caps = "、".join(sorted(DERIVED_SOURCE_CAPABILITIES))
    lines.append(f"  - derived 参数（carrier_code、tracking_no）由上游 order/read@v1 结果提供：用户没直接给运单号时，请计划一个 order/read 节点并用 edges+bindings 从它派生，不要因此请求澄清；只有连 order_id 都没有时才澄清。")
    lines.append("  - phone_last4 由可信会话上下文提供，不需要用户提供，也不要因此请求澄清。")
    lines.append(f"  - 只有用户必须提供且消息里确实缺失的实体（order_id、sku、service，或仅凭运单号查询时的 tracking_no）才请求澄清；derived 允许的上游为：{source_caps}，bindings 必须与 plan.edges 一致。")
    lines.append("  - 物流查询不强制先查订单：用户明确提供 carrier_code+tracking_no 可独立读取；只有存在 order/read 上游且有 edge 时才从其结果绑定。")
    return "\n".join(lines) + "\n"


def build_router_planner_prompt(user_message: str, *, allowed_capabilities: frozenset[str]) -> str:
    capabilities = "、".join(sorted(allowed_capabilities))
    intents = "、".join(sorted(R5_INTENTS))
    entity_keys = "、".join(sorted(R5_ENTITY_KEYS))
    return (
        "你是一个电商客服系统的意图与计划组件。只输出一个 JSON 对象，不要输出解释或代码块。\n"
        "JSON 顶层包含 router 和 plan 两个键。\n"
        "router 字段：\n"
        f"  - intents：数组，取值必须严格来自（区分大小写）：{intents}\n"
        f"  - entities：对象，键必须来自：{entity_keys}；值为字符串\n"
        "  - missing_slots：数组，缺少的实体键\n"
        "  - needs_clarification：布尔值\n"
        "  - clarification_question：字符串或 null\n"
        "  - business_goal：字符串\n"
        "plan 字段：\n"
        "  - schema_version：固定为 \"r5.plan.v1\"\n"
        "  - nodes：数组，每项形如 {\"node_id\": 小写字母数字下划线, \"capability_ref\": 注册能力, \"args\": {}, \"bindings\": {}, \"failure_strategy\": \"FAIL_RUN\"}\n"
        f"  - edges：数组，每项形如 {{\"upstream_node_id\": ..., \"downstream_node_id\": ...}}\n"
        "  - needs_clarification：布尔值；clarification_reason：字符串或 null\n"
        "  - business_goal：字符串\n"
        "规则：\n"
        "  1) node_id 必须唯一；capability_ref 只能使用下面注册能力；\n"
        "  2) 用户消息里已经明确的业务参数（sku、order_id、phone_last4、service 等）直接写进该节点的 args，例如 {\"sku\": \"SKU-1002\"}；\n"
        "  3) bindings 只在参数来自上游节点结果或可信上下文时使用，键是参数名，值是对象，"
        "形如 {\"kind\": \"result\", \"source_node_id\": \"上游node_id\", \"path\": \"order_id\"} 或 "
        "{\"kind\": \"context\", \"context_key\": \"phone_last4\"}；绝不要把字面参数或 $.xxx 路径字符串放进 bindings；\n"
        "  4) 只有当**用户必须提供**的实体（如 order_id、sku、service，或仅凭运单号查物流时的 tracking_no）在消息里确实缺失时才设 needs_clarification=true 且 plan.nodes 留空；"
        "由运行时提供的参数缺失不构成澄清理由：phone_last4 来自可信会话上下文，carrier_code/tracking_no 可由上游 order/read@v1 结果派生，二者都不需要用户提供；\n"
        "  5) 用户提出售后**动作**请求（申请/创建、取消、修改）时，计划中必须包含 aftersales/write@v1 节点（它代表等待用户确认的动作）；"
        "申请/创建还必须在其上游放 aftersales/eligibility@v1 节点并用 edges 连接；取消/修改可用 aftersales/read@v1 或 eligibility 定位 case；"
        "只做资格判断、不含 write 节点不算完成动作；\n"
        "  6) 不要输出固定模板名称或能力汇总字段，能力集合会从节点自动派生；\n"
        "  7) 用户要求投诉、或明确要求人工处理时，使用 human/handoff@v1 节点升级人工；不要自行承诺赔付。\n"
        "正确示例（查商品）：{\"router\": {\"intents\": [\"PRODUCT_QUERY\"], \"entities\": {\"sku\": \"SKU-1002\"}, "
        "\"missing_slots\": [], \"needs_clarification\": false, \"clarification_question\": null, \"business_goal\": \"查询商品价格库存\"}, "
        "\"plan\": {\"schema_version\": \"r5.plan.v1\", \"nodes\": [{\"node_id\": \"product_read\", \"capability_ref\": \"product/read@v1\", "
        "\"args\": {\"sku\": \"SKU-1002\"}, \"bindings\": {}, \"failure_strategy\": \"FAIL_RUN\"}], \"edges\": [], "
        "\"needs_clarification\": false, \"clarification_reason\": null, \"business_goal\": \"查询商品价格库存\"}}\n"
        + _capability_contract_prompt(allowed_capabilities)
        + f"注册能力：{capabilities}\n"
        + f"用户消息：{user_message}"
    )


def apply_entity_policy(result: "R5RouterPlannerResult", user_message: str) -> "R5RouterPlannerResult":
    """Deterministic candidate policy on top of the model decision.

    - multiple distinct candidates for a kind the plan needs, and the model
      did not pick one: force a clarification instead of guessing;
    - exactly one candidate for a needed kind the model omitted: fill it in
      (structured identifiers are parsed deterministically, role stays with
      the model when it did choose).
    """
    candidates = extract_candidates(user_message)
    ambiguous = ambiguous_kinds(candidates)
    singles = single_values(candidates)
    needed: set[str] = set()
    ancestors = result.plan.ancestors()
    node_by_id = {node.node_id: node for node in result.plan.nodes}
    for node in result.plan.nodes:
        contract = CAPABILITY_ARGS.get(node.capability_ref, {})
        for arg, source in contract.items():
            if source == ENTITY:
                needed.add(arg)
            elif source == DERIVED:
                # A direct logistics query still needs explicitly stated
                # carrier/tracking entities.  An order-backed query already
                # has a trusted source and must not be made ambiguous by
                # unrelated candidate text in the same message.
                has_order_source = any(
                    node_by_id.get(ancestor_id) is not None
                    and node_by_id[ancestor_id].capability_ref in DERIVED_SOURCE_CAPABILITIES
                    for ancestor_id in ancestors.get(node.node_id, set())
                )
                if not has_order_source and arg not in node.bindings:
                    needed.add(arg)
    entities = dict(result.router.entities)

    # Coerce service aliases (e.g. "退款" -> "refund") so a valid intent is not
    # rejected later as an invalid entity value; drop an unrecognized value so
    # a deterministic candidate can still fill it.
    coerced_service = False
    if "service" in entities:
        raw_service = str(entities["service"]).strip()
        normalized_service = SERVICE_ALIASES.get(raw_service)
        if normalized_service is None:
            entities.pop("service", None)
            coerced_service = True
        elif normalized_service != raw_service:
            entities["service"] = normalized_service
            coerced_service = True

    if any(kind in needed and kind not in entities for kind in ambiguous):
        plan = R5PlanV1(nodes=(), edges=(), needs_clarification=True, clarification_reason="ambiguous_entity_candidates", business_goal=result.plan.business_goal)
        router = result.router.model_copy(update={"needs_clarification": True, "clarification_question": result.router.clarification_question or "检测到多个候选，请确认您要处理的目标。"})
        return result.model_copy(update={"router": router, "plan": plan})

    filled = coerced_service
    for kind, value in singles.items():
        if kind in needed and kind not in entities:
            entities[kind] = value
            filled = True

    # Plan-time legality for required entity arguments: if the model left a
    # required entity unfilled and no candidate exists, ask instead of
    # emitting a plan that cannot execute.
    missing: set[str] = set()
    for node in result.plan.nodes:
        contract = CAPABILITY_ARGS.get(node.capability_ref, {})
        for arg, source in contract.items():
            if source == ENTITY and arg not in entities and node.args.get(arg) in (None, ""):
                missing.add(arg)
            elif source == DERIVED:
                has_order_source = any(
                    node_by_id.get(ancestor_id) is not None
                    and node_by_id[ancestor_id].capability_ref in DERIVED_SOURCE_CAPABILITIES
                    for ancestor_id in ancestors.get(node.node_id, set())
                )
                if not has_order_source and arg not in entities and node.args.get(arg) in (None, "") and arg not in node.bindings:
                    missing.add(arg)
    if missing and not result.plan.needs_clarification:
        plan = R5PlanV1(nodes=(), edges=(), needs_clarification=True, clarification_reason=f"missing_required_entity:{','.join(sorted(missing))}", business_goal=result.plan.business_goal)
        router = result.router.model_copy(update={"needs_clarification": True, "clarification_question": result.router.clarification_question or "请补充必要信息（如订单号）。", "entities": entities})
        return result.model_copy(update={"router": router, "plan": plan})

    if filled:
        return result.model_copy(update={"router": result.router.model_copy(update={"entities": entities})})
    return result


class R5RouterPlannerBoundary:
    """Deterministic boundary around a model provider with bounded repair."""

    def __init__(self, *, allowed_capabilities: frozenset[str] | None = None, model_attempt_budget: int = 2):
        if int(model_attempt_budget) not in {1, 2}:
            raise ValueError("model_attempt_budget must be 1 or 2 (R5 pre-registration)")
        self.allowed_capabilities = frozenset(allowed_capabilities or R5_ALL_CAPABILITIES)
        if not self.allowed_capabilities.issubset(R5_ALL_CAPABILITIES):
            raise ValueError("allowed_capabilities must be a subset of registered capabilities")
        self.model_attempt_budget = int(model_attempt_budget)

    def _admit_capabilities(self, plan: R5PlanV1) -> None:
        unadmitted = plan.derived_capabilities() - self.allowed_capabilities
        if unadmitted:
            raise ValueError(f"plan references unadmitted capabilities: {sorted(unadmitted)}")

    @staticmethod
    def _coerce_router(raw: Any) -> R5RouterDecisionV1:
        if isinstance(raw, R5RouterDecisionV1):
            return raw
        return R5RouterDecisionV1.model_validate(raw)

    @staticmethod
    def _coerce_plan(raw: Any) -> R5PlanV1:
        if isinstance(raw, R5PlanV1):
            return raw
        return R5PlanV1.model_validate(raw)

    def run(self, *, user_message: str, provider: R5RouterPlannerProvider) -> R5RouterPlannerResult:
        prompt = build_router_planner_prompt(user_message, allowed_capabilities=self.allowed_capabilities)
        attempts: list[R5RouterPlannerAttemptV1] = []
        first_root_cause: str | None = None
        last_error: str | None = None
        router: R5RouterDecisionV1 | None = None
        plan: R5PlanV1 | None = None
        provider_called = False
        provider_returned = False
        for attempt_no in range(1, self.model_attempt_budget + 1):
            provider_called = True
            retry_prompt = prompt if last_error is None else f"{prompt}\n\n上一次输出无效：{last_error}。请修正后重新输出。"
            try:
                raw = provider(retry_prompt, R5RouterDecisionV1, R5PlanV1)
                provider_returned = True
            except Exception as exc:  # provider failure is a first-class recorded outcome
                detail = f"{type(exc).__name__}: {str(exc)[:300]}"
                last_error = f"provider_error:{detail}"
                attempts.append(R5RouterPlannerAttemptV1(attempt_no=attempt_no, router_schema_valid=False, plan_schema_valid=False, error_code="PROVIDER_ERROR", error_detail=detail))
                if first_root_cause is None:
                    first_root_cause = "PROVIDER_ERROR"
                continue
            router_valid = plan_valid = False
            try:
                router = self._coerce_router(raw.get("router") if isinstance(raw, Mapping) else getattr(raw, "router", None))
                router_valid = True
            except Exception as exc:
                last_error = f"router_schema:{exc}"
                if first_root_cause is None:
                    first_root_cause = "ROUTER_SCHEMA_INVALID"
            if router_valid:
                try:
                    plan = self._coerce_plan(raw.get("plan") if isinstance(raw, Mapping) else getattr(raw, "plan", None))
                    plan_valid = True
                except Exception as exc:
                    last_error = f"plan_schema:{exc}"
                    if first_root_cause is None:
                        first_root_cause = "PLAN_SCHEMA_INVALID"
            if plan_valid and plan is not None:
                try:
                    self._admit_capabilities(plan)
                except ValueError as exc:
                    plan_valid = False
                    last_error = f"capability_admission:{exc}"
                    if first_root_cause is None:
                        first_root_cause = "CAPABILITY_NOT_ADMITTED"
            if router_valid and plan_valid and router is not None and _over_clarified(router) and attempt_no < self.model_attempt_budget:
                # The model asked the user for runtime-supplied arguments; give
                # it one bounded repair attempt instead of accepting a wrong
                # clarification.
                plan_valid = False
                last_error = (
                    f"over_clarification: missing_slots={sorted(router.missing_slots)} 由运行时提供（会话上下文/上游派生），不应请求澄清；请直接生成计划。"
                )
                if first_root_cause is None:
                    first_root_cause = "OVER_CLARIFICATION"
            attempts.append(R5RouterPlannerAttemptV1(attempt_no=attempt_no, router_schema_valid=router_valid, plan_schema_valid=plan_valid, error_code=None if (router_valid and plan_valid) else (first_root_cause or "SCHEMA_INVALID")))
            if router_valid and plan_valid and router is not None and plan is not None:
                result = R5RouterPlannerResult(
                    router=router, plan=plan, attempts=tuple(attempts), first_attempt_schema_valid=attempts[0].router_schema_valid and attempts[0].plan_schema_valid,
                    eventual_schema_valid=True, first_root_cause=first_root_cause, provider_called=provider_called, provider_returned=provider_returned,
                )
                return apply_entity_policy(result, user_message)
        # Exhausted budget: return the last router if valid, else a safe clarification plan.
        safe_router = router or R5RouterDecisionV1(intents=("UNKNOWN",), needs_clarification=True, clarification_question="请补充更多信息以便我理解您的需求。", business_goal="clarify")
        safe_plan = plan or R5PlanV1(nodes=(), edges=(), needs_clarification=True, clarification_reason=first_root_cause or "schema_invalid", business_goal=safe_router.business_goal)
        return R5RouterPlannerResult(
            router=safe_router, plan=safe_plan, attempts=tuple(attempts), first_attempt_schema_valid=False,
            eventual_schema_valid=False, first_root_cause=first_root_cause or "SCHEMA_INVALID", provider_called=provider_called, provider_returned=provider_returned,
        )


class DeterministicRouterPlannerAdapter:
    """Labelled deterministic baseline (keyword rules), NOT the model.

    Used by tests and by the ``keyword_router`` baseline arm so the target
    system is compared against a real weak method on the same inputs.
    """

    name = "deterministic_keyword_v1"

    def __call__(self, prompt: str, router_schema, plan_schema) -> Mapping[str, Any]:
        text = prompt.rsplit("用户消息：", 1)[-1]
        intents: list[str] = []
        entities: dict[str, str] = {}
        missing: list[str] = []
        if any(token in text for token in ("物流", "快递", "到哪", "运单", "tracking")):
            intents.append("LOGISTICS_QUERY")
        if any(token in text for token in ("售后", "退货", "退款", "换货", "申请")):
            if "申请" in text or "退货" in text or "退款" in text:
                intents.append("AFTERSALES_CREATE")
            else:
                intents.append("AFTERSALES_STATUS")
        if any(token in text for token in ("订单", "买")):
            intents.append("ORDER_QUERY")
        if any(token in text for token in ("政策", "规则", "规定", "能不能退")):
            intents.append("POLICY_QA")
        if any(token in text for token in ("商品", "库存", "多少钱", "SKU", "sku")):
            intents.append("PRODUCT_QUERY")
        import re

        order_match = re.search(r"\b(20\d{9,})\b", text)
        if order_match:
            entities["order_id"] = order_match.group(1)
        phone_match = re.search(r"(?<!\d)(\d{4})(?!\d)\s*(?:$|，|。)", text)
        if phone_match:
            entities["phone_last4"] = phone_match.group(1)
        for kind, value in single_values(extract_candidates(text)).items():
            if kind in {"carrier_code", "tracking_no"} and kind not in entities:
                entities[kind] = value
        if not intents:
            intents = ["UNKNOWN"]
        for intent in intents:
            if intent == "ORDER_QUERY" and "order_id" not in entities:
                missing.append("order_id")
            if intent == "LOGISTICS_QUERY":
                if "tracking_no" not in entities:
                    missing.append("tracking_no")
                if "carrier_code" not in entities:
                    missing.append("carrier_code")
        needs_clarification = bool(missing)
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        capability_map = {
            "PRODUCT_QUERY": "product/read@v1",
            "ORDER_QUERY": "order/read@v1",
            "LOGISTICS_QUERY": "logistics/read@v1",
            "POLICY_QA": "policy/read@v1",
            "AFTERSALES_STATUS": "aftersales/read@v1",
            "AFTERSALES_CREATE": "aftersales/eligibility@v1",
        }
        seen: set[str] = set()
        for intent in intents:
            capability = capability_map.get(intent)
            if capability and capability not in seen and not needs_clarification:
                seen.add(capability)
                namespace, rest = capability.split("/", 1)
                node_id = f"{namespace}_{rest.split('@', 1)[0]}"
                nodes.append({"node_id": node_id, "capability_ref": capability, "args": {}, "bindings": {}, "failure_strategy": "FAIL_RUN"})
        router = {
            "intents": tuple(intents),
            "entities": entities,
            "missing_slots": tuple(sorted(set(missing))),
            "needs_clarification": needs_clarification,
            "clarification_question": (
                "请提供承运商编码和运单号以便查询。"
                if needs_clarification and "LOGISTICS_QUERY" in intents
                else ("请提供订单号（和手机号后四位）以便查询。" if needs_clarification else None)
            ),
            "business_goal": "、".join(intents),
        }
        plan = {
            "schema_version": "r5.plan.v1",
            "nodes": tuple(nodes),
            "edges": tuple(edges),
            "needs_clarification": needs_clarification,
            "clarification_reason": "missing_trusted_entities" if needs_clarification else None,
            "business_goal": router["business_goal"],
        }
        return {"router": router, "plan": plan}


__all__ = [
    "DeterministicRouterPlannerAdapter",
    "R5_ENTITY_KEYS",
    "R5_INTENTS",
    "R5RouterDecisionV1",
    "R5RouterPlannerAttemptV1",
    "R5RouterPlannerBoundary",
    "R5RouterPlannerProvider",
    "R5RouterPlannerResult",
    "apply_entity_policy",
    "build_router_planner_prompt",
]
