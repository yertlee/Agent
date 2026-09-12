"""R1.5 structured routing boundary.

The router proposes business intent and entity candidates only.  Capability
references are derived by the frozen policy below, never accepted as model
authority.  ``DeterministicRouterAdapter`` is an explicit test adapter; it has
no keyword fallback and returns only decisions supplied by the caller.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Callable, Mapping, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class BusinessIntent(str, Enum):
    ORDER_QUERY = "ORDER_QUERY"
    LOGISTICS_QUERY = "LOGISTICS_QUERY"
    ORDER_AND_LOGISTICS = "ORDER_AND_LOGISTICS"
    AFTERSALES_CREATE = "AFTERSALES_CREATE"
    AFTERSALES_STATUS = "AFTERSALES_STATUS"
    POLICY_QA = "POLICY_QA"
    PRODUCT_QA = "PRODUCT_QA"
    COMPLAINT = "COMPLAINT"
    CHITCHAT = "CHITCHAT"
    UNKNOWN = "UNKNOWN"
    MULTI_INTENT = "MULTI_INTENT"


ENTITY_NAMES = frozenset(
    {
        "order_id",
        "phone_last4",
        "carrier_code",
        "tracking_no",
        "product_sku",
        "aftersales_ticket_id",
    }
)


class EntityCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    value: str = Field(min_length=1)
    confidence: float = Field(default=1.0, ge=0, le=1)

    @field_validator("name")
    @classmethod
    def valid_name(cls, value: str) -> str:
        if value not in ENTITY_NAMES:
            raise ValueError("unsupported entity name")
        return value


def _coerce_entities(value: Any) -> tuple[EntityCandidate, ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        value = [{"name": key, "value": item} for key, item in value.items() if item is not None]
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("entities must be a mapping or sequence")
    return tuple(item if isinstance(item, EntityCandidate) else EntityCandidate.model_validate(item) for item in value)


class RoutingDecisionV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(default="r1.5.routing.v1", pattern=r"^r1\.5\.routing\.v1$")
    primary_intent: BusinessIntent
    secondary_intents: tuple[BusinessIntent, ...] = ()
    entities: tuple[EntityCandidate, ...] = ()
    required_capabilities: tuple[str, ...] = ()
    needs_clarification: bool = False
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="before")
    @classmethod
    def normalize_entities(cls, values: Any) -> Any:
        if isinstance(values, Mapping) and "entities" in values:
            result = dict(values)
            result["entities"] = _coerce_entities(result["entities"])
            return result
        return values

    @property
    def entity_map(self) -> dict[str, str]:
        return {item.name: item.value for item in self.entities}


class CustomerGoalV1(BaseModel):
    """The restricted R2 planner input derived from a routing decision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(default="r1.5.customer-goal.v1", pattern=r"^r1\.5\.customer-goal\.v1$")
    goal_type: BusinessIntent
    entities: tuple[EntityCandidate, ...] = ()
    required_capabilities: tuple[str, ...] = ()
    needs_clarification: bool = False

    @model_validator(mode="before")
    @classmethod
    def normalize_entities(cls, values: Any) -> Any:
        if isinstance(values, Mapping) and "entities" in values:
            result = dict(values)
            result["entities"] = _coerce_entities(result["entities"])
            return result
        return values

    @property
    def entity_map(self) -> dict[str, str]:
        return {item.name: item.value for item in self.entities}


INTENT_CAPABILITY_POLICY: dict[BusinessIntent, tuple[str, ...]] = {
    BusinessIntent.ORDER_QUERY: ("order/read@v1",),
    BusinessIntent.LOGISTICS_QUERY: ("logistics/read@v1",),
    BusinessIntent.ORDER_AND_LOGISTICS: ("order/read@v1", "logistics/read@v1"),
    BusinessIntent.AFTERSALES_CREATE: ("aftersales/write@v1",),
    BusinessIntent.AFTERSALES_STATUS: ("aftersales/read@v1",),
    BusinessIntent.POLICY_QA: ("policy/read@v1",),
    BusinessIntent.PRODUCT_QA: ("product/read@v1",),
    BusinessIntent.COMPLAINT: ("human/handoff@v1",),
    BusinessIntent.CHITCHAT: (),
    BusinessIntent.UNKNOWN: (),
    BusinessIntent.MULTI_INTENT: (),
}


class RouterContractError(ValueError):
    pass


class RouterProvider(Protocol):
    def __call__(self, prompt: str, output_schema: type[Any]) -> Any: ...


ROUTER_PROMPT_VERSION = "r1.5.router.prompt.v1"


def build_router_prompt(message: str) -> str:
    """Build the production routing prompt without dataset/gold knowledge."""
    return """你是电商客服业务路由器。只根据用户当前表达识别业务目标和实体，不推断数据库状态，不执行工具，不把过期、冲突、超时、重试或安全停止当成意图。
可选 primary_intent：ORDER_QUERY 订单信息；LOGISTICS_QUERY 运输轨迹；ORDER_AND_LOGISTICS 同时查询订单和物流；AFTERSALES_CREATE 创建退款/退换申请；AFTERSALES_STATUS 查询已有售后；POLICY_QA 规则咨询；PRODUCT_QA 商品信息；COMPLAINT 投诉/人工介入；CHITCHAT 闲聊；UNKNOWN 非电商或无法判断；MULTI_INTENT 同时存在不属于 ORDER_AND_LOGISTICS 专项组合的多个目标。
secondary_intents 仅在 MULTI_INTENT 时列出实际子意图。entities 只提取文本中明确出现的值，并为每个明确实体各输出一项。
entity.name 只能逐字使用以下英文标识：order_id、phone_last4、carrier_code、tracking_no、product_sku、aftersales_ticket_id；禁止翻译字段名，例如订单号必须写成 order_id，手机尾号必须写成 phone_last4。
required_capabilities 必须输出空数组，由策略层计算。缺少完成当前目标必需的信息时 needs_clarification=true。
用户原文：
""" + str(message)


def _capabilities(intent: BusinessIntent) -> tuple[str, ...]:
    return INTENT_CAPABILITY_POLICY.get(intent, ())


def normalize_decision(decision: RoutingDecisionV1) -> RoutingDecisionV1:
    """Apply the policy and reject model-supplied capability escalation."""
    expected = _capabilities(decision.primary_intent)
    if decision.required_capabilities and tuple(decision.required_capabilities) != expected:
        raise RouterContractError("model capability proposal does not match frozen policy")
    return decision.model_copy(update={"required_capabilities": expected})


def to_customer_goal(decision: RoutingDecisionV1) -> CustomerGoalV1:
    decision = normalize_decision(decision)
    if decision.primary_intent not in {
        BusinessIntent.ORDER_QUERY,
        BusinessIntent.LOGISTICS_QUERY,
        BusinessIntent.ORDER_AND_LOGISTICS,
    }:
        raise RouterContractError("R2 accepts only order/logistics customer goals")
    entities = decision.entity_map
    required = {
        BusinessIntent.ORDER_QUERY: ("order_id", "phone_last4"),
        BusinessIntent.LOGISTICS_QUERY: ("carrier_code", "tracking_no", "phone_last4"),
        BusinessIntent.ORDER_AND_LOGISTICS: ("order_id", "phone_last4"),
    }[decision.primary_intent]
    missing = any(not entities.get(name) for name in required)
    return CustomerGoalV1(
        goal_type=decision.primary_intent,
        entities=decision.entities,
        required_capabilities=decision.required_capabilities,
        needs_clarification=decision.needs_clarification or missing,
    )


def normalize_customer_goal(goal: CustomerGoalV1) -> CustomerGoalV1:
    """Validate a goal at the R2 boundary without granting new capabilities."""
    if goal.goal_type not in {
        BusinessIntent.ORDER_QUERY,
        BusinessIntent.LOGISTICS_QUERY,
        BusinessIntent.ORDER_AND_LOGISTICS,
    }:
        raise RouterContractError("R2 accepts only order/logistics customer goals")
    expected = _capabilities(goal.goal_type)
    if tuple(goal.required_capabilities) != expected:
        raise RouterContractError("customer goal capabilities do not match frozen policy")
    return goal


class StructuredRouter:
    """Provider-backed router.  It has no keyword or rule-based fallback."""

    def __init__(self, provider: RouterProvider):
        self.provider = provider

    def route(self, message: str) -> RoutingDecisionV1:
        raw = self.provider(build_router_prompt(message), RoutingDecisionV1)
        if hasattr(raw, "output"):
            raw = raw.output
        try:
            decision = raw if isinstance(raw, RoutingDecisionV1) else RoutingDecisionV1.model_validate(raw)
            return normalize_decision(decision)
        except Exception as exc:
            if isinstance(exc, RouterContractError):
                raise
            raise RouterContractError("structured routing output is invalid") from exc


ExternalStructuredRouter = StructuredRouter


class DeterministicRouterAdapter:
    """Explicit deterministic test adapter keyed by caller-provided decisions."""

    test_only = True

    def __init__(self, decision: RoutingDecisionV1 | Mapping[str, Any] | Callable[[str], RoutingDecisionV1 | Mapping[str, Any]]):
        self._decision = decision

    def __call__(self, prompt: str, output_schema: type[Any]) -> Any:
        if output_schema is not RoutingDecisionV1:
            raise RouterContractError("deterministic router only supports RoutingDecisionV1")
        value = self._decision(prompt) if callable(self._decision) else self._decision
        return value if isinstance(value, RoutingDecisionV1) else RoutingDecisionV1.model_validate(value)


__all__ = [
    "BusinessIntent",
    "ROUTER_PROMPT_VERSION",
    "CustomerGoalV1",
    "DeterministicRouterAdapter",
    "EntityCandidate",
    "INTENT_CAPABILITY_POLICY",
    "RouterContractError",
    "RoutingDecisionV1",
    "StructuredRouter",
    "build_router_prompt",
    "ExternalStructuredRouter",
    "normalize_decision",
    "normalize_customer_goal",
    "to_customer_goal",
]
