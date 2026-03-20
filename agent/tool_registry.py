from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from langchain_core.tools import StructuredTool

from .agent_tools import (
    create_aftersales_tool,
    get_order_info_tool,
    handoff_to_human_tool,
    query_aftersales_tool,
)
from .rag_retriever import build_citation_text, retrieve_policy_evidence


ToolCallable = Callable[..., Dict[str, Any]]


@dataclass
class ToolSpec:
    name: str
    description: str
    args_schema: Dict[str, Any]
    callable: ToolCallable
    tool_kind: str
    required_slots: List[str]
    retryable: bool
    max_retries: int
    failure_policy: str
    business_acceptable_failure_codes: List[str]

    def to_langchain_tool(self) -> StructuredTool:
        return StructuredTool.from_function(
            func=self.callable,
            name=self.name,
            description=self.description,
        )


def _schema_object(properties: Dict[str, Any], required: Optional[List[str]] = None) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


def policy_rag_search_tool(query: str, top_k: int = 3) -> Dict[str, Any]:
    hits = retrieve_policy_evidence(query, top_k=top_k)
    if not hits:
        return {
            "success": False,
            "code": "NO_HITS",
            "message": "未检索到高质量规则证据。",
            "data": {"hit_count": 0, "sources": [], "hits": []},
            "user_hint": "当前没有检索到足够相关的规则内容，请换一种更具体的问法。",
        }

    citation_text = build_citation_text(hits)
    return {
        "success": True,
        "code": "OK",
        "message": "规则检索成功",
        "data": {
            "hit_count": len(hits),
            "sources": sorted({hit.source for hit in hits}),
            "hits": [hit.model_dump() for hit in hits],
            "citation_text": citation_text,
        },
        "user_hint": "",
    }


def build_tool_registry() -> Dict[str, ToolSpec]:
    tools = [
        ToolSpec(
            name="get_order_info_tool",
            description="查询订单信息，包括订单状态、支付状态、商品名称和是否支持售后。",
            args_schema=_schema_object(
                {
                    "order_id": {"type": "string", "description": "订单号"},
                    "phone_last4": {"type": "string", "description": "手机号后四位"},
                },
                required=["order_id", "phone_last4"],
            ),
            callable=get_order_info_tool,
            tool_kind="business",
            required_slots=["order_id", "phone_last4"],
            retryable=False,
            max_retries=1,
            failure_policy="ask_user",
            business_acceptable_failure_codes=["PHONE_MISMATCH", "ORDER_NOT_FOUND"],
        ),
        ToolSpec(
            name="query_aftersales_tool",
            description="查询售后进度，用于退款/退货/换货申请后的进展查询。",
            args_schema=_schema_object(
                {
                    "order_id": {"type": "string", "description": "订单号"},
                    "phone_last4": {"type": "string", "description": "手机号后四位"},
                },
                required=["order_id", "phone_last4"],
            ),
            callable=query_aftersales_tool,
            tool_kind="business",
            required_slots=["order_id", "phone_last4"],
            retryable=False,
            max_retries=1,
            failure_policy="ask_user",
            business_acceptable_failure_codes=["PHONE_MISMATCH", "ORDER_NOT_FOUND", "AFTERSALES_NOT_FOUND"],
        ),
        ToolSpec(
            name="create_aftersales_tool",
            description="创建售后申请，用于退款、退货、换货。",
            args_schema=_schema_object(
                {
                    "order_id": {"type": "string", "description": "订单号"},
                    "phone_last4": {"type": "string", "description": "手机号后四位"},
                    "service_type": {"type": "string", "description": "售后类型：退款/退货/换货"},
                    "reason": {"type": "string", "description": "售后原因"},
                },
                required=["order_id", "phone_last4", "service_type", "reason"],
            ),
            callable=create_aftersales_tool,
            tool_kind="business",
            required_slots=["order_id", "phone_last4", "service_type", "reason"],
            retryable=False,
            max_retries=1,
            failure_policy="ask_user",
            business_acceptable_failure_codes=[
                "PHONE_MISMATCH",
                "ORDER_NOT_FOUND",
                "AFTERSALES_ALREADY_EXISTS",
                "AFTERSALES_NOT_ALLOWED",
            ],
        ),
        ToolSpec(
            name="handoff_to_human_tool",
            description="创建转人工请求，用于系统无法继续自动处理、用户要求人工或需要高风险兜底时。",
            args_schema=_schema_object(
                {
                    "summary": {"type": "string", "description": "当前对话摘要"},
                    "reason": {"type": "string", "description": "转人工原因"},
                },
                required=["summary", "reason"],
            ),
            callable=handoff_to_human_tool,
            tool_kind="handoff",
            required_slots=["summary", "reason"],
            retryable=False,
            max_retries=1,
            failure_policy="handoff",
            business_acceptable_failure_codes=["HANDOFF_CREATED"],
        ),
        ToolSpec(
            name="policy_rag_search_tool",
            description="在本地规则知识库中检索相关证据，返回结构化片段与引用摘要。",
            args_schema=_schema_object(
                {
                    "query": {"type": "string", "description": "规则检索 query"},
                    "top_k": {"type": "integer", "description": "返回条数"},
                },
                required=["query"],
            ),
            callable=policy_rag_search_tool,
            tool_kind="retrieval",
            required_slots=["query"],
            retryable=True,
            max_retries=2,
            failure_policy="explain_limit",
            business_acceptable_failure_codes=["NO_HITS"],
        ),
    ]
    return {tool.name: tool for tool in tools}


def build_langchain_tools_from_registry(registry: Optional[Dict[str, ToolSpec]] = None) -> List[StructuredTool]:
    registry = registry or build_tool_registry()
    return [tool.to_langchain_tool() for tool in registry.values()]


def describe_registry(registry: Optional[Dict[str, ToolSpec]] = None) -> str:
    registry = registry or build_tool_registry()
    lines = []
    for tool in registry.values():
        lines.append(
            f"- {tool.name}: required_slots={tool.required_slots}, retryable={tool.retryable}, "
            f"failure_policy={tool.failure_policy}"
        )
    return "\n".join(lines)

