from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from langchain_core.tools import StructuredTool

from .agent_tools import (
    create_aftersales_tool,
    get_order_info_tool,
    handoff_to_human_tool,
    query_aftersales_tool,
    query_logistics_snapshot_tool,
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
            "message": "No high-quality policy evidence found.",
            "data": {"hit_count": 0, "sources": [], "hits": []},
            "user_hint": "Try a more specific policy question.",
        }

    citation_text = build_citation_text(hits)
    return {
        "success": True,
        "code": "OK",
        "message": "Policy evidence ready.",
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
            description="Query order profile facts, including status, product, amount, carrier code and tracking number.",
            args_schema=_schema_object(
                {
                    "order_id": {"type": "string", "description": "Order ID"},
                    "phone_last4": {"type": "string", "description": "Last four digits of the receiver phone"},
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
            description="Query the latest aftersales progress for an order.",
            args_schema=_schema_object(
                {
                    "order_id": {"type": "string", "description": "Order ID"},
                    "phone_last4": {"type": "string", "description": "Last four digits of the receiver phone"},
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
            description="Create an aftersales request for refund, return, or exchange.",
            args_schema=_schema_object(
                {
                    "order_id": {"type": "string", "description": "Order ID"},
                    "phone_last4": {"type": "string", "description": "Last four digits of the receiver phone"},
                    "service_type": {"type": "string", "description": "Service type"},
                    "reason": {"type": "string", "description": "Reason for aftersales"},
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
            name="query_logistics_snapshot_tool",
            description="Query a unified logistics snapshot with cache-first fallback to mock or kuaidi100 providers.",
            args_schema=_schema_object(
                {
                    "carrier_code": {"type": "string", "description": "Carrier code"},
                    "tracking_no": {"type": "string", "description": "Tracking number"},
                    "phone_last4": {"type": "string", "description": "Optional phone tail"},
                },
                required=["carrier_code", "tracking_no"],
            ),
            callable=query_logistics_snapshot_tool,
            tool_kind="business",
            required_slots=["carrier_code", "tracking_no"],
            retryable=False,
            max_retries=1,
            failure_policy="explain_limit",
            business_acceptable_failure_codes=[],
        ),
        ToolSpec(
            name="handoff_to_human_tool",
            description="Create a human handoff request for cases that should not stay automated.",
            args_schema=_schema_object(
                {
                    "summary": {"type": "string", "description": "Conversation summary"},
                    "reason": {"type": "string", "description": "Why handoff is needed"},
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
            description="Retrieve policy evidence from the local policy knowledge base.",
            args_schema=_schema_object(
                {
                    "query": {"type": "string", "description": "Retrieval query"},
                    "top_k": {"type": "integer", "description": "How many chunks to return"},
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
