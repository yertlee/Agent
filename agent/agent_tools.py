from typing import Any, Dict

from langchain_core.tools import StructuredTool

from .langsmith_utils import traceable
from .tools import aftersales_service, get_order_info, handoff_to_human, query_logistics_snapshot


@traceable(name="get_order_info_tool")
def get_order_info_tool(order_id: str, phone_last4: str) -> Dict[str, Any]:
    return get_order_info(order_id=order_id, phone_last4=phone_last4)


@traceable(name="create_aftersales_tool")
def create_aftersales_tool(
    order_id: str,
    phone_last4: str,
    service_type: str,
    reason: str,
) -> Dict[str, Any]:
    return aftersales_service(
        action="create",
        order_id=order_id,
        phone_last4=phone_last4,
        service_type=service_type,
        reason=reason,
    )


@traceable(name="query_aftersales_tool")
def query_aftersales_tool(order_id: str, phone_last4: str) -> Dict[str, Any]:
    return aftersales_service(
        action="query",
        order_id=order_id,
        phone_last4=phone_last4,
    )


@traceable(name="query_logistics_snapshot_tool")
def query_logistics_snapshot_tool(
    carrier_code: str,
    tracking_no: str,
    phone_last4: str = "",
) -> Dict[str, Any]:
    return query_logistics_snapshot(
        carrier_code=carrier_code,
        tracking_no=tracking_no,
        phone_last4=phone_last4 or None,
    )


@traceable(name="handoff_to_human_tool")
def handoff_to_human_tool(summary: str, reason: str) -> Dict[str, Any]:
    return handoff_to_human(summary=summary, reason=reason)


def build_langchain_tools():
    if StructuredTool is None:
        raise ImportError(
            "StructuredTool is unavailable. Install a compatible langchain-core version first."
        )

    return [
        StructuredTool.from_function(
            func=get_order_info_tool,
            name="get_order_info_tool",
            description="Query order profile facts such as product, amount, status, carrier and tracking number.",
        ),
        StructuredTool.from_function(
            func=create_aftersales_tool,
            name="create_aftersales_tool",
            description="Create an aftersales request for refund, return, or exchange.",
        ),
        StructuredTool.from_function(
            func=query_aftersales_tool,
            name="query_aftersales_tool",
            description="Query aftersales progress for an order.",
        ),
        StructuredTool.from_function(
            func=query_logistics_snapshot_tool,
            name="query_logistics_snapshot_tool",
            description="Query a unified logistics snapshot with cache-first fallback to mock or kuaidi100 providers.",
        ),
        StructuredTool.from_function(
            func=handoff_to_human_tool,
            name="handoff_to_human_tool",
            description="Create a human handoff request when the agent should stop automated handling.",
        ),
    ]


if __name__ == "__main__":
    print(get_order_info_tool("20260226003", "8820"))
    print(query_aftersales_tool("20260226004", "1027"))
    print(query_logistics_snapshot_tool("yuantong", "YT25569986666541"))
    print(handoff_to_human_tool("User asked for human support.", "manual review"))
