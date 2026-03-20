from typing import Any, Dict

from .tools import aftersales_service, get_order_info, handoff_to_human
from langchain_core.tools import StructuredTool

from .langsmith_utils import traceable

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


@traceable(name="handoff_to_human_tool")
def handoff_to_human_tool(summary: str, reason: str) -> Dict[str, Any]:
    return handoff_to_human(summary=summary, reason=reason)


def build_langchain_tools():
    if StructuredTool is None:
        raise ImportError(
            "无法导入 StructuredTool，请确认已安装兼容版本的 LangChain："
            "推荐安装 `langchain-core` 或升级 `langchain`。"
        )
    tools = [
        StructuredTool.from_function(
            func=get_order_info_tool,
            name="get_order_info_tool",
            description=(
                "查询订单信息。适用于用户想查询订单状态、支付状态、商品信息、"
                "是否发货等场景。需要提供 order_id 和 phone_last4。"
            ),
        ),
        StructuredTool.from_function(
            func=create_aftersales_tool,
            name="create_aftersales_tool",
            description=(
                "创建售后申请。适用于用户想发起退款、退货、换货。"
                "需要提供 order_id、phone_last4、service_type、reason。"
            ),
        ),
        StructuredTool.from_function(
            func=query_aftersales_tool,
            name="query_aftersales_tool",
            description=(
                "查询售后进度。适用于用户询问退款进度、售后处理状态。"
                "需要提供 order_id 和 phone_last4。"
            ),
        ),
        StructuredTool.from_function(
            func=handoff_to_human_tool,
            name="handoff_to_human_tool",
            description=(
                "转接人工客服。适用于用户投诉、情绪激烈、身份无法确认、"
                "系统无法处理的场景。需要提供 summary 和 reason。"
            ),
        ),
    ]
    return tools


if __name__ == "__main__":
    print(get_order_info_tool("20260226003", "8820"))
    print(query_aftersales_tool("20260226004", "1027"))
    print(handoff_to_human_tool("用户要求人工介入。", "情绪激烈"))
    try:
        lc_tools = build_langchain_tools()
        print(len(lc_tools), "tools:", [t.name for t in lc_tools])
    except Exception as e:
        print(e)