from __future__ import annotations


DEMO_CASES_V3 = [
    {
        "name": "clarify_order_then_resume",
        "turns": [
            "帮我查一下订单",
            "订单号 20260226003，后四位 8820",
        ],
        "expected": {
            "first_route": "order",
            "expected_clarify_slots": ["order_id", "phone_last4"],
            "allow_handoff": False,
            "expect_policy_hits": False,
        },
    },
    {
        "name": "tool_failure_then_ask_user",
        "turns": [
            "帮我查订单 20260226003，后四位 0000",
        ],
        "expected": {
            "first_route": "order",
            "expected_response_mode": "ask_user",
            "allow_handoff": False,
            "expect_policy_hits": False,
        },
    },
    {
        "name": "policy_no_hit_then_explain_limit",
        "turns": [
            "宇宙飞船保修政策是什么？",
        ],
        "expected": {
            "first_route": "policy",
            "expected_response_mode": "explain_limit",
            "allow_handoff": False,
            "expect_policy_hits": False,
        },
    },
    {
        "name": "order_then_aftersales_create",
        "turns": [
            "帮我查一下订单 20260226003，后四位 8820，如果支持的话再帮我申请退货，因为不想要了",
        ],
        "expected": {
            "first_route": "order",
            "allow_handoff": False,
            "expect_policy_hits": False,
        },
    },
    {
        "name": "aftersales_failure_then_handoff",
        "turns": [
            "我要申请售后",
            "订单号 20260226005，后四位 6673，退货，因为商品有问题",
        ],
        "expected": {
            "first_route": "aftersales",
            "allow_handoff": True,
            "expect_policy_hits": False,
        },
    },
    {
        "name": "mixed_order_and_policy",
        "turns": [
            "帮我查一下订单 20260226003，后四位 8820，顺便说下七天无理由退货邮费谁承担",
        ],
        "expected": {
            "first_route": "order",
            "allow_handoff": False,
            "expect_policy_hits": True,
        },
    },
    {
        "name": "smalltalk_then_business",
        "turns": [
            "你好",
            "七天无理由退货邮费谁承担？",
        ],
        "expected": {
            "first_route": "general",
            "allow_handoff": False,
            "expect_policy_hits": True,
        },
    },
    {
        "name": "resume_existing_context",
        "turns": [
            "我要查订单",
            "订单号 20260226004",
            "后四位 1027",
        ],
        "expected": {
            "first_route": "order",
            "expected_clarify_slots": ["order_id", "phone_last4"],
            "allow_handoff": False,
            "expect_policy_hits": False,
        },
    },
]

