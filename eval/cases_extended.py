from __future__ import annotations


EXTENDED_AGENT_CASES_V3 = [
    # =========================
    # 1) 订单查询成功 / 身份校验失败（5）
    # =========================
    {
        "name": "order_success_a1_direct",
        "turns": [
            "帮我查一下订单 20260301001，后四位 4812",
        ],
        "expected": {
            "first_route": "order",
            "allow_handoff": False,
            "expect_policy_hits": False,
        },
    },
    {
        "name": "order_success_a4_direct",
        "turns": [
            "查订单 20260301004，手机号后四位 3408",
        ],
        "expected": {
            "first_route": "order",
            "allow_handoff": False,
            "expect_policy_hits": False,
        },
    },
    {
        "name": "order_phone_mismatch_c1",
        "turns": [
            "帮我查订单 20260301013，后四位 0000",
        ],
        "expected": {
            "first_route": "order",
            "expected_response_mode": "ask_user",
            "allow_handoff": False,
            "expect_policy_hits": False,
        },
    },
    {
        "name": "order_phone_mismatch_c4",
        "turns": [
            "查询订单 20260301016，后四位 1111",
        ],
        "expected": {
            "first_route": "order",
            "expected_response_mode": "ask_user",
            "allow_handoff": False,
            "expect_policy_hits": False,
        },
    },
    {
        "name": "order_not_found_direct",
        "turns": [
            "查一下订单 20991231001，后四位 8888",
        ],
        "expected": {
            "first_route": "order",
            "expected_response_mode": "ask_user",
            "allow_handoff": False,
            "expect_policy_hits": False,
        },
    },

    # =========================
    # 2) 多轮补槽 / resume（5）
    # =========================
    {
        "name": "order_clarify_then_resume_a2",
        "turns": [
            "帮我查订单",
            "订单号 20260301002",
            "后四位 5734",
        ],
        "expected": {
            "first_route": "order",
            "expected_clarify_slots": ["order_id", "phone_last4"],
            "allow_handoff": False,
            "expect_policy_hits": False,
        },
    },
    {
        "name": "order_clarify_missing_phone_only_a3",
        "turns": [
            "帮我查订单 20260301003",
            "后四位 9261",
        ],
        "expected": {
            "first_route": "order",
            "expected_clarify_slots": ["phone_last4"],
            "allow_handoff": False,
            "expect_policy_hits": False,
        },
    },
    {
        "name": "aftersales_create_clarify_all_slots_a5",
        "turns": [
            "我要申请售后",
            "订单号 20260301005",
            "后四位 6157，退货，因为不想要了",
        ],
        "expected": {
            "first_route": "aftersales",
            "expected_clarify_slots": ["order_id", "phone_last4", "service_type", "reason"],
            "allow_handoff": False,
            "expect_policy_hits": False,
        },
    },
    {
        "name": "aftersales_create_clarify_reason_only_a6",
        "turns": [
            "帮我给订单 20260301006，后四位 8043 申请退款",
            "收到后感觉不合适",
        ],
        "expected": {
            "first_route": "aftersales",
            "expected_clarify_slots": ["reason"],
            "allow_handoff": False,
            "expect_policy_hits": False,
        },
    },
    {
        "name": "aftersales_query_clarify_order_and_phone_f1",
        "turns": [
            "帮我查一下售后进度",
            "订单号 20260301031",
            "后四位 3185",
        ],
        "expected": {
            "first_route": "aftersales",
            "expected_clarify_slots": ["order_id", "phone_last4"],
            "allow_handoff": False,
            "expect_policy_hits": False,
        },
    },

    # =========================
    # 3) 售后创建成功 / 不允许 / 已存在（5）
    # =========================
    {
        "name": "aftersales_create_success_a1_refund",
        "turns": [
            "帮我给订单 20260301001，后四位 4812 申请退款，因为耳机佩戴不舒服",
        ],
        "expected": {
            "first_route": "aftersales",
            "allow_handoff": False,
            "expect_policy_hits": False,
        },
    },
    {
        "name": "aftersales_create_success_e1_history_but_allowed",
        "turns": [
            "帮我给订单 20260301025，后四位 3916 再申请一次退款，因为还是有问题",
        ],
        "expected": {
            "first_route": "aftersales",
            "allow_handoff": False,
            "expect_policy_hits": False,
        },
    },
    {
        "name": "aftersales_not_allowed_b1",
        "turns": [
            "帮我给订单 20260301007，后四位 1935 申请退款，因为不想要了",
        ],
        "expected": {
            "first_route": "aftersales",
            "expected_response_mode": "ask_user",
            "allow_handoff": False,
            "expect_policy_hits": False,
        },
    },
    {
        "name": "aftersales_not_allowed_b4",
        "turns": [
            "订单 20260301010，后四位 7316，帮我申请退货，因为买错了",
        ],
        "expected": {
            "first_route": "aftersales",
            "expected_response_mode": "ask_user",
            "allow_handoff": False,
            "expect_policy_hits": False,
        },
    },
    {
        "name": "aftersales_already_exists_d1",
        "turns": [
            "帮我给订单 20260301019，后四位 4402 再申请退款，因为质量问题",
        ],
        "expected": {
            "first_route": "aftersales",
            "expected_response_mode": "ask_user",
            "allow_handoff": False,
            "expect_policy_hits": False,
        },
    },

    # =========================
    # 4) 售后查询成功 / 无记录 / latest ticket（5）
    # =========================
    {
        "name": "aftersales_query_success_d2",
        "turns": [
            "帮我查一下订单 20260301020，后四位 8137 的售后进度",
        ],
        "expected": {
            "first_route": "aftersales",
            "allow_handoff": False,
            "expect_policy_hits": False,
        },
    },
    {
        "name": "aftersales_query_success_f1_latest",
        "turns": [
            "查询订单 20260301031，后四位 3185 的售后状态",
        ],
        "expected": {
            "first_route": "aftersales",
            "allow_handoff": False,
            "expect_policy_hits": False,
        },
    },
    {
        "name": "aftersales_query_success_f3_latest",
        "turns": [
            "查询订单 20260301033，后四位 4527 的售后处理到哪了",
        ],
        "expected": {
            "first_route": "aftersales",
            "allow_handoff": False,
            "expect_policy_hits": False,
        },
    },
    {
        "name": "aftersales_query_not_found_f2",
        "turns": [
            "帮我查一下订单 20260301032，后四位 7641 的售后进度",
        ],
        "expected": {
            "first_route": "aftersales",
            "expected_response_mode": "ask_user",
            "allow_handoff": False,
            "expect_policy_hits": False,
        },
    },
    {
        "name": "aftersales_query_phone_mismatch_f5",
        "turns": [
            "帮我查订单 20260301035，后四位 0000 的售后进度",
        ],
        "expected": {
            "first_route": "aftersales",
            "expected_response_mode": "ask_user",
            "allow_handoff": False,
            "expect_policy_hits": False,
        },
    },

    # =========================
    # 5) mixed intent：订单 + 售后 / 订单 + policy（5）
    # =========================
    {
        "name": "mixed_order_then_aftersales_create_a1",
        "turns": [
            "帮我查一下订单 20260301001，后四位 4812，如果支持的话再帮我申请退款，因为耳机戴着不舒服",
        ],
        "expected": {
            "first_route": "order",
            "allow_handoff": False,
            "expect_policy_hits": False,
        },
    },
    {
        "name": "mixed_order_then_aftersales_blocked_d3",
        "turns": [
            "先帮我查订单 20260301021，后四位 2714，再帮我申请退款，因为用了后不舒服",
        ],
        "expected": {
            "first_route": "order",
            "allow_handoff": False,
            "expect_policy_hits": False,
        },
    },
    {
        "name": "mixed_order_then_policy_a2",
        "turns": [
            "帮我查一下订单 20260301002，后四位 5734，顺便说下七天无理由退货邮费谁承担",
        ],
        "expected": {
            "first_route": "order",
            "allow_handoff": False,
            "expect_policy_hits": True,
        },
    },
    {
        "name": "mixed_aftersales_then_policy_a6",
        "turns": [
            "帮我给订单 20260301006，后四位 8043 申请退货，因为不想要了，另外告诉我退货运费一般谁承担",
        ],
        "expected": {
            "first_route": "aftersales",
            "allow_handoff": False,
            "expect_policy_hits": True,
        },
    },
    {
        "name": "mixed_smalltalk_then_order",
        "turns": [
            "你好呀",
            "帮我查一下订单 20260301004，后四位 3408",
        ],
        "expected": {
            "first_route": "general",
            "allow_handoff": False,
            "expect_policy_hits": False,
        },
    },

    # =========================
    # 6) 边界 / 安全收口 / 人工介入（5）
    # =========================
    {
        "name": "policy_no_hit_explain_limit_again",
        "turns": [
            "你们对火星移民仓的保修政策是什么？",
        ],
        "expected": {
            "first_route": "policy",
            "expected_response_mode": "explain_limit",
            "allow_handoff": False,
            "expect_policy_hits": False,
        },
    },
    {
        "name": "aftersales_create_not_found_then_ask_user",
        "turns": [
            "帮我给订单 20991231009，后四位 1234 申请退款，因为拍错了",
        ],
        "expected": {
            "first_route": "aftersales",
            "expected_response_mode": "ask_user",
            "allow_handoff": False,
            "expect_policy_hits": False,
        },
    },
    {
        "name": "order_then_user_requests_human",
        "turns": [
            "帮我查订单 20260301003，后四位 9261",
            "我觉得太麻烦了，直接给我转人工",
        ],
        "expected": {
            "first_route": "order",
            "allow_handoff": True,
            "expect_policy_hits": False,
        },
    },
    {
        "name": "complaint_style_handoff_request",
        "turns": [
            "你们这服务太差了，我现在就要人工处理",
        ],
        "expected": {
            "allow_handoff": True,
            "expect_policy_hits": False,
        },
    },
    {
        "name": "aftersales_query_then_human_followup",
        "turns": [
            "帮我查一下订单 20260301031，后四位 3185 的售后进度",
            "我不接受这个结果，转人工",
        ],
        "expected": {
            "first_route": "aftersales",
            "allow_handoff": True,
            "expect_policy_hits": False,
        },
    },
]

