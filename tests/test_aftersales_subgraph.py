import os
import re
import unittest
from unittest.mock import MagicMock, patch

os.environ["LANGSMITH_TRACING"] = "false"

import agent.specialists as specialists
from agent.planner import classify_intent
from agent.runtime import build_agent_graph
from agent.state import ActionType, IntentType, RetrievalEvidence, initial_state
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command


def _order_result(
    *,
    can_apply: int = 1,
    carrier_code: str = "yuantong",
    tracking_no: str = "YT123456789",
    order_status: str = "\u5df2\u652f\u4ed8",
) -> dict:
    return {
        "success": True,
        "code": "OK",
        "message": "order ready",
        "data": {
            "order_id": "20260320001",
            "phone_last4": "1234",
            "product_name": "\u8fd0\u52a8\u978b",
            "amount": 299.0,
            "order_status": order_status,
            "pay_status": "\u5df2\u652f\u4ed8",
            "created_at": "2026-03-20 10:00:00",
            "carrier_code": carrier_code,
            "tracking_no": tracking_no,
            "can_apply_aftersales": can_apply,
            "source": "order_tool",
        },
    }


def _logistics_tool_result(delivery_state: str = "in_transit") -> dict:
    flags = {
        "signed": (True, False, False),
        "received": (True, False, False),
        "in_transit": (False, False, False),
        "delivering": (False, False, False),
        "abnormal": (False, False, True),
        "returning": (False, True, False),
    }
    is_signed, is_returning, is_abnormal = flags.get(delivery_state, (False, False, False))
    return {
        "success": True,
        "code": "OK",
        "message": "snapshot ready",
        "data": {
            "carrier_code": "yuantong",
            "tracking_no": "YT123456789",
            "delivery_state": delivery_state,
            "delivery_state_name": {
                "in_transit": "\u8fd0\u8f93\u4e2d",
                "signed": "\u5df2\u7b7e\u6536",
                "delivering": "\u6d3e\u9001\u4e2d",
                "abnormal": "\u5f02\u5e38",
                "returning": "\u9000\u56de\u4e2d",
            }.get(delivery_state, delivery_state),
            "delivery_status_code": delivery_state,
            "last_event": "\u5305\u88f9\u5df2\u5230\u8fbe\u6606\u5c71\u7ad9\u70b9",
            "last_event_time": "2026-03-21 10:00:00",
            "current_location": "\u6606\u5c71",
            "route_from": "\u798f\u5dde",
            "route_to": "\u6606\u5c71",
            "is_signed": is_signed,
            "is_returning": is_returning,
            "is_abnormal": is_abnormal,
            "source": "mock_logistics",
            "fetched_at": "2026-03-21T10:00:00Z",
            "raw_payload_ref": "",
            "_cache_meta": {
                "cache_key": "yuantong:YT123456789",
                "cache_hit": False,
                "last_query_at": "2026-03-21T10:00:00Z",
                "ttl_minutes": 30,
            },
        },
    }


def _logistics_rate_limited_result() -> dict:
    return {
        "success": False,
        "code": "QUERY_TOO_FREQUENT",
        "message": "query too frequent",
        "data": {
            "provider": "cache_guard",
            "success": False,
            "error_code": "QUERY_TOO_FREQUENT",
            "error_message": "query too frequent",
            "retryable": True,
            "suggested_action": "retry_later",
            "_cache_meta": {
                "cache_key": "yuantong:YT123456789",
                "cache_hit": False,
                "last_query_at": "2026-03-21T10:00:00Z",
                "ttl_minutes": 30,
            },
        },
        "missing_slots": [],
    }


def _policy_hits() -> list[RetrievalEvidence]:
    return [
        RetrievalEvidence(
            query_used="\u4e03\u5929\u65e0\u7406\u7531\u9000\u8d27",
            rewritten_from="",
            score=0.95,
            source="kb",
            title="\u4e03\u5929\u65e0\u7406\u7531\u9000\u8d27",
            chunk_id="chunk-7day",
            text="\u4e03\u5929\u65e0\u7406\u7531\u4e00\u822c\u652f\u6301\u7b7e\u6536\u540e\u4e03\u5929\u5185\u7533\u8bf7\u3002",
            evidence_summary="\u4e03\u5929\u65e0\u7406\u7531\u4e00\u822c\u652f\u6301\u7b7e\u6536\u540e\u4e03\u5929\u5185\u53d1\u8d77\u9000\u8d27\u7533\u8bf7\uff0c\u5177\u4f53\u4ee5\u9875\u9762\u8bf4\u660e\u4e3a\u51c6\u3002",
        )
    ]


class EscalationRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.order_mock = MagicMock(return_value=_order_result())
        self.logistics_mock = MagicMock(return_value=_logistics_tool_result("signed"))
        self.create_mock = MagicMock(
            return_value={
                "success": True,
                "code": "OK",
                "message": "created",
                "data": {
                    "ticket_id": "AS-2001",
                    "order_id": "20260320001",
                    "service_type": "\u9000\u8d27",
                    "reason": "\u4e0d\u5408\u9002",
                    "ticket_status": "\u5f85\u5ba1\u6838",
                    "source": "aftersales_tool",
                },
            }
        )
        self.query_mock = MagicMock(
            return_value={
                "success": True,
                "code": "OK",
                "message": "query ready",
                "data": {
                    "ticket_id": "AS-2001",
                    "order_id": "20260320001",
                    "service_type": "\u9000\u8d27",
                    "reason": "\u4e0d\u5408\u9002",
                    "ticket_status": "\u5904\u7406\u4e2d",
                    "updated_at": "2026-03-22 12:00:00",
                    "source": "aftersales_tool",
                },
            }
        )
        self.policy_mock = MagicMock(return_value=_policy_hits())
        self.handoff_mock = MagicMock(
            return_value={
                "success": True,
                "code": "HANDOFF_CREATED",
                "message": "handoff created",
                "data": {
                    "handoff_id": "HO-9001",
                    "status": "waiting_human",
                },
            }
        )

        self.patchers = [
            patch.object(specialists.TOOL_REGISTRY["get_order_info_tool"], "callable", new=self.order_mock),
            patch.object(specialists.TOOL_REGISTRY["query_logistics_snapshot_tool"], "callable", new=self.logistics_mock),
            patch.object(specialists.TOOL_REGISTRY["create_aftersales_tool"], "callable", new=self.create_mock),
            patch.object(specialists.TOOL_REGISTRY["query_aftersales_tool"], "callable", new=self.query_mock),
            patch.object(specialists.TOOL_REGISTRY["handoff_to_human_tool"], "callable", new=self.handoff_mock),
            patch.object(specialists, "retrieve_policy_evidence", new=self.policy_mock),
        ]
        for patcher in self.patchers:
            patcher.start()

        self.graph = build_agent_graph().compile(checkpointer=InMemorySaver())
        self.session_id = f"runtime-{self._testMethodName}"

    def tearDown(self) -> None:
        for patcher in reversed(self.patchers):
            patcher.stop()

    def _config(self) -> dict:
        return {"configurable": {"thread_id": self.session_id}}

    def _snapshot(self) -> dict:
        snapshot = self.graph.get_state(self._config())
        state = dict(snapshot.values)
        if snapshot.interrupts:
            payload = getattr(snapshot.interrupts[0], "value", None)
            if isinstance(payload, dict):
                state["interrupt_payload"] = payload
                state["final_response"] = str(payload.get("pending_question") or state.get("final_response") or "")
        return state

    def _invoke_turn(self, user_input: str) -> dict:
        self.graph.invoke({"session_id": self.session_id, "user_input": user_input}, config=self._config())
        return self._snapshot()

    def _resume_turn(self, user_input: str) -> dict:
        self.graph.invoke(Command(resume={"user_input": user_input}), config=self._config())
        return self._snapshot()

    def _complete_create_aftersales(self, reason: str = "\u4e0d\u5408\u9002") -> tuple[dict, dict, dict]:
        turn1 = self._invoke_turn("\u6211\u8981\u9000\u8d27")
        turn2 = self._resume_turn("20260320001\uff0c1234")
        turn3 = self._resume_turn(reason)
        return turn1, turn2, turn3

    def test_scenario_1_user_directly_requests_human(self) -> None:
        state = self._invoke_turn("\u4f60\u522b\u67e5\u4e86\uff0c\u76f4\u63a5\u7ed9\u6211\u8f6c\u4eba\u5de5")
        self.assertEqual(state["intent_type"], IntentType.ESCALATION)
        self.assertEqual(state["escalation_type"], "HUMAN_REQUEST")
        self.assertEqual(state["escalation_decision"], "HANDOFF_HUMAN")
        self.assertIn("\u8f6c\u5165\u4eba\u5de5\u5904\u7406", state["final_response"])
        self.assertEqual(self.handoff_mock.call_count, 1)

    def test_scenario_1b_handoff_failure_downgrades_to_case(self) -> None:
        failing_handoff = MagicMock(
            return_value={
                "success": False,
                "code": "HANDOFF_UNAVAILABLE",
                "message": "handoff unavailable",
                "data": {},
            }
        )
        with patch.object(specialists.TOOL_REGISTRY["handoff_to_human_tool"], "callable", new=failing_handoff):
            state = self._invoke_turn("\u4f60\u522b\u67e5\u4e86\uff0c\u76f4\u63a5\u7ed9\u6211\u8f6c\u4eba\u5de5")

        self.assertEqual(state["intent_type"], IntentType.ESCALATION)
        self.assertEqual(state["escalation_decision"], "CREATE_CASE")
        self.assertNotIn("\u8f6c\u5165\u4eba\u5de5\u5904\u7406", state["final_response"])
        self.assertIn("\u5347\u7ea7\u5904\u7406\u8bb0\u5f55", state["final_response"])
        self.assertEqual(state["handoff_case_payload"]["decision"], "CREATE_CASE")
        self.assertTrue(state["handoff_case_payload"]["handoff_failed"])
        self.assertRegex(state["handoff_case_payload"]["case_id"], r"^ESC-[0-9A-F]{12}$")
        fallback_observation = next(
            observation for observation in state["observations"] if observation.code == "ESCALATION_HANDOFF_FALLBACK_CASE"
        )
        self.assertEqual(fallback_observation.summary, "\u4eba\u5de5\u8f6c\u63a5\u5931\u8d25\uff0c\u5df2\u6539\u4e3a\u751f\u6210\u5347\u7ea7\u5de5\u5355\u8bb0\u5f55\u3002")

    def test_scenario_2_normal_order_query_stays_in_order_domain(self) -> None:
        state = self._invoke_turn("\u5e2e\u6211\u67e5\u4e00\u4e0b\u8ba2\u5355\u72b6\u6001\uff0c\u8ba2\u5355\u53f7 20260320001\uff0c\u624b\u673a\u53f7\u540e\u56db\u4f4d 1234")
        self.assertEqual(state["intent_type"], IntentType.ORDER)
        self.assertEqual(state["order_action"], ActionType.QUERY_ORDER)
        self.assertEqual(state.get("escalation_decision") or "", "")
        self.assertEqual(self.order_mock.call_count, 1)
        self.assertIn("\u8ba2\u5355\u72b6\u6001", state["final_response"])

    def test_scenario_3_strong_complaint_and_compensation_routes_to_escalation(self) -> None:
        state = self._invoke_turn("\u4f60\u4eec\u8fd9\u7269\u6d41\u4e00\u76f4\u4e0d\u66f4\u65b0\uff0c\u6211\u8981\u6295\u8bc9\uff0c\u8fd8\u8981\u8d54\u507f")
        self.assertEqual(state["intent_type"], IntentType.ESCALATION)
        self.assertEqual(state["escalation_type"], "COMPENSATION_DISPUTE")
        self.assertEqual(state["escalation_decision"], "CREATE_CASE")
        self.assertIn("\u5347\u7ea7\u5904\u7406\u8bb0\u5f55", state["final_response"])
        self.assertEqual(self.handoff_mock.call_count, 0)

    def test_scenario_4_system_continuous_failure_escalates(self) -> None:
        logistics_failure = {
            "success": False,
            "code": "500",
            "message": "logistics provider unavailable",
            "data": {
                "provider": "kuaidi100",
                "success": False,
                "error_code": "500",
                "error_message": "provider unavailable",
                "retryable": True,
                "suggested_action": "retry_later",
                "missing_slots": [],
            },
            "missing_slots": [],
        }
        with patch.object(
            specialists.TOOL_REGISTRY["query_logistics_snapshot_tool"],
            "callable",
            new=MagicMock(side_effect=[logistics_failure, logistics_failure]),
        ) as failing_logistics:
            state = self._invoke_turn("\u67e5\u4e00\u4e0b\u8fd9\u4e2a\u8ba2\u5355\u7269\u6d41\uff0c\u8ba2\u5355\u53f7 20260320001\uff0c\u624b\u673a\u53f7\u540e\u56db\u4f4d 1234")

        self.assertEqual(state["intent_type"], IntentType.ORDER)
        self.assertEqual(state["escalation_type"], "SYSTEM_FAILURE_ESCALATION")
        self.assertEqual(state["escalation_decision"], "CREATE_CASE")
        self.assertIn("\u5347\u7ea7\u5904\u7406\u8bb0\u5f55", state["final_response"])
        self.assertEqual(failing_logistics.call_count, 2)

    def test_scenario_5_rule_and_fact_conflict_escalates(self) -> None:
        state = self._invoke_turn("\u4f60\u4eec\u89c4\u5219\u8bf4\u53ef\u4ee5\u9000\uff0c\u4f46\u7cfb\u7edf\u53c8\u4e0d\u7ed9\u6211\u9000\uff0c\u8fd9\u600e\u4e48\u56de\u4e8b")
        self.assertIn(state["intent_type"], {IntentType.POLICY, IntentType.ORDER, IntentType.MIXED})
        self.assertEqual(state["escalation_type"], "RULE_FACT_CONFLICT")
        self.assertEqual(state["escalation_decision"], "CREATE_CASE")
        self.assertIn("\u5347\u7ea7\u5904\u7406\u8bb0\u5f55", state["final_response"])

    def test_direct_logistics_query_with_explicit_phone_label_still_works(self) -> None:
        state = self._invoke_turn("\u90a3\u5e2e\u6211\u67e5\u4e00\u4e0b\u8fd9\u4e2a\u8ba2\u5355\u7684\u7269\u6d41\uff0c\u8ba2\u5355\u53f7\u4e3a '20260320001'\uff0c\u624b\u673a\u53f7\u4e3a '1234'")
        self.assertEqual(state["intent_type"], IntentType.ORDER)
        self.assertEqual(state["order_action"], ActionType.QUERY_LOGISTICS)
        self.assertEqual(self.order_mock.call_count, 1)
        self.assertEqual(self.logistics_mock.call_count, 1)
        self.assertNotIn("\u8bf7\u63d0\u4f9b\u8ba2\u5355\u53f7", state["final_response"])

    def test_query_too_frequent_with_cached_result_prefers_cached_logistics_reply(self) -> None:
        rate_limited_logistics = MagicMock(side_effect=[_logistics_tool_result("signed"), _logistics_rate_limited_result()])
        with patch.object(specialists.TOOL_REGISTRY["query_logistics_snapshot_tool"], "callable", new=rate_limited_logistics):
            first = self._invoke_turn(
                "\u5e2e\u6211\u67e5\u4e00\u4e0b\u8fd9\u4e2a\u8ba2\u5355\u7684\u7269\u6d41\uff0c\u8ba2\u5355\u53f7 20260320001\uff0c\u624b\u673a\u53f7\u540e\u56db\u4f4d 1234"
            )
            second = self._invoke_turn(
                "\u518d\u5e2e\u6211\u67e5\u4e00\u4e0b\u8fd9\u4e2a\u8ba2\u5355\u7684\u7269\u6d41\uff0c\u8ba2\u5355\u53f7 20260320001\uff0c\u624b\u673a\u53f7\u540e\u56db\u4f4d 1234"
            )

        self.assertIn("\u5df2\u7b7e\u6536", first["final_response"])
        self.assertEqual(second["order_action"], ActionType.QUERY_LOGISTICS)
        self.assertEqual(second.get("escalation_decision") or "", "")
        self.assertIn("\u6700\u8fd1\u4e00\u6b21\u7f13\u5b58\u7ed3\u679c", second["final_response"])
        self.assertIn("\u5df2\u7b7e\u6536", second["final_response"])
        self.assertEqual(rate_limited_logistics.call_count, 2)

    def test_query_too_frequent_without_cache_returns_friendly_message(self) -> None:
        with patch.object(
            specialists.TOOL_REGISTRY["query_logistics_snapshot_tool"],
            "callable",
            new=MagicMock(return_value=_logistics_rate_limited_result()),
        ) as rate_limited_logistics:
            state = self._invoke_turn(
                "\u5e2e\u6211\u67e5\u4e00\u4e0b\u8fd9\u4e2a\u8ba2\u5355\u7684\u7269\u6d41\uff0c\u8ba2\u5355\u53f7 20260320001\uff0c\u624b\u673a\u53f7\u540e\u56db\u4f4d 1234"
            )

        self.assertEqual(state["order_action"], ActionType.QUERY_LOGISTICS)
        self.assertEqual(state.get("escalation_decision") or "", "")
        self.assertEqual(
            state["final_response"],
            "\u8be5\u7269\u6d41\u5355\u53f7\u6700\u8fd1 30 \u5206\u949f\u5185\u5df2\u67e5\u8be2\u8fc7\uff0c\u4e3a\u907f\u514d\u63a5\u53e3\u9501\u5b9a\uff0c\u6682\u65f6\u4e0d\u80fd\u91cd\u590d\u8c03\u7528\u3002\u8bf7\u7a0d\u540e\u518d\u8bd5\u3002",
        )
        self.assertEqual(rate_limited_logistics.call_count, 1)

    def test_mixed_order_and_policy_runs_both_branches(self) -> None:
        state = self._invoke_turn("\u67e5\u8ba2\u5355 20260320007\uff0c\u540e\u56db\u4f4d 9156\uff0c\u5e76\u544a\u8bc9\u6211\u4e03\u5929\u65e0\u7406\u7531\u9000\u8d27\u89c4\u5219")
        self.assertEqual(state["intent_type"], IntentType.MIXED)
        self.assertEqual(state["order_action"], ActionType.QUERY_ORDER)
        self.assertIsNotNone(state["order_context"])
        self.assertGreater(len(state["retrieval_evidence"]), 0)
        self.assertEqual(self.order_mock.call_count, 1)
        self.assertEqual(self.policy_mock.call_count, 1)
        self.assertIn("\n\n", state["final_response"])

    def test_create_aftersales_happy_path_still_works(self) -> None:
        turn1 = self._invoke_turn("\u6211\u8981\u9000\u8d27")
        self.assertEqual(turn1["intent_type"], IntentType.ORDER)
        self.assertEqual(turn1["order_action"], ActionType.CREATE_AFTERSALES)
        self.assertIn("\u8ba2\u5355\u53f7", turn1["final_response"])

        turn2 = self._resume_turn("20260320001\uff0c1234")
        self.assertIn("\u539f\u56e0", turn2["final_response"])

        turn3 = self._resume_turn("\u4e0d\u5408\u9002")
        self.assertEqual(turn3["order_action"], ActionType.CREATE_AFTERSALES)
        self.assertIn("\u552e\u540e\u5355\u53f7", turn3["final_response"])
        self.assertTrue(turn3["trace_tags"]["aftersales_requires_logistics"])
        source_names = [observation.source_name for observation in turn3["observations"]]
        self.assertLess(source_names.index("query_logistics_snapshot_tool"), source_names.index("create_aftersales_tool"))
        self.assertEqual(self.logistics_mock.call_count, 1)
        self.assertEqual(self.create_mock.call_count, 1)

    def test_create_aftersales_in_transit_is_blocked(self) -> None:
        with patch.object(
            specialists.TOOL_REGISTRY["query_logistics_snapshot_tool"],
            "callable",
            new=MagicMock(return_value=_logistics_tool_result("in_transit")),
        ) as blocked_logistics:
            _, _, turn3 = self._complete_create_aftersales()

        self.assertEqual(turn3["aftersales_context"].eligibility, "NOT_ALLOWED")
        self.assertEqual(self.create_mock.call_count, 0)
        self.assertEqual(blocked_logistics.call_count, 1)
        self.assertIn("\u6682\u4e0d\u652f\u6301\u81ea\u52a8\u63d0\u4ea4", turn3["final_response"])

    def test_create_aftersales_delivering_is_blocked(self) -> None:
        with patch.object(
            specialists.TOOL_REGISTRY["query_logistics_snapshot_tool"],
            "callable",
            new=MagicMock(return_value=_logistics_tool_result("delivering")),
        ) as blocked_logistics:
            _, _, turn3 = self._complete_create_aftersales()

        self.assertEqual(turn3["aftersales_context"].eligibility, "NOT_ALLOWED")
        self.assertEqual(self.create_mock.call_count, 0)
        self.assertEqual(blocked_logistics.call_count, 1)
        self.assertIn("\u6682\u4e0d\u652f\u6301\u81ea\u52a8\u63d0\u4ea4", turn3["final_response"])

    def test_create_aftersales_without_logistics_identifiers_is_explainable(self) -> None:
        missing_logistics_order = MagicMock(return_value=_order_result(carrier_code="", tracking_no=""))
        with patch.object(specialists.TOOL_REGISTRY["get_order_info_tool"], "callable", new=missing_logistics_order):
            _, _, turn3 = self._complete_create_aftersales()

        self.assertEqual(turn3["aftersales_context"].eligibility, "NEED_MANUAL")
        self.assertEqual(self.logistics_mock.call_count, 0)
        self.assertEqual(self.create_mock.call_count, 0)
        self.assertIn("\u7269\u6d41\u5355\u53f7\u6216\u627f\u8fd0\u5546", turn3["final_response"])

    def test_aftersales_requires_logistics_trace_tag_is_stable(self) -> None:
        _, _, create_state = self._complete_create_aftersales()
        query_state = self._invoke_turn(
            "\u6211\u60f3\u67e5\u4e00\u4e0b\u8fd9\u4e2a\u8ba2\u5355\u7684\u552e\u540e\u8fdb\u5ea6\uff0c\u8ba2\u5355\u53f7 20260320001\uff0c\u624b\u673a\u53f7\u540e\u56db\u4f4d 1234"
        )
        order_state = self._invoke_turn(
            "\u5e2e\u6211\u67e5\u4e00\u4e0b\u8ba2\u5355\u72b6\u6001\uff0c\u8ba2\u5355\u53f7 20260320001\uff0c\u624b\u673a\u53f7\u540e\u56db\u4f4d 1234"
        )
        logistics_state = self._invoke_turn(
            "\u5e2e\u6211\u67e5\u4e00\u4e0b\u8fd9\u4e2a\u8ba2\u5355\u7684\u7269\u6d41\uff0c\u8ba2\u5355\u53f7 20260320001\uff0c\u624b\u673a\u53f7\u540e\u56db\u4f4d 1234"
        )

        self.assertIs(create_state["trace_tags"].get("aftersales_requires_logistics"), True)
        self.assertIs(query_state["trace_tags"].get("aftersales_requires_logistics"), False)
        self.assertIs(order_state["trace_tags"].get("aftersales_requires_logistics"), False)
        self.assertIs(logistics_state["trace_tags"].get("aftersales_requires_logistics"), False)

    def test_classify_direct_escalation_phrase(self) -> None:
        state = initial_state("intent-escalation")
        state["user_input"] = "\u4f60\u5904\u7406\u4e0d\u4e86\uff0c\u7ed9\u6211\u627e\u4eba\u5de5"
        self.assertEqual(classify_intent(state), IntentType.ESCALATION)

    def test_classify_mixed_order_and_policy_phrase(self) -> None:
        state = initial_state("intent-mixed")
        state["user_input"] = "\u67e5\u8ba2\u5355 20260320007\uff0c\u540e\u56db\u4f4d 9156\uff0c\u5e76\u544a\u8bc9\u6211\u4e03\u5929\u65e0\u7406\u7531\u9000\u8d27\u89c4\u5219"
        self.assertEqual(classify_intent(state), IntentType.MIXED)

    def test_switching_to_new_order_clears_failure_window(self) -> None:
        logistics_failure = {
            "success": False,
            "code": "500",
            "message": "logistics provider unavailable",
            "data": {
                "provider": "kuaidi100",
                "success": False,
                "error_code": "500",
                "error_message": "provider unavailable",
                "retryable": True,
                "suggested_action": "retry_later",
                "missing_slots": [],
            },
            "missing_slots": [],
        }
        fresh_success = _logistics_tool_result("in_transit")
        flaky_logistics = MagicMock(side_effect=[logistics_failure, logistics_failure, logistics_failure, fresh_success])
        with patch.object(specialists.TOOL_REGISTRY["query_logistics_snapshot_tool"], "callable", new=flaky_logistics):
            first = self._invoke_turn("\u67e5\u4e00\u4e0b\u8fd9\u4e2a\u8ba2\u5355\u7269\u6d41\uff0c\u8ba2\u5355\u53f7 20260320001\uff0c\u624b\u673a\u53f7\u540e\u56db\u4f4d 1234")
            second = self._invoke_turn("\u67e5\u4e00\u4e0b\u8fd9\u4e2a\u8ba2\u5355\u7269\u6d41\uff0c\u8ba2\u5355\u53f7 20260320007\uff0c\u624b\u673a\u53f7\u540e\u56db\u4f4d 9156")

        self.assertEqual(first["escalation_type"], "SYSTEM_FAILURE_ESCALATION")
        self.assertEqual(first["escalation_decision"], "CREATE_CASE")
        self.assertEqual(second["intent_type"], IntentType.ORDER)
        self.assertEqual(second["order_action"], ActionType.QUERY_LOGISTICS)
        self.assertEqual(second.get("escalation_decision") or "", "")
        self.assertTrue(second["final_response"])
        self.assertEqual(flaky_logistics.call_count, 4)


if __name__ == "__main__":
    unittest.main()
