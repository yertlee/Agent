import hashlib
import json
import os
import tempfile
import unittest
import urllib.parse
from typing import Dict
from unittest.mock import patch

os.environ["LANGSMITH_TRACING"] = "false"

from agent.kuaidi100_provider import Kuaidi100Provider
from agent.logistics_provider import query_logistics_snapshot_with_fallback
from agent.logistics_types import LogisticsRequest, error_result, now_timestamp, success_result
from agent.specialists import _tool_observation, eligibility_check_node
from agent.state import (
    ActionType,
    LogisticsSnapshot,
    OrderContext,
    PlanStep,
    ResponseMode,
    SpecialistName,
    extract_slots_from_text,
    initial_state,
)
from agent.tools import query_logistics_snapshot
from agent.verifier import verify_state


def _snapshot(delivery_state: str, *, delivery_state_name: str, delivery_status_code: str) -> Dict[str, object]:
    flags = {
        "signed": (True, False, False),
        "in_transit": (False, False, False),
        "delivering": (False, False, False),
        "abnormal": (False, False, True),
        "returning": (False, True, False),
    }
    is_signed, is_returning, is_abnormal = flags.get(delivery_state, (False, False, False))
    return {
        "carrier_code": "yuantong",
        "tracking_no": "YT25569986666541",
        "delivery_state": delivery_state,
        "delivery_state_name": delivery_state_name,
        "delivery_status_code": delivery_status_code,
        "last_event": "latest event",
        "last_event_time": "2026-03-20 10:00:00",
        "current_location": "Kunshan",
        "route_from": "Shenzhen",
        "route_to": "Kunshan",
        "route_info": {
            "from": {"number": "CN440305000000", "name": "Shenzhen"},
            "cur": {"number": "CN320583000000", "name": "Kunshan"},
            "to": {"number": "CN320583000000", "name": "Kunshan"},
        },
        "arrival_time": "2026-03-21 18",
        "predicted_route": [{"name": "Suzhou hub", "state": "predicted"}],
        "is_signed": is_signed,
        "is_returning": is_returning,
        "is_abnormal": is_abnormal,
        "source": "kuaidi100_api",
        "fetched_at": now_timestamp(),
        "raw_payload_ref": "",
    }


class StaticProvider:
    provider_name = "kuaidi100"

    def __init__(self, result):
        self._result = result

    def query(self, request):
        return self._result


class ExplodingProvider:
    provider_name = "kuaidi100"

    def query(self, request):
        raise AssertionError("provider should not be called on cache hit")


class FakeResponse:
    def __init__(self, payload: Dict[str, object]) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload, ensure_ascii=False).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class LogisticsPipelineTests(unittest.TestCase):
    def _set_cache_env(self, cache_dir: str):
        return patch.dict(
            os.environ,
            {
                "LOGISTICS_CACHE_DIR": cache_dir,
                "LOGISTICS_CACHE_TTL_MINUTES": "30",
                "LOGISTICS_MIN_QUERY_INTERVAL_MINUTES": "30",
                "LANGSMITH_TRACING": "false",
            },
            clear=False,
        )

    def _make_create_aftersales_state(self, delivery_state: str) -> dict:
        state = initial_state("test-session")
        state["current_plan"] = [
            PlanStep(
                step_id="step-1",
                owner_agent=SpecialistName.ORDER,
                action_type=ActionType.CREATE_AFTERSALES,
                goal="check eligibility",
                success_condition="eligibility ready",
                fallback_action="handoff",
            )
        ]
        state["current_step_index"] = 0
        state["slot_values"] = {"service_type": "退款", "reason": "不想要了"}
        state["order_context"] = OrderContext(order_id="20260320001", phone_last4="1234", can_apply_aftersales=1)
        state["logistics_snapshot"] = LogisticsSnapshot(
            **_snapshot(delivery_state, delivery_state_name=delivery_state, delivery_status_code=delivery_state)
        )
        return state

    def test_extract_slots_prefers_tracking_number_label(self) -> None:
        slots = extract_slots_from_text("帮我看一下物流，运单号 YT25569986666541")
        self.assertEqual(slots.get("tracking_no"), "YT25569986666541")
        self.assertNotIn("order_id", slots)

    def test_extract_slots_uses_awaited_phone_last4_and_reason(self) -> None:
        slots = extract_slots_from_text("20260320001，1234", awaited_slots=["order_id", "phone_last4"])
        self.assertEqual(slots.get("order_id"), "20260320001")
        self.assertEqual(slots.get("phone_last4"), "1234")

        reason_slots = extract_slots_from_text("不想要了", awaited_slots=["reason"])
        self.assertEqual(reason_slots.get("reason"), "不想要了")

    def test_extract_slots_accepts_explicit_phone_label_without_awaiting(self) -> None:
        slots = extract_slots_from_text("订单号为'20260320007',手机号为'9156'")
        self.assertEqual(slots.get("order_id"), "20260320007")
        self.assertEqual(slots.get("phone_last4"), "9156")

    def test_cache_hit_skips_provider_after_first_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as cache_dir, self._set_cache_env(cache_dir):
            first_result = success_result(
                provider="kuaidi100",
                message="snapshot ready",
                snapshot=_snapshot("signed", delivery_state_name="signed", delivery_status_code="3"),
            )
            with patch("agent.logistics_provider._select_provider", return_value=StaticProvider(first_result)):
                first = query_logistics_snapshot("yuantong", "YT25569986666541", "1234")

            self.assertTrue(first["success"])
            self.assertFalse(first["data"]["_cache_meta"]["cache_hit"])

            with patch("agent.logistics_provider._select_provider", return_value=ExplodingProvider()):
                second = query_logistics_snapshot("yuantong", "YT25569986666541", "1234")

            self.assertTrue(second["success"])
            self.assertEqual(second["data"]["source"], "cache")
            self.assertTrue(second["data"]["_cache_meta"]["cache_hit"])

    def test_kuaidi100_provider_builds_signed_form_request_and_maps_response(self) -> None:
        captured = {}
        payload = {
            "message": "ok",
            "nu": "YT25569986666541",
            "ischeck": "0",
            "com": "yuantong",
            "status": "200",
            "data": [
                {
                    "time": "2025-06-13 15:56:19",
                    "context": "package in transit",
                    "ftime": "2025-06-13 15:56:19",
                    "status": "in transit",
                    "location": "Lu'an",
                    "statusCode": "0",
                }
            ],
            "state": "0",
            "routeInfo": {
                "from": {"number": "CN430112000000", "name": "Changsha"},
                "cur": {"number": "CN341522000000", "name": "Lu'an"},
                "to": {"number": "CN320583104000", "name": "Kunshan"},
            },
            "arrivalTime": "2025-06-14 13",
            "predictedRoute": [
                {
                    "arriveTime": "2025-06-13 19:05:19",
                    "leaveTime": "2025-06-13 19:59:19",
                    "province": "Anhui",
                    "city": "Lu'an",
                    "district": "Jin'an",
                    "name": "Lu'an node",
                    "state": "predicted",
                    "type": "site",
                }
            ],
        }

        def fake_urlopen(request, timeout=0):
            captured["request"] = request
            captured["timeout"] = timeout
            return FakeResponse(payload)

        provider = Kuaidi100Provider(customer="customer-demo", key="key-demo", resultv2="4")
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            result = provider.query(LogisticsRequest(carrier_code="yuantong", tracking_no="YT25569986666541", phone_last4="1234"))

        self.assertTrue(result.success)
        self.assertEqual(captured["request"].get_method(), "POST")
        self.assertEqual(captured["request"].get_header("Content-type"), "application/x-www-form-urlencoded")
        body = urllib.parse.parse_qs(captured["request"].data.decode("utf-8"))
        param_str = body["param"][0]
        self.assertEqual(body["customer"][0], "customer-demo")
        self.assertEqual(
            body["sign"][0],
            hashlib.md5(f"{param_str}key-democustomer-demo".encode("utf-8")).hexdigest().upper(),
        )
        self.assertIn('"resultv2":"4"', param_str)
        self.assertEqual(result.snapshot["delivery_state"], "in_transit")
        self.assertEqual(result.snapshot["current_location"], "Lu'an")
        self.assertEqual(result.snapshot["route_to"], "Kunshan")
        self.assertEqual(result.snapshot["arrival_time"], "2025-06-14 13")

    def test_cache_miss_api_success_returns_standardized_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as cache_dir, self._set_cache_env(cache_dir):
            provider_result = success_result(
                provider="kuaidi100",
                message="snapshot ready",
                snapshot=_snapshot("in_transit", delivery_state_name="in_transit", delivery_status_code="0"),
            )
            with patch("agent.logistics_provider._select_provider", return_value=StaticProvider(provider_result)):
                result = query_logistics_snapshot_with_fallback(
                    carrier_code="yuantong",
                    tracking_no="YT25569986666541",
                    phone_last4="1234",
                )

            self.assertTrue(result["success"])
            self.assertEqual(result["data"]["delivery_state"], "in_transit")
            self.assertEqual(result["data"]["route_info"]["to"]["name"], "Kunshan")
            self.assertFalse(result["data"]["_cache_meta"]["cache_hit"])
            self.assertTrue(result["data"]["raw_payload_ref"].endswith(".json"))

    def test_eligibility_judges_logistics_states(self) -> None:
        cases = [
            ("signed", "ALLOWED"),
            ("in_transit", "NOT_ALLOWED"),
            ("delivering", "NOT_ALLOWED"),
            ("abnormal", "NEED_MANUAL"),
            ("returning", "NEED_MANUAL"),
        ]
        for delivery_state, expected in cases:
            with self.subTest(delivery_state=delivery_state):
                state = self._make_create_aftersales_state(delivery_state)
                patch_state = eligibility_check_node(state)
                self.assertEqual(patch_state["trace_tags"]["aftersales_eligibility"], expected)

    def test_logistics_error_is_structured(self) -> None:
        with tempfile.TemporaryDirectory() as cache_dir, self._set_cache_env(cache_dir):
            provider_result = error_result(
                provider="kuaidi100",
                error_code="408",
                error_message="phone tail mismatch",
                retryable=False,
                suggested_action="ask_user",
                missing_slots=["phone_last4"],
            )
            with patch("agent.logistics_provider._select_provider", return_value=StaticProvider(provider_result)):
                result = query_logistics_snapshot("yuantong", "YT25569986666541")

        self.assertFalse(result["success"])
        self.assertEqual(result["code"], "408")
        self.assertEqual(result["data"]["provider"], "kuaidi100")
        self.assertEqual(result["data"]["suggested_action"], "ask_user")
        self.assertEqual(result["missing_slots"], ["phone_last4"])

    def test_query_too_frequent_single_attempt_does_not_escalate(self) -> None:
        tool_result = {
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
                    "cache_key": "yuantong:YT25569986666541",
                    "cache_hit": False,
                    "last_query_at": "2026-03-20T10:00:00Z",
                    "ttl_minutes": 30,
                },
            },
            "missing_slots": [],
        }
        observation = _tool_observation("step-1", "query_logistics_snapshot_tool", tool_result, False)
        state = initial_state("verify-too-frequent-single")
        state["current_plan"] = [
            PlanStep(
                step_id="step-1",
                owner_agent=SpecialistName.ORDER,
                action_type=ActionType.QUERY_LOGISTICS,
                goal="query logistics",
                success_condition="done",
                fallback_action="handoff",
            )
        ]
        state["current_step_index"] = 0
        state["last_observation"] = observation
        state["retry_count_by_stage"] = {}

        verification = verify_state(state)
        self.assertTrue(verification.can_finalize)
        self.assertFalse(verification.should_escalate)
        self.assertFalse(verification.retry_same_step)
        self.assertEqual(verification.recommended_response_mode, ResponseMode.EXPLAIN_LIMIT)

    def test_query_too_frequent_repeated_attempt_does_not_escalate(self) -> None:
        tool_result = {
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
                    "cache_key": "yuantong:YT25569986666541",
                    "cache_hit": False,
                    "last_query_at": "2026-03-20T10:00:00Z",
                    "ttl_minutes": 30,
                },
            },
            "missing_slots": [],
        }
        observation = _tool_observation("step-1", "query_logistics_snapshot_tool", tool_result, False)
        state = initial_state("verify-too-frequent-repeat")
        state["current_plan"] = [
            PlanStep(
                step_id="step-1",
                owner_agent=SpecialistName.ORDER,
                action_type=ActionType.QUERY_LOGISTICS,
                goal="query logistics",
                success_condition="done",
                fallback_action="handoff",
            )
        ]
        state["current_step_index"] = 0
        state["last_observation"] = observation
        state["retry_count_by_stage"] = {"step-1:query_logistics_snapshot_tool": 3}

        verification = verify_state(state)
        self.assertTrue(verification.can_finalize)
        self.assertFalse(verification.should_escalate)
        self.assertEqual(verification.recommended_response_mode, ResponseMode.EXPLAIN_LIMIT)

    def test_verifier_maps_logistics_error_actions(self) -> None:
        scenarios = [
            ("408", "ask_user", ["phone_last4"], "must_ask_user", ResponseMode.ASK_USER),
            ("400", "ask_user", [], "can_finalize", ResponseMode.EXPLAIN_LIMIT),
            ("500", "retry_later", [], "retry_same_step", ResponseMode.IDLE),
            ("503", "handoff", [], "should_escalate", ResponseMode.HANDOFF),
        ]

        for code, action, missing_slots, expected_flag, expected_mode in scenarios:
            with self.subTest(code=code):
                tool_result = {
                    "success": False,
                    "code": code,
                    "message": f"logistics failed: {code}",
                    "data": {
                        "provider": "kuaidi100",
                        "success": False,
                        "error_code": code,
                        "error_message": f"logistics failed: {code}",
                        "retryable": action == "retry_later",
                        "suggested_action": action,
                        "missing_slots": missing_slots,
                    },
                    "missing_slots": missing_slots,
                }
                observation = _tool_observation("step-1", "query_logistics_snapshot_tool", tool_result, False)
                state = initial_state("verify-session")
                state["current_plan"] = [
                    PlanStep(
                        step_id="step-1",
                        owner_agent=SpecialistName.ORDER,
                        action_type=ActionType.QUERY_LOGISTICS,
                        goal="query logistics",
                        success_condition="done",
                        fallback_action="handoff",
                    )
                ]
                state["current_step_index"] = 0
                state["last_observation"] = observation
                if code == "500":
                    state["retry_count_by_stage"] = {}
                else:
                    state["retry_count_by_stage"] = {"step-1:query_logistics_snapshot_tool": 1}

                verification = verify_state(state)
                self.assertTrue(getattr(verification, expected_flag))
                self.assertEqual(verification.recommended_response_mode, expected_mode)
                self.assertTrue(any(flag.startswith("logistics_failure:") for flag in verification.guardrail_flags))


if __name__ == "__main__":
    unittest.main()
