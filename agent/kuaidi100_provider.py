from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

from .logistics_types import (
    LogisticsProviderResult,
    LogisticsRequest,
    error_result,
    now_timestamp,
    success_result,
)


KUAIDI100_QUERY_URL = "https://poll.kuaidi100.com/poll/query.do"
KUAIDI100_SUCCESS_CODE = "200"

SIGNED_CODES = {"3", "301", "302", "303", "304"}
DELIVERING_CODES = {"5", "501"}
IN_TRANSIT_CODES = {"0", "1", "8", "10", "11", "12", "101", "102", "103", "1001", "1002", "1003"}
RETURNING_CODES = {"4", "6", "7", "14", "203", "401"}
ABNORMAL_CODES = {"2", "13", "201", "202", "204", "205", "206", "207", "208", "209", "210"}

ERROR_ACTIONS = {
    "400": (False, "ask_user"),
    "408": (False, "ask_user"),
    "500": (True, "retry_later"),
    "501": (True, "handoff"),
    "502": (True, "handoff"),
    "503": (False, "handoff"),
    "601": (False, "handoff"),
}


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _sign_payload(param_str: str, key: str, customer: str) -> str:
    digest = hashlib.md5()
    digest.update(f"{param_str}{key}{customer}".encode("utf-8"))
    return digest.hexdigest().upper()


def _state_name_from_delivery_state(delivery_state: str) -> str:
    mapping = {
        "signed": "签收",
        "received": "签收",
        "delivering": "派件中",
        "in_transit": "在途",
        "returning": "退回中",
        "reject": "拒签",
        "abnormal": "异常",
        "not_found": "查无结果",
        "unknown": "状态未知",
    }
    return mapping.get(delivery_state, "状态未知")


def _normalize_delivery_state(top_state: str, latest_status_code: str) -> str:
    candidate = _clean_text(latest_status_code) or _clean_text(top_state)
    if candidate in SIGNED_CODES:
        return "signed"
    if candidate in DELIVERING_CODES:
        return "delivering"
    if candidate in RETURNING_CODES:
        if candidate == "14":
            return "reject"
        return "returning"
    if candidate in ABNORMAL_CODES:
        return "abnormal"
    if candidate in IN_TRANSIT_CODES:
        return "in_transit"
    return "unknown"


def _normalize_predicted_route(items: Any) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    if not isinstance(items, list):
        return normalized
    for item in items:
        if not isinstance(item, dict):
            continue
        normalized.append(
            {
                "arrive_time": _clean_text(item.get("arriveTime")),
                "leave_time": _clean_text(item.get("leaveTime")),
                "province": _clean_text(item.get("province")),
                "city": _clean_text(item.get("city")),
                "district": _clean_text(item.get("district")),
                "name": _clean_text(item.get("name")),
                "state": _clean_text(item.get("state")),
                "type": _clean_text(item.get("type")),
            }
        )
    return normalized


def _normalize_route_info(route_info: Any) -> Dict[str, Dict[str, str]]:
    if not isinstance(route_info, dict):
        return {}

    def _normalize_node(value: Any) -> Dict[str, str]:
        if not isinstance(value, dict):
            return {"number": "", "name": ""}
        return {
            "number": _clean_text(value.get("number")),
            "name": _clean_text(value.get("name")),
        }

    return {
        "from": _normalize_node(route_info.get("from")),
        "cur": _normalize_node(route_info.get("cur")),
        "to": _normalize_node(route_info.get("to")),
    }


def _normalize_snapshot(payload: Dict[str, Any], request: LogisticsRequest) -> Dict[str, Any]:
    entries = payload.get("data")
    latest = entries[0] if isinstance(entries, list) and entries else {}
    latest_status_code = _clean_text(latest.get("statusCode"))
    normalized_state = _normalize_delivery_state(_clean_text(payload.get("state")), latest_status_code)
    route_info = _normalize_route_info(payload.get("routeInfo"))
    delivery_state_name = _clean_text(latest.get("status")) or _state_name_from_delivery_state(normalized_state)
    current_location = _clean_text(latest.get("location")) or _clean_text(route_info.get("cur", {}).get("name"))

    return {
        "carrier_code": _clean_text(payload.get("com")) or request.carrier_code,
        "tracking_no": _clean_text(payload.get("nu")) or request.tracking_no,
        "delivery_state": normalized_state,
        "delivery_state_name": delivery_state_name,
        "delivery_status_code": latest_status_code or _clean_text(payload.get("state")),
        "last_event": _clean_text(latest.get("context")),
        "last_event_time": _clean_text(latest.get("ftime")) or _clean_text(latest.get("time")),
        "current_location": current_location,
        "route_from": _clean_text(route_info.get("from", {}).get("name")) or request.ship_from,
        "route_to": _clean_text(route_info.get("to", {}).get("name")) or request.ship_to,
        "is_signed": normalized_state in {"signed", "received"},
        "is_returning": normalized_state in {"returning", "reject"},
        "is_abnormal": normalized_state in {"abnormal", "returning", "reject", "unknown"},
        "route_info": route_info,
        "arrival_time": _clean_text(payload.get("arrivalTime")),
        "predicted_route": _normalize_predicted_route(payload.get("predictedRoute")),
        "source": "kuaidi100_api",
        "fetched_at": now_timestamp(),
        "raw_payload_ref": "",
    }


class Kuaidi100Provider:
    provider_name = "kuaidi100"

    def __init__(
        self,
        *,
        customer: Optional[str] = None,
        key: Optional[str] = None,
        query_url: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        resultv2: Optional[str] = None,
        sign_type: Optional[str] = None,
    ) -> None:
        self.customer = _clean_text(customer or os.getenv("KUAIDI100_CUSTOMER"))
        self.key = _clean_text(key or os.getenv("KUAIDI100_KEY"))
        self.query_url = _clean_text(query_url or os.getenv("KUAIDI100_API_URL")) or KUAIDI100_QUERY_URL
        self.timeout_seconds = float(timeout_seconds or os.getenv("LOGISTICS_TIMEOUT_SECONDS") or 10)
        self.resultv2 = _clean_text(resultv2 or os.getenv("LOGISTICS_QUERY_RESULTV2")) or "4"
        self.sign_type = _clean_text(sign_type or os.getenv("KUAIDI100_SIGN_TYPE"))

    def is_configured(self) -> bool:
        return bool(self.customer and self.key)

    def query(self, request: LogisticsRequest) -> LogisticsProviderResult:
        if not self.is_configured():
            return error_result(
                provider=self.provider_name,
                error_code="KUAIDI100_CONFIG_MISSING",
                error_message="Missing KUAIDI100_CUSTOMER or KUAIDI100_KEY.",
                retryable=False,
                suggested_action="handoff",
            )

        param = {
            "com": request.carrier_code,
            "num": request.tracking_no,
            "phone": request.phone_last4 or "",
            "from": request.ship_from or "",
            "to": request.ship_to or "",
            "resultv2": request.resultv2 or self.resultv2,
            "show": request.show or "0",
            "order": request.order or "desc",
            "lang": request.lang or "zh",
        }
        if not param["phone"]:
            param.pop("phone", None)
        if not param["from"]:
            param.pop("from", None)
        if not param["to"]:
            param.pop("to", None)

        param_str = json.dumps(param, ensure_ascii=False, separators=(",", ":"))
        request_body = {
            "customer": self.customer,
            "param": param_str,
            "sign": _sign_payload(param_str, self.key, self.customer),
        }
        if self.sign_type:
            request_body["signType"] = self.sign_type

        encoded_body = urllib.parse.urlencode(request_body).encode("utf-8")
        http_request = urllib.request.Request(
            self.query_url,
            data=encoded_body,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(http_request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            return error_result(
                provider=self.provider_name,
                error_code="KUAIDI100_HTTP_ERROR",
                error_message=f"Kuaidi100 HTTP error: {exc.code}",
                retryable=True,
                suggested_action="retry_later",
                raw_payload={"http_status": exc.code},
            )
        except urllib.error.URLError as exc:
            return error_result(
                provider=self.provider_name,
                error_code="KUAIDI100_NETWORK_ERROR",
                error_message=f"Kuaidi100 network error: {exc.reason}",
                retryable=True,
                suggested_action="retry_later",
            )
        except json.JSONDecodeError:
            return error_result(
                provider=self.provider_name,
                error_code="KUAIDI100_PARSE_ERROR",
                error_message="Kuaidi100 response was not valid JSON.",
                retryable=True,
                suggested_action="handoff",
            )
        except Exception as exc:
            return error_result(
                provider=self.provider_name,
                error_code="KUAIDI100_UNKNOWN_ERROR",
                error_message=f"Kuaidi100 query failed: {exc}",
                retryable=True,
                suggested_action="retry_later",
            )

        success_code = _clean_text(payload.get("status"))
        if success_code == KUAIDI100_SUCCESS_CODE:
            snapshot = _normalize_snapshot(payload, request)
            return success_result(
                provider=self.provider_name,
                message=f"logistics snapshot ready: {snapshot['delivery_state_name']}",
                snapshot=snapshot,
                raw_payload=payload,
            )

        error_code = _clean_text(payload.get("returnCode")) or success_code or "KUAIDI100_ERROR"
        error_message = _clean_text(payload.get("message")) or "Kuaidi100 query failed."
        retryable, suggested_action = ERROR_ACTIONS.get(error_code, (True, "handoff"))
        missing_slots = ["phone_last4"] if error_code == "408" else []
        return error_result(
            provider=self.provider_name,
            error_code=error_code,
            error_message=error_message,
            retryable=retryable,
            suggested_action=suggested_action,
            missing_slots=missing_slots,
            raw_payload=payload,
        )
