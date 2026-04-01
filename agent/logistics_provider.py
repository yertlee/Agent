from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Any, Dict, Optional

from .kuaidi100_provider import Kuaidi100Provider
from .logistics_cache import FileLogisticsCacheProvider
from .logistics_types import (
    LogisticsProvider,
    LogisticsProviderResult,
    LogisticsRequest,
    error_result,
    now_timestamp,
    provider_result_to_tool_response,
    success_result,
)


def _mock_logistics_profile(carrier_code: str, tracking_no: str) -> Dict[str, Any]:
    normalized = tracking_no.strip().lower()
    if "signed" in normalized or "delivered" in normalized:
        bucket = "signed"
    elif "abnormal" in normalized or "exception" in normalized:
        bucket = "abnormal"
    elif "notfound" in normalized or "missing" in normalized:
        bucket = "not_found"
    elif "delivering" in normalized or "dispatch" in normalized:
        bucket = "delivering"
    elif "transit" in normalized or "route" in normalized:
        bucket = "in_transit"
    else:
        match = re.search(r"(\d)$", tracking_no)
        last_digit = int(match.group(1)) if match else 0
        if last_digit in {0, 1, 2}:
            bucket = "signed"
        elif last_digit in {3, 4}:
            bucket = "delivering"
        elif last_digit in {5, 6, 7}:
            bucket = "in_transit"
        elif last_digit == 8:
            bucket = "abnormal"
        else:
            bucket = "not_found"

    base = {
        "carrier_code": carrier_code,
        "tracking_no": tracking_no,
        "route_from": "广东省深圳市南山区",
        "route_to": "江苏省苏州市昆山市",
        "route_info": {
            "from": {"number": "CN440305000000", "name": "广东,深圳市,南山区"},
            "cur": {"number": "CN320583000000", "name": "江苏,苏州市,昆山市"},
            "to": {"number": "CN320583000000", "name": "江苏,苏州市,昆山市"},
        },
        "arrival_time": "",
        "predicted_route": [],
        "source": "mock_provider",
        "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "raw_payload_ref": f"mock://logistics/{carrier_code}/{tracking_no}",
    }

    profiles = {
        "signed": {
            "delivery_state": "signed",
            "delivery_state_name": "签收",
            "delivery_status_code": "3",
            "last_event": "包裹已由本人签收",
            "last_event_time": "2026-03-20 12:30:00",
            "current_location": "苏州市,昆山市",
            "is_signed": True,
            "is_returning": False,
            "is_abnormal": False,
        },
        "delivering": {
            "delivery_state": "delivering",
            "delivery_state_name": "派件中",
            "delivery_status_code": "5",
            "last_event": "快件正在派送途中",
            "last_event_time": "2026-03-20 09:15:00",
            "current_location": "苏州市,昆山市",
            "is_signed": False,
            "is_returning": False,
            "is_abnormal": False,
        },
        "in_transit": {
            "delivery_state": "in_transit",
            "delivery_state_name": "在途",
            "delivery_status_code": "0",
            "last_event": "快件已离开上一站，正在运输中",
            "last_event_time": "2026-03-20 06:20:00",
            "current_location": "无锡市,新吴区",
            "is_signed": False,
            "is_returning": False,
            "is_abnormal": False,
        },
        "abnormal": {
            "delivery_state": "abnormal",
            "delivery_state_name": "异常",
            "delivery_status_code": "204",
            "last_event": "派件异常，等待人工处理",
            "last_event_time": "2026-03-20 08:05:00",
            "current_location": "苏州市,昆山市",
            "is_signed": False,
            "is_returning": False,
            "is_abnormal": True,
        },
        "not_found": {
            "delivery_state": "not_found",
            "delivery_state_name": "查不到",
            "delivery_status_code": "",
            "last_event": "暂未查询到物流轨迹",
            "last_event_time": "",
            "current_location": "",
            "is_signed": False,
            "is_returning": False,
            "is_abnormal": True,
        },
    }
    return {**base, **profiles[bucket]}


class MockLogisticsProvider:
    provider_name = "mock_provider"

    def query(self, request: LogisticsRequest) -> LogisticsProviderResult:
        snapshot = _mock_logistics_profile(request.carrier_code, request.tracking_no)
        if request.phone_last4:
            snapshot["phone_last4"] = request.phone_last4
        return success_result(
            provider=self.provider_name,
            message=f"mock logistics snapshot ready: {snapshot['delivery_state_name']}",
            snapshot=snapshot,
            raw_payload={"provider": self.provider_name, "snapshot": dict(snapshot)},
        )


def _provider_mode() -> str:
    return (os.getenv("LOGISTICS_PROVIDER_MODE") or "auto").strip().lower()


def _select_provider(mode: Optional[str] = None) -> LogisticsProvider:
    selected_mode = (mode or _provider_mode()).strip().lower()
    kuaidi100 = Kuaidi100Provider()
    if selected_mode == "mock":
        return MockLogisticsProvider()
    if selected_mode == "kuaidi100":
        return kuaidi100
    if selected_mode == "auto" and kuaidi100.is_configured():
        return kuaidi100
    return MockLogisticsProvider()


def query_logistics_snapshot_with_fallback(
    *,
    carrier_code: str,
    tracking_no: str,
    phone_last4: Optional[str] = None,
    ship_from: str = "",
    ship_to: str = "",
    provider_mode: Optional[str] = None,
) -> Dict[str, Any]:
    request = LogisticsRequest(
        carrier_code=carrier_code,
        tracking_no=tracking_no,
        phone_last4=phone_last4,
        ship_from=ship_from,
        ship_to=ship_to,
    )
    cache = FileLogisticsCacheProvider()
    cache_lookup = cache.lookup(request.carrier_code, request.tracking_no)
    cache_meta = cache.cache_meta(cache_lookup)

    if cache_lookup.cache_hit and cache_lookup.snapshot:
        cached_snapshot = dict(cache_lookup.snapshot)
        cached_snapshot["source"] = "cache"
        result = success_result(
            provider="cache",
            message=f"cached logistics snapshot ready: {cached_snapshot.get('delivery_state_name') or cached_snapshot.get('delivery_state')}",
            snapshot=cached_snapshot,
            raw_payload=None,
            cache_meta=cache_meta,
        )
        return provider_result_to_tool_response(result)

    provider = _select_provider(provider_mode)
    if provider.provider_name == "kuaidi100" and cache_lookup.should_block_api:
        result = error_result(
            provider="cache_guard",
            error_code="QUERY_TOO_FREQUENT",
            error_message="The same tracking number was queried within the last 30 minutes. Skip API call to avoid lock risk.",
            retryable=True,
            suggested_action="retry_later",
            cache_meta=cache_meta,
        )
        return provider_result_to_tool_response(result)

    result = provider.query(request)
    result.cache_meta = cache_meta

    if result.success and result.snapshot:
        cache_path = cache.save_success(
            cache_key=cache_lookup.cache_key,
            snapshot=result.snapshot,
            raw_payload=result.raw_payload,
            provider_name=provider.provider_name,
        )
        result.snapshot = dict(result.snapshot)
        result.snapshot["raw_payload_ref"] = cache_path
        result.cache_meta = {
            **cache_meta,
            "cache_hit": False,
            "last_query_at": result.snapshot.get("fetched_at") or "",
        }
        return provider_result_to_tool_response(result)

    if provider.provider_name == "kuaidi100" and result.code != "KUAIDI100_CONFIG_MISSING":
        cache_path = cache.save_error(
            cache_key=cache_lookup.cache_key,
            provider_name=provider.provider_name,
            error=result.error or {},
            raw_payload=result.raw_payload,
        )
        error_payload = dict(result.error or {})
        error_payload["raw_payload_ref"] = cache_path
        result.error = error_payload
        result.cache_meta = {
            **cache_meta,
            "cache_hit": False,
            "last_query_at": now_timestamp(),
        }

    return provider_result_to_tool_response(result)
