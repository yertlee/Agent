"""Versioned deterministic logistics simulator.

The simulator is local-only and accepts an optional failure script keyed by
tracking number. It never performs network I/O and derives default scenarios
from stable input text, making tests and replay runs reproducible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping

from .logistics_types import LogisticsExecutionResult, LogisticsRequest, error_result, success_result

SIMULATOR_VERSION = "logistics-simulator-v1"


def _profile(tracking_no: str) -> str:
    normalized = tracking_no.strip().lower()
    for marker, bucket in (("signed", "signed"), ("delivered", "signed"), ("abnormal", "abnormal"), ("exception", "abnormal"), ("missing", "not_found"), ("delivering", "delivering"), ("dispatch", "delivering"), ("transit", "in_transit"), ("route", "in_transit")):
        if marker in normalized:
            return bucket
    match = re.search(r"(\d)$", tracking_no)
    digit = int(match.group(1)) if match else 0
    return "signed" if digit in {0, 1, 2} else "delivering" if digit in {3, 4} else "in_transit" if digit in {5, 6, 7} else "abnormal" if digit == 8 else "not_found"


@dataclass
class LogisticsSimulator:
    """Deterministic simulator with declarative, injectable failures."""

    failure_script: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    scene_clock: str = "2026-03-20T00:00:00Z"
    version: str = SIMULATOR_VERSION
    source_name: str = "logistics_simulator"

    def query(self, request: LogisticsRequest) -> LogisticsExecutionResult:
        fault = self.failure_script.get(request.tracking_no)
        if fault:
            return error_result(source=self.source_name, error_code=str(fault.get("code") or "SIMULATOR_FAULT"), error_message=str(fault.get("message") or "simulator fault"), retryable=bool(fault.get("retryable", False)), suggested_action=str(fault.get("action") or "ask_user"))
        bucket = _profile(request.tracking_no)
        labels = {"signed": "签收", "delivering": "派送中", "in_transit": "运输中", "abnormal": "异常", "not_found": "未知"}
        events = {"signed": "包裹已签收", "delivering": "包裹正在派送", "in_transit": "包裹正在运输", "abnormal": "物流状态异常", "not_found": "暂无物流轨迹"}
        snapshot = {"carrier_code": request.carrier_code, "tracking_no": request.tracking_no, "delivery_state": bucket, "delivery_state_name": labels[bucket], "delivery_status_code": bucket, "last_event": events[bucket], "last_event_time": self.scene_clock, "current_location": "模拟站点" if bucket != "not_found" else "", "route_from": "模拟发货地", "route_to": "模拟收货地", "is_signed": bucket == "signed", "is_returning": False, "is_abnormal": bucket in {"abnormal", "not_found"}, "source": self.source_name, "simulator_version": self.version, "scene_clock": self.scene_clock, "raw_payload_ref": "", "fetched_at": self.scene_clock}
        if request.phone_last4:
            snapshot["phone_last4"] = request.phone_last4
        return success_result(source=self.source_name, message=f"simulated logistics snapshot ready: {snapshot['delivery_state_name']}", snapshot=snapshot, raw_payload=None)
