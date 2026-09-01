"""Local logistics orchestration: cache plus deterministic simulator."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from .logistics_cache import FileLogisticsCache
from .logistics_simulator import LogisticsSimulator
from .logistics_types import LogisticsRequest, error_result, execution_result_to_tool_response, success_result


def _select_simulator(mode: Optional[str] = None) -> LogisticsSimulator:
    selected = (mode or os.getenv("LOGISTICS_MODE") or "SIMULATED").strip().upper()
    if selected != "SIMULATED":
        raise ValueError("LOGISTICS_MODE must be SIMULATED")
    return LogisticsSimulator()


def query_logistics_snapshot_simulated(*, carrier_code: str, tracking_no: str, phone_last4: Optional[str] = None, ship_from: str = "", ship_to: str = "", mode: Optional[str] = None) -> Dict[str, Any]:
    request = LogisticsRequest(carrier_code=carrier_code, tracking_no=tracking_no, phone_last4=phone_last4, ship_from=ship_from, ship_to=ship_to)
    cache = FileLogisticsCache()
    lookup = cache.lookup(request.carrier_code, request.tracking_no)
    cache_meta = cache.cache_meta(lookup)
    if lookup.cache_hit and lookup.snapshot:
        snapshot = dict(lookup.snapshot)
        snapshot["source"] = "cache"
        return execution_result_to_tool_response(success_result(source="cache", message="cached logistics snapshot ready", snapshot=snapshot, cache_meta=cache_meta))
    if lookup.should_throttle_query:
        return execution_result_to_tool_response(error_result(source="cache_guard", error_code="QUERY_TOO_FREQUENT", error_message="simulator query is rate limited", retryable=True, suggested_action="retry_later", cache_meta=cache_meta))
    try:
        simulator = _select_simulator(mode)
    except ValueError as exc:
        return execution_result_to_tool_response(error_result(source="logistics_simulator", error_code="SIMULATOR_CONFIG_INVALID", error_message=str(exc), retryable=False, suggested_action="stop", cache_meta=cache_meta))
    result = simulator.query(request)
    result.cache_meta = cache_meta
    if result.success and result.snapshot:
        path = cache.save_success(cache_key=lookup.cache_key, snapshot=result.snapshot, raw_payload=None, source_name=simulator.source_name)
        result.snapshot = dict(result.snapshot)
        result.snapshot["raw_payload_ref"] = path
        result.cache_meta = {**cache_meta, "cache_hit": False, "last_query_at": result.snapshot.get("fetched_at", "")}
    return execution_result_to_tool_response(result)
