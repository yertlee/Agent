from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol


TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"


def now_timestamp() -> str:
    return datetime.now().strftime(TIMESTAMP_FORMAT)


def parse_timestamp(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.strptime(value, TIMESTAMP_FORMAT)
    except ValueError:
        return None


@dataclass
class LogisticsRequest:
    carrier_code: str
    tracking_no: str
    phone_last4: Optional[str] = None
    ship_from: str = ""
    ship_to: str = ""
    resultv2: str = ""
    order: str = "desc"
    lang: str = "zh"
    show: str = "0"


@dataclass
class LogisticsProviderResult:
    success: bool
    code: str
    message: str
    provider_name: str
    snapshot: Optional[Dict[str, Any]] = None
    raw_payload: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    cache_meta: Dict[str, Any] = field(default_factory=dict)
    missing_slots: List[str] = field(default_factory=list)
    retryable: bool = False


class LogisticsProvider(Protocol):
    provider_name: str

    def query(self, request: LogisticsRequest) -> LogisticsProviderResult:
        ...


def make_provider_error(
    *,
    provider: str,
    error_code: str,
    error_message: str,
    retryable: bool,
    suggested_action: str,
    missing_slots: Optional[List[str]] = None,
    raw_payload_ref: str = "",
) -> Dict[str, Any]:
    return {
        "provider": provider,
        "success": False,
        "error_code": str(error_code),
        "error_message": str(error_message),
        "retryable": bool(retryable),
        "suggested_action": str(suggested_action),
        "missing_slots": list(missing_slots or []),
        "raw_payload_ref": str(raw_payload_ref or ""),
    }


def error_result(
    *,
    provider: str,
    error_code: str,
    error_message: str,
    retryable: bool,
    suggested_action: str,
    missing_slots: Optional[List[str]] = None,
    raw_payload: Optional[Dict[str, Any]] = None,
    cache_meta: Optional[Dict[str, Any]] = None,
) -> LogisticsProviderResult:
    return LogisticsProviderResult(
        success=False,
        code=str(error_code),
        message=str(error_message),
        provider_name=provider,
        raw_payload=raw_payload if isinstance(raw_payload, dict) else None,
        error=make_provider_error(
            provider=provider,
            error_code=error_code,
            error_message=error_message,
            retryable=retryable,
            suggested_action=suggested_action,
            missing_slots=missing_slots,
        ),
        cache_meta=dict(cache_meta or {}),
        missing_slots=list(missing_slots or []),
        retryable=bool(retryable),
    )


def success_result(
    *,
    provider: str,
    message: str,
    snapshot: Dict[str, Any],
    raw_payload: Optional[Dict[str, Any]] = None,
    cache_meta: Optional[Dict[str, Any]] = None,
) -> LogisticsProviderResult:
    return LogisticsProviderResult(
        success=True,
        code="OK",
        message=message,
        provider_name=provider,
        snapshot=dict(snapshot or {}),
        raw_payload=raw_payload if isinstance(raw_payload, dict) else None,
        cache_meta=dict(cache_meta or {}),
        retryable=False,
    )


def provider_result_to_tool_response(result: LogisticsProviderResult) -> Dict[str, Any]:
    if result.success:
        data = dict(result.snapshot or {})
        if result.cache_meta:
            data["_cache_meta"] = dict(result.cache_meta)
        return {
            "success": True,
            "code": result.code or "OK",
            "message": result.message,
            "data": data,
            "user_hint": "",
            "retryable": False,
            "missing_slots": [],
        }

    data = dict(result.error or {})
    if result.cache_meta:
        data["_cache_meta"] = dict(result.cache_meta)
    return {
        "success": False,
        "code": result.code,
        "message": result.message,
        "data": data,
        "user_hint": str(data.get("suggested_action") or ""),
        "retryable": bool(result.retryable),
        "missing_slots": list(result.missing_slots),
    }
