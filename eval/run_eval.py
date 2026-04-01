from __future__ import annotations

import json
import os
import shutil
import tempfile
import traceback
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import sys

from langsmith import Client
from langsmith.evaluation import evaluate, run_evaluator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import agent.specialists as specialists
from eval.cases_demo import DEMO_CASES_V3
from eval.cases_extended import EXTENDED_AGENT_CASES_V3
from agent.graph_agent import get_interrupt_payload, get_runtime_snapshot, invoke_turn_v3
from agent.state import ActionType, IntentType


REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_JSON = REPORT_DIR / "agent_v3_eval.json"
REPORT_MD = REPORT_DIR / "agent_v3_eval.md"
DSET_NAME = "agent_v3_demo_dataset"
EXTENDED_DATASET_NAME = "agent_v3_extended_eval_dataset"
DB_PATH_ENV = "ECOMMERCE_DB_PATH"
EVAL_LOGISTICS_MODE_ENV = "AGENT_EVAL_LOGISTICS_MODE"
DEFAULT_DB_PATH = PROJECT_ROOT / "ecommerce.db"

ALL_CASES_V3 = DEMO_CASES_V3 + EXTENDED_AGENT_CASES_V3


@contextmanager
def isolated_db():
    import agent.tools as tools

    db_path = os.environ.get(DB_PATH_ENV)
    if not db_path and DEFAULT_DB_PATH.exists():
        db_path = str(DEFAULT_DB_PATH)
    if not db_path:
        raise RuntimeError(
            "本仓库不包含 sqlite 数据库文件。\n"
            f"运行 eval 时请设置环境变量 {DB_PATH_ENV} 指向你本地的 ecommerce.db。\n"
            "TODO: 后续可将 order/aftersales provider 替换为真实 API。"
        )

    temp_dir = tempfile.mkdtemp(prefix="agent_v3_eval_")
    temp_db = Path(temp_dir) / Path(db_path).name
    shutil.copy2(db_path, temp_db)
    old_path = tools.DB_PATH
    tools.DB_PATH = str(temp_db)
    try:
        yield str(temp_db)
    finally:
        tools.DB_PATH = old_path
        shutil.rmtree(temp_dir, ignore_errors=True)


def _ensure_report_dir() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def _first_route_from_output(output: Dict[str, Any]) -> str:
    return str(output.get("first_route") or "")


def _obs_attr(obs: Any, key: str, default: Any = None) -> Any:
    if obs is None:
        return default
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _source_value(obs: Any) -> str:
    source_type = _obs_attr(obs, "source_type")
    return str(getattr(source_type, "value", source_type) or "")


def _extract_observations(state: Dict[str, Any]) -> List[Any]:
    return list(state.get("observations") or [])


def _extract_tool_signals(observations: List[Any]) -> Tuple[List[str], List[str]]:
    tool_names: List[str] = []
    tool_codes: List[str] = []
    for obs in observations:
        if _source_value(obs) != "tool":
            continue
        tool_names.append(str(_obs_attr(obs, "source_name") or ""))
        tool_codes.append(str(_obs_attr(obs, "code") or ""))
    return tool_names, tool_codes


def _extract_judgement_codes(observations: List[Any]) -> List[str]:
    codes: List[str] = []
    for obs in observations:
        if _source_value(obs) == "judgement":
            codes.append(str(_obs_attr(obs, "code") or ""))
    return codes


GENERIC_JUDGEMENT_CODES = {
    "ORDER_CONTEXT_READY",
    "AFTERSALES_INTENT_READY",
    "LOGISTICS_NEED_DECIDED",
    "LOGISTICS_SLOTS_READY",
    "AFTERSALES_ELIGIBILITY_READY",
    "AFTERSALES_RESULT_READY",
    "POLICY_EVIDENCE_READY",
    "POLICY_RETRY_REWRITE",
}


def _last_business_code(observations: List[Any]) -> str:
    for obs in reversed(observations):
        code = str(_obs_attr(obs, "code") or "")
        if not code:
            continue
        source = _source_value(obs)
        if source == "tool":
            return code
        if source == "judgement" and code not in GENERIC_JUDGEMENT_CODES:
            return code
    for obs in reversed(observations):
        code = str(_obs_attr(obs, "code") or "")
        if code:
            return code
    return ""


def _extract_latest_logistics_meta(observations: List[Any], state: Dict[str, Any]) -> Tuple[Optional[bool], str]:
    for obs in reversed(observations):
        if _source_value(obs) != "tool":
            continue
        if str(_obs_attr(obs, "source_name") or "") != "query_logistics_snapshot_tool":
            continue
        data = _obs_attr(obs, "structured_data") or {}
        cache_meta = data.get("_cache_meta") if isinstance(data, dict) else {}
        cache_hit = cache_meta.get("cache_hit") if isinstance(cache_meta, dict) else None
        source = str(data.get("source") or "")
        return cache_hit if isinstance(cache_hit, bool) else None, source

    cache_meta_state = state.get("logistics_cache_meta") or {}
    cache_hit_state = cache_meta_state.get("cache_hit") if isinstance(cache_meta_state, dict) else None
    snapshot = state.get("logistics_snapshot")
    source_state = str(getattr(snapshot, "source", "") or "")
    return cache_hit_state if isinstance(cache_hit_state, bool) else None, source_state


def _extract_latest_ticket_status(state: Dict[str, Any]) -> str:
    context = state.get("aftersales_context")
    if context is None:
        return ""
    return str(getattr(context, "aftersales_status", "") or "")


def _extract_response_text(state: Dict[str, Any], error_traceback: str) -> str:
    final_response = state.get("final_response") or state.get("pending_question")
    if final_response:
        return str(final_response)
    if error_traceback:
        return "EVAL_ERROR\n" + error_traceback
    return ""


def _normalize_route(value: Any) -> str:
    return str(value or "").strip().lower()


def _derive_top_route(first_route_raw: str, state: Dict[str, Any]) -> str:
    normalized = _normalize_route(first_route_raw)
    if normalized in {"order", "policy", "escalation", "unknown"}:
        return normalized

    intent = _normalize_route(getattr(state.get("intent_type"), "value", state.get("intent_type")))
    if intent == IntentType.MIXED.value:
        return "order"
    if intent in {IntentType.ORDER.value, IntentType.POLICY.value, IntentType.ESCALATION.value, IntentType.UNKNOWN.value}:
        return intent
    return "unknown"


def _derive_business_route(state: Dict[str, Any], top_route: str, response_mode: str) -> str:
    escalation_decision = str(state.get("escalation_decision") or "").upper()
    if escalation_decision == "HANDOFF_HUMAN":
        return "handoff"

    if response_mode == "handoff" or state.get("handoff_reason"):
        return "handoff"

    if top_route == "escalation":
        return "escalation"
    if top_route == "policy":
        return "policy"

    order_action = _normalize_route(getattr(state.get("order_action"), "value", state.get("order_action")))
    if order_action in {ActionType.CREATE_AFTERSALES.value, ActionType.QUERY_AFTERSALES.value}:
        return "aftersales"
    if order_action == ActionType.QUERY_LOGISTICS.value:
        return "logistics"
    if order_action == ActionType.QUERY_ORDER.value:
        return "order_query"

    if top_route == "order":
        return "order_query"
    if top_route == "unknown":
        return "general"
    return "general"


def _expected_top_route(expected: Dict[str, Any]) -> str:
    explicit = _normalize_route(expected.get("top_route"))
    if explicit:
        return explicit

    legacy = _normalize_route(expected.get("first_route"))
    if legacy in {"order", "policy", "escalation", "unknown"}:
        return legacy
    if legacy in {"aftersales", "logistics", "order_query"}:
        return "order"
    if legacy == "handoff":
        return "escalation"
    return ""


def _expected_business_route(expected: Dict[str, Any]) -> str:
    explicit = _normalize_route(expected.get("business_route"))
    if explicit:
        return explicit

    legacy = _normalize_route(expected.get("first_route"))
    if legacy in {"aftersales", "logistics", "order_query", "policy", "escalation", "handoff", "general"}:
        return legacy

    group = _normalize_route(expected.get("group"))
    if group.startswith("aftersales"):
        return "aftersales"
    if group == "policy":
        return "policy"
    if group == "handoff" and bool(expected.get("expected_handoff")):
        return "handoff"
    return ""


def _build_eval_stub_snapshot(
    carrier_code: str,
    tracking_no: str,
    phone_last4: str,
    delivery_state: str,
    source: str = "mock",
) -> Dict[str, Any]:
    signed = delivery_state in {"signed", "received"}
    returning = delivery_state in {"returning", "reject"}
    abnormal = delivery_state in {"abnormal", "not_found", "unknown"}
    return {
        "carrier_code": carrier_code,
        "tracking_no": tracking_no,
        "delivery_state": delivery_state,
        "delivery_state_name": delivery_state,
        "delivery_status_code": delivery_state,
        "last_event": f"stub:{delivery_state}",
        "last_event_time": "2026-03-23 10:00:00",
        "current_location": "stub_city",
        "route_from": "stub_from",
        "route_to": "stub_to",
        "is_signed": signed,
        "is_returning": returning,
        "is_abnormal": abnormal,
        "source": source,
        "fetched_at": "2026-03-23T10:00:00Z",
        "raw_payload_ref": "",
        "_cache_meta": {
            "cache_key": f"{carrier_code}:{tracking_no}:{phone_last4}",
            "cache_hit": False,
            "last_query_at": "2026-03-23T10:00:00Z",
            "ttl_minutes": 30,
        },
    }


def _eval_stub_logistics_tool(carrier_code: str, tracking_no: str, phone_last4: Optional[str] = None) -> Dict[str, Any]:
    phone = str(phone_last4 or "").strip()
    key = (str(carrier_code).strip(), str(tracking_no).strip(), phone)
    state_by_key = {
        ("yuantong", "7609205232746", "1234"): "signed",      # 20260320001
        ("zhongtong", "78986914191424", "9156"): "in_transit",  # 20260320007
        ("zhongtong", "78986914191424", "6403"): "delivering",  # 20260320009
        ("zhongtong", "78986914191424", "4517"): "signed",      # 20260320018
    }
    delivery_state = state_by_key.get(key)
    if delivery_state is None:
        # 未命中映射时不再兜底为 signed，避免掩盖物流标识配置错误。
        fallback_states = ("not_found", "unknown", "abnormal")
        stable_idx = abs(hash("|".join(key))) % len(fallback_states)
        delivery_state = fallback_states[stable_idx]
    data = _build_eval_stub_snapshot(key[0], key[1], key[2], delivery_state, source="mock")
    return {
        "success": True,
        "code": "OK",
        "message": "stub logistics snapshot ready",
        "data": data,
        "user_hint": "",
    }


@contextmanager
def eval_logistics_mode():
    mode = str(os.environ.get(EVAL_LOGISTICS_MODE_ENV, "stub") or "stub").strip().lower()
    if mode == "real":
        yield
        return

    original = specialists.TOOL_REGISTRY["query_logistics_snapshot_tool"].callable
    specialists.TOOL_REGISTRY["query_logistics_snapshot_tool"].callable = _eval_stub_logistics_tool
    try:
        yield
    finally:
        specialists.TOOL_REGISTRY["query_logistics_snapshot_tool"].callable = original


def _case_to_example(case: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "inputs": {
            "name": case["name"],
            "turns": case["turns"],
        },
        "outputs": case.get("expected", {}),
        "metadata": {"case_name": case["name"]},
    }


def ensure_langsmith_dataset(client: Client, cases: Iterable[Dict[str, Any]], dataset_name: str = DSET_NAME) -> str:
    existing = next(client.list_datasets(dataset_name=dataset_name, limit=1), None)
    if existing is None:
        client.create_dataset(
            dataset_name=dataset_name,
            description="Curated ecommerce customer service graph-agent demo cases",
        )
        client.create_examples(dataset_name=dataset_name, examples=[_case_to_example(case) for case in cases])
    return dataset_name


def run_case(inputs: Dict[str, Any]) -> Dict[str, Any]:
    turns = list(inputs.get("turns") or [])
    session_id = f"eval_{uuid.uuid4().hex[:10]}"
    clarification_slots_seen: List[List[str]] = []
    first_route_raw = ""
    error_traceback = ""
    aggregate_observations: List[Any] = []
    aggregate_rag_hits = 0
    with isolated_db(), eval_logistics_mode():
        last_state: Optional[Dict[str, Any]] = None
        try:
            for turn in turns:
                last_state = invoke_turn_v3(session_id, turn)
                if not first_route_raw:
                    plan = last_state.get("current_plan") or []
                    if plan:
                        first_route_raw = plan[0].owner_agent.value
                for obs in _extract_observations(last_state):
                    if _source_value(obs) == "user_clarification" and _obs_attr(obs, "missing_slots"):
                        clarification_slots_seen.append(list(_obs_attr(obs, "missing_slots") or []))
                aggregate_observations.extend(_extract_observations(last_state))
                aggregate_rag_hits += len(last_state.get("retrieval_evidence") or [])
            state = last_state or {}
        except Exception:
            error_traceback = traceback.format_exc()
            state = last_state or {}

    snapshot = get_runtime_snapshot(session_id)
    verification = state.get("verification_status")
    response_mode = getattr(state.get("response_mode"), "value", state.get("response_mode"))
    rag_hits = aggregate_rag_hits
    observations = aggregate_observations or _extract_observations(state)
    tool_names, tool_codes = _extract_tool_signals(observations)
    judgement_codes = _extract_judgement_codes(observations)
    tool_calls = len(tool_names)
    safe_termination = bool(snapshot and (not snapshot.next or snapshot.interrupts))
    pending_interrupt = get_interrupt_payload(session_id)
    logistics_used = "query_logistics_snapshot_tool" in tool_names
    logistics_cache_hit, logistics_source = _extract_latest_logistics_meta(observations, state)
    top_route = _derive_top_route(first_route_raw, state)
    business_route = _derive_business_route(state, top_route, str(response_mode or ""))
    handoff = business_route == "handoff" or bool(state.get("handoff_reason"))

    return {
        "case_name": inputs.get("name", ""),
        "final_response": _extract_response_text(state, error_traceback),
        "intent_type": getattr(state.get("intent_type"), "value", state.get("intent_type")),
        "plan_mode": getattr(state.get("plan_mode"), "value", state.get("plan_mode")),
        "response_mode": response_mode or ("error" if error_traceback else ""),
        "first_route": top_route,
        "first_route_raw": first_route_raw,
        "top_route": top_route,
        "business_route": business_route,
        "clarification_slots_seen": clarification_slots_seen,
        "tool_names": tool_names,
        "tool_codes": tool_codes,
        "judgement_codes": judgement_codes,
        "last_business_code": _last_business_code(observations),
        "logistics_used": logistics_used,
        "logistics_cache_hit": logistics_cache_hit,
        "logistics_source": logistics_source,
        "create_aftersales_called": "create_aftersales_tool" in tool_names,
        "query_aftersales_called": "query_aftersales_tool" in tool_names,
        "get_order_info_called": "get_order_info_tool" in tool_names,
        "eligibility": str((state.get("trace_tags") or {}).get("aftersales_eligibility") or ""),
        "requires_logistics": bool((state.get("trace_tags") or {}).get("aftersales_requires_logistics")),
        "latest_ticket_status": _extract_latest_ticket_status(state),
        "tool_calls": tool_calls,
        "replans": int(state.get("replan_count") or 0),
        "handoff": handoff,
        "rag_hits": rag_hits,
        "safe_termination": safe_termination,
        "unsupported_answer": bool(verification and verification.unsupported_answer_risk and response_mode == "finalize"),
        "tool_retry_counts": dict(state.get("tool_retry_counts") or {}),
        "repeated_failures": int(sum(1 for value in (state.get("tool_retry_counts") or {}).values() if value > 1)),
        "thread_id": session_id,
        "pending_interrupt": bool(pending_interrupt),
        "error_traceback": error_traceback,
    }


@run_evaluator
def first_route_evaluator(run, example):
    outputs = run.outputs or {}
    expected = (example.outputs if example else {}) or {}
    expected_route = _expected_top_route(expected)
    if not expected_route:
        return {"key": "correct_first_route_rate", "score": 1.0}
    score = 1.0 if _normalize_route(outputs.get("top_route") or outputs.get("first_route")) == expected_route else 0.0
    return {"key": "correct_first_route_rate", "score": score}


@run_evaluator
def business_route_evaluator(run, example):
    outputs = run.outputs or {}
    expected = (example.outputs if example else {}) or {}
    expected_route = _expected_business_route(expected)
    if not expected_route:
        return {"key": "correct_business_route_rate", "score": 1.0}
    score = 1.0 if _normalize_route(outputs.get("business_route")) == expected_route else 0.0
    return {"key": "correct_business_route_rate", "score": score}


@run_evaluator
def clarification_evaluator(run, example):
    outputs = run.outputs or {}
    expected = (example.outputs if example else {}) or {}
    target_slots = expected.get("expected_clarify_slots") or []
    if not target_slots:
        return {"key": "slot_clarification_accuracy", "score": 1.0}
    seen = outputs.get("clarification_slots_seen") or []
    normalized_seen = {tuple(sorted(item)) for item in seen}
    score = 1.0 if tuple(sorted(target_slots)) in normalized_seen else 0.0
    return {"key": "slot_clarification_accuracy", "score": score}


@run_evaluator
def handoff_evaluator(run, example):
    outputs = run.outputs or {}
    expected = (example.outputs if example else {}) or {}
    expected_handoff = expected.get("expected_handoff")
    if expected_handoff is None and ("allow_handoff" in expected):
        expected_handoff = bool(expected.get("allow_handoff"))
    if expected_handoff is None:
        return {"key": "handoff_expected", "score": 1.0}
    score = 1.0 if bool(outputs.get("handoff")) == bool(expected_handoff) else 0.0
    return {"key": "handoff_expected", "score": score}


@run_evaluator
def response_mode_evaluator(run, example):
    outputs = run.outputs or {}
    expected = (example.outputs if example else {}) or {}
    expected_mode = expected.get("expected_response_mode")
    if not expected_mode:
        return {"key": "response_mode_accuracy", "score": 1.0}
    score = 1.0 if outputs.get("response_mode") == expected_mode else 0.0
    return {"key": "response_mode_accuracy", "score": score}


@run_evaluator
def policy_hit_evaluator(run, example):
    outputs = run.outputs or {}
    expected = (example.outputs if example else {}) or {}
    expect_hits = bool(expected.get("expect_policy_hits", False))
    if not expect_hits:
        return {"key": "policy_hit_expected", "score": 1.0}
    score = 1.0 if float(outputs.get("rag_hits", 0) or 0) > 0 else 0.0
    return {"key": "policy_hit_expected", "score": score}


@run_evaluator
def unsupported_answer_evaluator(run, example):
    outputs = run.outputs or {}
    score = 0.0 if outputs.get("unsupported_answer") else 1.0
    return {"key": "unsupported_answer_safe", "score": score}


@run_evaluator
def safe_termination_evaluator(run, example):
    outputs = run.outputs or {}
    score = 1.0 if outputs.get("safe_termination") else 0.0
    return {"key": "safe_termination_expected", "score": score}


@run_evaluator
def business_code_match_evaluator(run, example):
    outputs = run.outputs or {}
    expected = (example.outputs if example else {}) or {}
    expected_code = expected.get("expected_last_business_code")
    expected_contains = list(expected.get("expected_codes_contains") or [])
    if not expected_code and not expected_contains:
        return {"key": "business_code_match_rate", "score": 1.0}
    actual = str(outputs.get("last_business_code") or "")
    score = 1.0 if (expected_code and actual == expected_code) or (expected_contains and any(code == actual for code in expected_contains)) else 0.0
    return {"key": "business_code_match_rate", "score": score}


@run_evaluator
def tool_path_match_evaluator(run, example):
    outputs = run.outputs or {}
    expected = (example.outputs if example else {}) or {}
    tool_names = list(outputs.get("tool_names") or [])
    prefix = list(expected.get("expected_tool_sequence_prefix") or [])
    contains = list(expected.get("expected_tool_sequence_contains") or [])
    if not prefix and not contains:
        return {"key": "tool_path_match_rate", "score": 1.0}

    prefix_ok = True
    contains_ok = True
    if prefix:
        prefix_ok = tool_names[: len(prefix)] == prefix
    if contains:
        idx = 0
        for name in contains:
            while idx < len(tool_names) and tool_names[idx] != name:
                idx += 1
            if idx >= len(tool_names):
                contains_ok = False
                break
            idx += 1
    return {"key": "tool_path_match_rate", "score": 1.0 if (prefix_ok and contains_ok) else 0.0}


@run_evaluator
def logistics_expected_evaluator(run, example):
    outputs = run.outputs or {}
    expected = (example.outputs if example else {}) or {}
    expects_requires = expected.get("requires_logistics")
    expects_tool = expected.get("expect_logistics_tool")
    if expects_requires is None and expects_tool is None:
        return {"key": "logistics_expected_rate", "score": 1.0}
    requires_ok = True if expects_requires is None else bool(outputs.get("requires_logistics")) == bool(expects_requires)
    tool_ok = True if expects_tool is None else bool(outputs.get("logistics_used")) == bool(expects_tool)
    return {"key": "logistics_expected_rate", "score": 1.0 if (requires_ok and tool_ok) else 0.0}


@run_evaluator
def eligibility_evaluator(run, example):
    outputs = run.outputs or {}
    expected = (example.outputs if example else {}) or {}
    expected_eligibility = expected.get("expected_eligibility")
    if not expected_eligibility:
        return {"key": "eligibility_accuracy", "score": 1.0}
    score = 1.0 if str(outputs.get("eligibility") or "") == str(expected_eligibility) else 0.0
    return {"key": "eligibility_accuracy", "score": score}


def _check_case_expectations(result: Dict[str, Any], case: Dict[str, Any]) -> List[str]:
    expected = case.get("expected", {})
    fail_reasons: List[str] = []
    expected_top_route = _expected_top_route(expected)
    expected_business_route = _expected_business_route(expected)
    if expected_top_route and _normalize_route(result.get("top_route") or result.get("first_route")) != expected_top_route:
        fail_reasons.append(f"top_route={result.get('top_route')} expected={expected_top_route}")
    if expected_business_route and _normalize_route(result.get("business_route")) != expected_business_route:
        fail_reasons.append(f"business_route={result.get('business_route')} expected={expected_business_route}")
    if expected.get("expected_response_mode") and result.get("response_mode") != expected.get("expected_response_mode"):
        fail_reasons.append(f"response_mode={result.get('response_mode')} expected={expected.get('expected_response_mode')}")
    if expected.get("expect_policy_hits") and float(result.get("rag_hits", 0) or 0) <= 0:
        fail_reasons.append("expected policy hit but rag_hits=0")
    expected_handoff = expected.get("expected_handoff")
    if expected_handoff is None and ("allow_handoff" in expected):
        expected_handoff = bool(expected.get("allow_handoff"))
    if expected_handoff is not None and bool(result.get("handoff")) != bool(expected_handoff):
        fail_reasons.append(f"handoff={result.get('handoff')} expected={expected_handoff}")
    expected_code = expected.get("expected_last_business_code")
    expected_codes_contains = list(expected.get("expected_codes_contains") or [])
    if expected_code and str(result.get("last_business_code") or "") != str(expected_code):
        fail_reasons.append(f"last_business_code={result.get('last_business_code')} expected={expected_code}")
    if expected_codes_contains and str(result.get("last_business_code") or "") not in expected_codes_contains:
        fail_reasons.append(f"last_business_code={result.get('last_business_code')} not in expected_codes_contains")
    prefix = list(expected.get("expected_tool_sequence_prefix") or [])
    contains = list(expected.get("expected_tool_sequence_contains") or [])
    tool_names = list(result.get("tool_names") or [])
    if prefix and tool_names[: len(prefix)] != prefix:
        fail_reasons.append(f"tool_prefix={tool_names[:len(prefix)]} expected={prefix}")
    if contains:
        idx = 0
        contains_ok = True
        for name in contains:
            while idx < len(tool_names) and tool_names[idx] != name:
                idx += 1
            if idx >= len(tool_names):
                contains_ok = False
                break
            idx += 1
        if not contains_ok:
            fail_reasons.append(f"tool_contains_order_mismatch expected={contains} actual={tool_names}")
    if expected.get("requires_logistics") is not None and bool(result.get("requires_logistics")) != bool(expected.get("requires_logistics")):
        fail_reasons.append(f"requires_logistics={result.get('requires_logistics')} expected={expected.get('requires_logistics')}")
    if expected.get("expect_logistics_tool") is not None and bool(result.get("logistics_used")) != bool(expected.get("expect_logistics_tool")):
        fail_reasons.append(f"logistics_used={result.get('logistics_used')} expected={expected.get('expect_logistics_tool')}")
    if expected.get("expected_eligibility") and str(result.get("eligibility") or "") != str(expected.get("expected_eligibility")):
        fail_reasons.append(f"eligibility={result.get('eligibility')} expected={expected.get('expected_eligibility')}")
    if result.get("unsupported_answer"):
        fail_reasons.append("unsupported_answer")
    if not result.get("safe_termination"):
        fail_reasons.append("not safe_termination")
    return fail_reasons


def _build_local_summary(results: List[Dict[str, Any]], cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    def mean(values: List[float]) -> float:
        return round(sum(values) / max(len(values), 1), 4)

    route_scores = []
    business_route_scores = []
    clarify_scores = []
    task_scores = []
    rag_scores = []
    handoff_scores = []  # raw handoff rate (handoff==True)
    handoff_expected_scores = []  # per-case expectation-aware
    safe_scores = []
    response_mode_expected_scores = []
    policy_hit_expected_scores = []
    business_code_scores = []
    tool_path_scores = []
    logistics_required_scores = []
    logistics_tool_scores = []
    eligibility_scores = []
    case_pass_scores = []
    business_outcome_scores = []
    per_group: Dict[str, Dict[str, float]] = {}
    top_failed_cases: List[Dict[str, Any]] = []

    for result, case in zip(results, cases):
        expected = case.get("expected", {})
        group = str(expected.get("group") or "ungrouped")
        expected_top_route = _expected_top_route(expected)
        expected_business_route = _expected_business_route(expected)
        route_scores.append(1.0 if not expected_top_route or _normalize_route(result.get("top_route") or result.get("first_route")) == expected_top_route else 0.0)
        if expected_business_route:
            business_route_scores.append(1.0 if _normalize_route(result.get("business_route")) == expected_business_route else 0.0)

        clarify_expected = expected.get("expected_clarify_slots") or []
        if not clarify_expected:
            clarify_scores.append(1.0)
        else:
            seen = {tuple(sorted(item)) for item in result["clarification_slots_seen"]}
            clarify_scores.append(1.0 if tuple(sorted(clarify_expected)) in seen else 0.0)

        has_output = bool(str(result.get("final_response") or "").strip())
        no_runtime_error = not bool(str(result.get("error_traceback") or "").strip())
        task_scores.append(1.0 if (result["safe_termination"] and has_output and no_runtime_error) else 0.0)

        expected_mode = expected.get("expected_response_mode")
        if expected_mode:
            response_mode_expected_scores.append(1.0 if result["response_mode"] == expected_mode else 0.0)

        if expected.get("expect_policy_hits"):
            rag_scores.append(1.0 if result["rag_hits"] > 0 else 0.0)
            policy_hit_expected_scores.append(1.0 if result["rag_hits"] > 0 else 0.0)

        handoff_scores.append(1.0 if result["handoff"] else 0.0)
        expected_handoff = expected.get("expected_handoff")
        if expected_handoff is None and ("allow_handoff" in expected):
            expected_handoff = bool(expected.get("allow_handoff"))
        handoff_expected_scores.append(
            1.0 if expected_handoff is None else (1.0 if bool(result["handoff"]) == bool(expected_handoff) else 0.0)
        )
        safe_scores.append(1.0 if result["safe_termination"] else 0.0)

        expected_code = expected.get("expected_last_business_code")
        expected_codes_contains = list(expected.get("expected_codes_contains") or [])
        expected_mode_for_outcome = expected.get("expected_response_mode")
        if expected_code or expected_codes_contains:
            actual_code = str(result.get("last_business_code") or "")
            business_code_scores.append(
                1.0 if (expected_code and actual_code == expected_code) or (expected_codes_contains and actual_code in expected_codes_contains) else 0.0
            )
        outcome_checks: List[bool] = []
        if expected_code:
            outcome_checks.append(str(result.get("last_business_code") or "") == str(expected_code))
        if expected_codes_contains:
            outcome_checks.append(str(result.get("last_business_code") or "") in expected_codes_contains)
        if expected_mode_for_outcome:
            outcome_checks.append(str(result.get("response_mode") or "") == str(expected_mode_for_outcome))
        if outcome_checks:
            business_outcome_scores.append(1.0 if all(outcome_checks) else 0.0)

        prefix = list(expected.get("expected_tool_sequence_prefix") or [])
        contains = list(expected.get("expected_tool_sequence_contains") or [])
        if prefix or contains:
            names = list(result.get("tool_names") or [])
            prefix_ok = (not prefix) or names[: len(prefix)] == prefix
            if not contains:
                contains_ok = True
            else:
                next_idx = 0
                contains_ok = True
                for name in contains:
                    while next_idx < len(names) and names[next_idx] != name:
                        next_idx += 1
                    if next_idx >= len(names):
                        contains_ok = False
                        break
                    next_idx += 1
            tool_path_scores.append(1.0 if (prefix_ok and contains_ok) else 0.0)

        if expected.get("requires_logistics") is not None:
            logistics_required_scores.append(1.0 if bool(result.get("requires_logistics")) == bool(expected.get("requires_logistics")) else 0.0)
        if expected.get("expect_logistics_tool") is not None:
            logistics_tool_scores.append(1.0 if bool(result.get("logistics_used")) == bool(expected.get("expect_logistics_tool")) else 0.0)
        if expected.get("expected_eligibility"):
            eligibility_scores.append(1.0 if str(result.get("eligibility") or "") == str(expected.get("expected_eligibility")) else 0.0)

        fail_reasons = _check_case_expectations(result, case)
        case_pass = not fail_reasons
        case_pass_scores.append(1.0 if case_pass else 0.0)
        result["case_pass"] = case_pass
        result["fail_reasons"] = fail_reasons
        result["group"] = group

        stat = per_group.setdefault(group, {"total": 0.0, "pass": 0.0})
        stat["total"] += 1.0
        stat["pass"] += 1.0 if case_pass else 0.0
        if not case_pass:
            top_failed_cases.append(
                {
                    "case_name": result["case_name"],
                    "group": group,
                    "fail_reasons": fail_reasons,
                    "top_route": result.get("top_route", ""),
                    "business_route": result.get("business_route", ""),
                    "tool_path": list(result.get("tool_names") or []),
                    "last_business_code": result.get("last_business_code", ""),
                    "response_mode": result.get("response_mode", ""),
                }
            )

    per_group_summary = {
        g: {
            "total_cases": int(v["total"]),
            "pass_cases": int(v["pass"]),
            "case_pass_rate": round(v["pass"] / max(v["total"], 1.0), 4),
        }
        for g, v in per_group.items()
    }
    top_failed_cases = sorted(top_failed_cases, key=lambda x: len(x.get("fail_reasons") or []), reverse=True)[:10]
    summary = {
        "total_cases": len(results),
        "task_completion_rate": mean(task_scores),
        "case_pass_rate": mean(case_pass_scores),
        "business_outcome_accuracy": mean(business_outcome_scores) if business_outcome_scores else 0.0,
        "correct_first_route_rate": mean(route_scores),
        "correct_top_route_rate": mean(route_scores),
        "correct_business_route_rate": mean(business_route_scores) if business_route_scores else 0.0,
        "slot_clarification_accuracy": mean(clarify_scores),
        "response_mode_accuracy": mean(response_mode_expected_scores) if response_mode_expected_scores else 0.0,
        "policy_hit_expected_rate": mean(policy_hit_expected_scores) if policy_hit_expected_scores else 0.0,
        "handoff_expected_rate": mean(handoff_expected_scores) if handoff_expected_scores else 0.0,
        "business_code_match_rate": mean(business_code_scores) if business_code_scores else 0.0,
        "tool_path_match_rate": mean(tool_path_scores) if tool_path_scores else 0.0,
        "logistics_required_accuracy": mean(logistics_required_scores) if logistics_required_scores else 0.0,
        "logistics_tool_expected_rate": mean(logistics_tool_scores) if logistics_tool_scores else 0.0,
        "eligibility_accuracy": mean(eligibility_scores) if eligibility_scores else 0.0,
        "unsupported_answer_rate": mean([1.0 if item["unsupported_answer"] else 0.0 for item in results]),
        "avg_tool_calls_per_case": mean([float(item["tool_calls"]) for item in results]),
        "avg_replans_per_case": mean([float(item["replans"]) for item in results]),
        "handoff_rate": mean(handoff_scores),
        "rag_hit_rate": mean(rag_scores) if rag_scores else 0.0,
        "safe_termination_rate": mean(safe_scores),
        "repeated_failure_count": int(sum(item["repeated_failures"] for item in results)),
        "per_group_summary": per_group_summary,
        "top_failed_cases": top_failed_cases,
        "results": results,
    }
    return summary


def _write_markdown(summary: Dict[str, Any]) -> None:
    lines = [
        "# Agent V3 Eval Summary",
        "",
        f"- Total cases: {summary['total_cases']}",
        f"- task_completion_rate: {summary['task_completion_rate']}",
        f"- correct_top_route_rate: {summary.get('correct_top_route_rate', summary['correct_first_route_rate'])}",
        f"- correct_business_route_rate: {summary.get('correct_business_route_rate', 0.0)}",
        f"- slot_clarification_accuracy: {summary['slot_clarification_accuracy']}",
        f"- response_mode_accuracy: {summary.get('response_mode_accuracy', 0.0)}",
        f"- policy_hit_expected_rate: {summary.get('policy_hit_expected_rate', 0.0)}",
        f"- handoff_expected_rate: {summary.get('handoff_expected_rate', 0.0)}",
        f"- case_pass_rate: {summary.get('case_pass_rate', 0.0)}",
        f"- business_outcome_accuracy: {summary.get('business_outcome_accuracy', 0.0)}",
        f"- business_code_match_rate: {summary.get('business_code_match_rate', 0.0)}",
        f"- tool_path_match_rate: {summary.get('tool_path_match_rate', 0.0)}",
        f"- logistics_required_accuracy: {summary.get('logistics_required_accuracy', 0.0)}",
        f"- logistics_tool_expected_rate: {summary.get('logistics_tool_expected_rate', 0.0)}",
        f"- eligibility_accuracy: {summary.get('eligibility_accuracy', 0.0)}",
        f"- unsupported_answer_rate: {summary['unsupported_answer_rate']}",
        f"- avg_tool_calls_per_case: {summary['avg_tool_calls_per_case']}",
        f"- avg_replans_per_case: {summary['avg_replans_per_case']}",
        f"- handoff_rate: {summary['handoff_rate']}",
        f"- rag_hit_rate: {summary['rag_hit_rate']}",
        f"- safe_termination_rate: {summary['safe_termination_rate']}",
        "",
        "## Group Breakdown",
        "",
    ]
    for group, stats in (summary.get("per_group_summary") or {}).items():
        lines.append(
            f"- {group}: total={stats.get('total_cases', 0)}, pass={stats.get('pass_cases', 0)}, case_pass_rate={stats.get('case_pass_rate', 0.0)}"
        )

    lines.extend(
        [
            "",
            "## Top Failed Cases",
            "",
        ]
    )
    for item in summary.get("top_failed_cases") or []:
        lines.append(
            f"- {item.get('case_name')} | group={item.get('group')} | top_route={item.get('top_route')} | business_route={item.get('business_route')} | response_mode={item.get('response_mode')} | "
            f"last_business_code={item.get('last_business_code')} | tool_path={item.get('tool_path')} | reasons={item.get('fail_reasons')}"
        )

    lines.extend(
        [
            "",
        "## Case Details",
        "",
        ]
    )
    for item in summary["results"]:
        final_resp = str(item.get("final_response") or "")
        if len(final_resp) > 160:
            final_resp = final_resp[:160] + "..."
        lines.extend(
            [
                f"### {item['case_name']}",
                f"- group: {item.get('group', '')}",
                f"- top_route: {item.get('top_route', item['first_route'])}",
                f"- business_route: {item.get('business_route', '')}",
                f"- first_route_raw: {item.get('first_route_raw', '')}",
                f"- response_mode: {item['response_mode']}",
                f"- tool_names: {item.get('tool_names', [])}",
                f"- tool_codes: {item.get('tool_codes', [])}",
                f"- last_business_code: {item.get('last_business_code', '')}",
                f"- requires_logistics: {item.get('requires_logistics')}",
                f"- logistics_used: {item.get('logistics_used')}",
                f"- logistics_source: {item.get('logistics_source')}",
                f"- logistics_cache_hit: {item.get('logistics_cache_hit')}",
                f"- eligibility: {item.get('eligibility', '')}",
                f"- handoff: {item['handoff']}",
                f"- pending_interrupt: {item.get('pending_interrupt')}",
                f"- final_response: {final_resp}",
                f"- case_pass: {item.get('case_pass')}",
                f"- fail_reasons: {item.get('fail_reasons', [])}",
                f"- thread_id: {item['thread_id']}",
                "",
            ]
        )
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def run_local_eval(cases: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    cases = cases or ALL_CASES_V3
    results = [run_case({"name": case["name"], "turns": case["turns"]}) for case in cases]
    summary = _build_local_summary(results, cases)
    _ensure_report_dir()
    REPORT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(summary)
    return summary


def run_local_eval_extended() -> Dict[str, Any]:
    # 在离线/受限网络环境下，强制 HuggingFace 走离线模式，避免长时间重试卡住本地 eval。
    old_hf_offline = os.environ.get("HF_HUB_OFFLINE")
    old_tf_offline = os.environ.get("TRANSFORMERS_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        return run_local_eval(cases=ALL_CASES_V3)
    finally:
        if old_hf_offline is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = old_hf_offline
        if old_tf_offline is None:
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
        else:
            os.environ["TRANSFORMERS_OFFLINE"] = old_tf_offline


def run_langsmith_eval(
    upload_results: bool = True,
    experiment_prefix: str = "agent_v3",
    cases: Optional[List[Dict[str, Any]]] = None,
    dataset_name: str = EXTENDED_DATASET_NAME,
) -> Any:
    client = Client()
    cases = cases or ALL_CASES_V3
    dataset_name = ensure_langsmith_dataset(client, cases, dataset_name=dataset_name)
    return evaluate(
        run_case,
        data=dataset_name,
        evaluators=[
            first_route_evaluator,
            business_route_evaluator,
            clarification_evaluator,
            handoff_evaluator,
            response_mode_evaluator,
            policy_hit_evaluator,
            business_code_match_evaluator,
            tool_path_match_evaluator,
            logistics_expected_evaluator,
            eligibility_evaluator,
            unsupported_answer_evaluator,
            safe_termination_evaluator,
        ],
        experiment_prefix=experiment_prefix,
        description="LangGraph-native ecommerce customer service agent evaluation",
        metadata={"system": "agent_v3"},
        max_concurrency=1,
        client=client,
        upload_results=upload_results,
    )


def run_langsmith_eval_extended(upload_results: bool = True, experiment_prefix: str = "agent_v3_extended") -> Any:
    return run_langsmith_eval(
        upload_results=upload_results,
        experiment_prefix=experiment_prefix,
        cases=ALL_CASES_V3,
        dataset_name=EXTENDED_DATASET_NAME,
    )


if __name__ == "__main__":
    summary = run_local_eval_extended()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
