from __future__ import annotations

import json
import os
import shutil
import tempfile
import traceback
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import sys

from langsmith import Client
from langsmith.evaluation import evaluate, run_evaluator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from eval.cases_demo import DEMO_CASES_V3
from eval.cases_extended import EXTENDED_AGENT_CASES_V3
from agent.graph_agent import get_interrupt_payload, get_runtime_snapshot, invoke_turn_v3


REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_JSON = REPORT_DIR / "agent_v3_eval.json"
REPORT_MD = REPORT_DIR / "agent_v3_eval.md"
DSET_NAME = "agent_v3_demo_dataset"
EXTENDED_DATASET_NAME = "agent_v3_extended_eval_dataset"
DB_PATH_ENV = "ECOMMERCE_DB_PATH"

ALL_CASES_V3 = DEMO_CASES_V3 + EXTENDED_AGENT_CASES_V3


@contextmanager
def isolated_db():
    import agent.tools as tools

    db_path = os.environ.get(DB_PATH_ENV)
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
    first_route = ""
    error_traceback = ""
    with isolated_db():
        last_state: Optional[Dict[str, Any]] = None
        try:
            for turn in turns:
                last_state = invoke_turn_v3(session_id, turn)
                if not first_route:
                    plan = last_state.get("current_plan") or []
                    if plan:
                        first_route = plan[0].owner_agent.value
                for obs in last_state.get("observations") or []:
                    if obs.source_type.value == "user_clarification" and obs.missing_slots:
                        clarification_slots_seen.append(obs.missing_slots)
            state = last_state or {}
        except Exception:
            error_traceback = traceback.format_exc()
            state = last_state or {}

    snapshot = get_runtime_snapshot(session_id)
    verification = state.get("verification_status")
    response_mode = getattr(state.get("response_mode"), "value", state.get("response_mode"))
    rag_hits = len(state.get("retrieval_evidence") or [])
    tool_calls = sum(1 for obs in (state.get("observations") or []) if obs.source_type.value == "tool")
    safe_termination = bool(snapshot and (not snapshot.next or snapshot.interrupts))
    pending_interrupt = get_interrupt_payload(session_id)

    return {
        "case_name": inputs.get("name", ""),
        "final_response": state.get("final_response") or state.get("pending_question") or (("EVAL_ERROR\n" + error_traceback) if error_traceback else ""),
        "intent_type": getattr(state.get("intent_type"), "value", state.get("intent_type")),
        "plan_mode": getattr(state.get("plan_mode"), "value", state.get("plan_mode")),
        "response_mode": response_mode or ("error" if error_traceback else ""),
        "first_route": first_route,
        "clarification_slots_seen": clarification_slots_seen,
        "tool_calls": tool_calls,
        "replans": int(state.get("replan_count") or 0),
        "handoff": bool(state.get("handoff_reason") or response_mode == "handoff"),
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
    expected_route = expected.get("first_route")
    if not expected_route:
        return {"key": "correct_first_route_rate", "score": 1.0}
    score = 1.0 if outputs.get("first_route") == expected_route else 0.0
    return {"key": "correct_first_route_rate", "score": score}


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
    allow_handoff = bool(expected.get("allow_handoff", False))
    score = 1.0 if (outputs.get("handoff") == allow_handoff or allow_handoff) else 0.0
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


def _build_local_summary(results: List[Dict[str, Any]], cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results) or 1

    def mean(values: List[float]) -> float:
        return round(sum(values) / max(len(values), 1), 4)

    route_scores = []
    clarify_scores = []
    task_scores = []
    rag_scores = []
    handoff_scores = []  # raw handoff rate (handoff==True)
    handoff_expected_scores = []  # per-case expectation-aware
    safe_scores = []
    response_mode_expected_scores = []
    policy_hit_expected_scores = []

    for result, case in zip(results, cases):
        expected = case.get("expected", {})
        expected_route = expected.get("first_route")
        route_scores.append(1.0 if not expected_route or result["first_route"] == expected_route else 0.0)

        clarify_expected = expected.get("expected_clarify_slots") or []
        if not clarify_expected:
            clarify_scores.append(1.0)
        else:
            seen = {tuple(sorted(item)) for item in result["clarification_slots_seen"]}
            clarify_scores.append(1.0 if tuple(sorted(clarify_expected)) in seen else 0.0)

        expected_mode = expected.get("expected_response_mode")
        if expected_mode:
            task_scores.append(1.0 if result["response_mode"] == expected_mode else 0.0)
            response_mode_expected_scores.append(1.0 if result["response_mode"] == expected_mode else 0.0)
        else:
            task_scores.append(1.0 if result["final_response"] else 0.0)

        if expected.get("expect_policy_hits"):
            rag_scores.append(1.0 if result["rag_hits"] > 0 else 0.0)
            policy_hit_expected_scores.append(1.0 if result["rag_hits"] > 0 else 0.0)

        handoff_scores.append(1.0 if result["handoff"] else 0.0)
        allow_handoff = bool(expected.get("allow_handoff", False))
        handoff_expected_scores.append(1.0 if (allow_handoff or (not result["handoff"])) else 0.0)
        safe_scores.append(1.0 if result["safe_termination"] else 0.0)

    summary = {
        "total_cases": len(results),
        "task_completion_rate": mean(task_scores),
        "correct_first_route_rate": mean(route_scores),
        "slot_clarification_accuracy": mean(clarify_scores),
        "response_mode_accuracy": mean(response_mode_expected_scores) if response_mode_expected_scores else 0.0,
        "policy_hit_expected_rate": mean(policy_hit_expected_scores) if policy_hit_expected_scores else 0.0,
        "handoff_expected_rate": mean(handoff_expected_scores) if handoff_expected_scores else 0.0,
        "unsupported_answer_rate": mean([1.0 if item["unsupported_answer"] else 0.0 for item in results]),
        "avg_tool_calls_per_case": mean([float(item["tool_calls"]) for item in results]),
        "avg_replans_per_case": mean([float(item["replans"]) for item in results]),
        "handoff_rate": mean(handoff_scores),
        "rag_hit_rate": mean(rag_scores) if rag_scores else 0.0,
        "safe_termination_rate": mean(safe_scores),
        "repeated_failure_count": int(sum(item["repeated_failures"] for item in results)),
        "results": results,
    }
    return summary


def _write_markdown(summary: Dict[str, Any]) -> None:
    lines = [
        "# Agent V3 Eval Summary",
        "",
        f"- Total cases: {summary['total_cases']}",
        f"- task_completion_rate: {summary['task_completion_rate']}",
        f"- correct_first_route_rate: {summary['correct_first_route_rate']}",
        f"- slot_clarification_accuracy: {summary['slot_clarification_accuracy']}",
        f"- response_mode_accuracy: {summary.get('response_mode_accuracy', 0.0)}",
        f"- policy_hit_expected_rate: {summary.get('policy_hit_expected_rate', 0.0)}",
        f"- handoff_expected_rate: {summary.get('handoff_expected_rate', 0.0)}",
        f"- unsupported_answer_rate: {summary['unsupported_answer_rate']}",
        f"- avg_tool_calls_per_case: {summary['avg_tool_calls_per_case']}",
        f"- avg_replans_per_case: {summary['avg_replans_per_case']}",
        f"- handoff_rate: {summary['handoff_rate']}",
        f"- rag_hit_rate: {summary['rag_hit_rate']}",
        f"- safe_termination_rate: {summary['safe_termination_rate']}",
        "",
        "## Case Details",
        "",
    ]
    for item in summary["results"]:
        final_resp = str(item.get("final_response") or "")
        if len(final_resp) > 120:
            final_resp = final_resp[:120] + "..."
        lines.extend(
            [
                f"### {item['case_name']}",
                f"- first_route: {item['first_route']}",
                f"- response_mode: {item['response_mode']}",
                f"- tool_calls: {item['tool_calls']}",
                f"- replans: {item['replans']}",
                f"- handoff: {item['handoff']}",
                f"- rag_hits: {item['rag_hits']}",
                f"- safe_termination: {item['safe_termination']}",
                f"- unsupported_answer: {item.get('unsupported_answer')}",
                f"- pending_interrupt: {item.get('pending_interrupt')}",
                f"- final_response: {final_resp}",
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
            clarification_evaluator,
            handoff_evaluator,
            response_mode_evaluator,
            policy_hit_evaluator,
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
