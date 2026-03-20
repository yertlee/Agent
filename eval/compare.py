from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Dict, Iterator

from langsmith import Client
from langsmith.evaluation import evaluate_comparative
from langsmith.evaluation.evaluator import ComparisonEvaluationResult

import sys
from pathlib import Path as _Path

PROJECT_ROOT = _Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from eval.run_eval import ALL_CASES_V3, EXTENDED_DATASET_NAME, run_langsmith_eval


@contextmanager
def temporary_env(overrides: Dict[str, str]) -> Iterator[None]:
    old_values = {key: os.environ.get(key) for key in overrides}
    try:
        for key, value in overrides.items():
            os.environ[key] = value
        yield
    finally:
        for key, old_value in old_values.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def _comparative_preference(runs, example):
    scores = {}
    comments = {}
    for run in runs:
        outputs = run.outputs or {}
        tool_calls = float(outputs.get("tool_calls", 0))
        replans = float(outputs.get("replans", 0))
        unsupported = 10.0 if outputs.get("unsupported_answer") else 0.0
        safe_bonus = 1.0 if outputs.get("safe_termination") else -2.0
        rag_bonus = 1.0 if outputs.get("rag_hits", 0) > 0 else 0.0
        score = safe_bonus + rag_bonus - tool_calls - (0.5 * replans) - unsupported
        scores[str(run.id)] = score
        comments[str(run.id)] = (
            f"tool_calls={tool_calls}, replans={replans}, "
            f"unsupported={bool(outputs.get('unsupported_answer'))}, safe={outputs.get('safe_termination')}"
        )
    return ComparisonEvaluationResult(
        key="pairwise_runtime_preference",
        scores=scores,
        comment=comments,
    )


def compare_planner_prompts():
    with temporary_env({"AGENT_V3_PLANNER_PROMPT_VARIANT": "default", "AGENT_V3_DISABLE_QUERY_REWRITE": "0"}):
        exp_a = run_langsmith_eval(
            upload_results=True,
            experiment_prefix="agent_v3_planner_default",
            cases=ALL_CASES_V3,
            dataset_name=EXTENDED_DATASET_NAME,
        )
        exp_a.wait()

    with temporary_env({"AGENT_V3_PLANNER_PROMPT_VARIANT": "compact", "AGENT_V3_DISABLE_QUERY_REWRITE": "0"}):
        exp_b = run_langsmith_eval(
            upload_results=True,
            experiment_prefix="agent_v3_planner_compact",
            cases=ALL_CASES_V3,
            dataset_name=EXTENDED_DATASET_NAME,
        )
        exp_b.wait()

    client = Client()
    return evaluate_comparative(
        (exp_a.experiment_name, exp_b.experiment_name),
        evaluators=[_comparative_preference],
        experiment_prefix="agent_v3_compare_planner",
        description="Compare planner prompt variants",
        client=client,
    )


def compare_query_rewrite():
    with temporary_env({"AGENT_V3_PLANNER_PROMPT_VARIANT": "default", "AGENT_V3_DISABLE_QUERY_REWRITE": "0"}):
        exp_a = run_langsmith_eval(
            upload_results=True,
            experiment_prefix="agent_v3_rewrite_on",
            cases=ALL_CASES_V3,
            dataset_name=EXTENDED_DATASET_NAME,
        )
        exp_a.wait()

    with temporary_env({"AGENT_V3_PLANNER_PROMPT_VARIANT": "default", "AGENT_V3_DISABLE_QUERY_REWRITE": "1"}):
        exp_b = run_langsmith_eval(
            upload_results=True,
            experiment_prefix="agent_v3_rewrite_off",
            cases=ALL_CASES_V3,
            dataset_name=EXTENDED_DATASET_NAME,
        )
        exp_b.wait()

    client = Client()
    return evaluate_comparative(
        (exp_a.experiment_name, exp_b.experiment_name),
        evaluators=[_comparative_preference],
        experiment_prefix="agent_v3_compare_rewrite",
        description="Compare query rewrite on vs off",
        client=client,
    )


if __name__ == "__main__":
    print("Running planner prompt comparison...")
    compare_planner_prompts()
    print("Running query rewrite comparison...")
    compare_query_rewrite()

