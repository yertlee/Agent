from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from eval.cases_demo import DEMO_CASES_V3
from eval.cases_extended import EXTENDED_AGENT_CASES_V3


REPORT_PATH = PROJECT_ROOT / "reports" / "agent_v3_eval.json"


def _load_report() -> Dict[str, Any]:
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def _check_case(case: Dict[str, Any], out: Dict[str, Any]) -> List[str]:
    exp = (case.get("expected") or {}) if isinstance(case, dict) else {}
    fails: List[str] = []

    if exp.get("first_route") and out.get("first_route") != exp.get("first_route"):
        fails.append(f"first_route {out.get('first_route')} != {exp.get('first_route')}")

    if exp.get("expected_response_mode") and out.get("response_mode") != exp.get("expected_response_mode"):
        fails.append(f"response_mode {out.get('response_mode')} != {exp.get('expected_response_mode')}")

    if exp.get("expect_policy_hits") and float(out.get("rag_hits", 0) or 0) <= 0:
        fails.append("policy_hit_expected but rag_hits==0")

    expected_handoff = exp.get("expected_handoff")
    if expected_handoff is None and ("allow_handoff" in exp):
        expected_handoff = bool(exp.get("allow_handoff"))
    if expected_handoff is not None and bool(out.get("handoff")) != bool(expected_handoff):
        fails.append(f"handoff {out.get('handoff')} != {expected_handoff}")

    if exp.get("expected_last_business_code") and out.get("last_business_code") != exp.get("expected_last_business_code"):
        fails.append(f"last_business_code {out.get('last_business_code')} != {exp.get('expected_last_business_code')}")

    if exp.get("expected_eligibility") and out.get("eligibility") != exp.get("expected_eligibility"):
        fails.append(f"eligibility {out.get('eligibility')} != {exp.get('expected_eligibility')}")

    if out.get("unsupported_answer"):
        fails.append("unsupported_answer")

    if not out.get("safe_termination"):
        fails.append("not safe_termination")

    return fails


def main() -> None:
    report = _load_report()
    cases = DEMO_CASES_V3 + EXTENDED_AGENT_CASES_V3
    results = {r.get("case_name"): r for r in (report.get("results") or [])}

    keys = [
        "total_cases",
        "task_completion_rate",
        "correct_first_route_rate",
        "slot_clarification_accuracy",
        "response_mode_accuracy",
        "policy_hit_expected_rate",
        "case_pass_rate",
        "business_code_match_rate",
        "tool_path_match_rate",
        "logistics_required_accuracy",
        "logistics_tool_expected_rate",
        "eligibility_accuracy",
        "unsupported_answer_rate",
        "safe_termination_rate",
    ]
    print("== summary ==")
    for k in keys:
        print(k, "=", report.get(k))

    bad: List[Tuple[str, List[str]]] = []
    for case in cases:
        name = str(case.get("name"))
        out = results.get(name) or {}
        fails = _check_case(case, out)
        if fails:
            bad.append((name, fails))

    print("\n== failures ==")
    print("bad_count =", len(bad))
    for name, fails in bad[:20]:
        print("-", name, "=>", "; ".join(fails))


if __name__ == "__main__":
    main()

