"""Independent executable evaluation for the corrected R2 DAG runtime."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from agent.interactive_runtime import ChatOpenAIStructuredProvider, LLMRuntimeConfig, ModelResult
from agent.r1_5_router import BusinessIntent, CustomerGoalV1
from agent.r2_logistics_repository import order_ref_hash, seed_rows
from agent.r2_order_logistics import DagCandidateV1, DagNodeV1, R2OrderLogisticsRuntime
from eval.harness.contracts import ExecutionMode


def _rows(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _goal(raw: Mapping[str, Any]) -> CustomerGoalV1:
    kind = BusinessIntent(str(raw["goal_type"]))
    caps = {
        BusinessIntent.ORDER_QUERY: ("order/read@v1",),
        BusinessIntent.LOGISTICS_QUERY: ("logistics/read@v1",),
        BusinessIntent.ORDER_AND_LOGISTICS: ("order/read@v1", "logistics/read@v1"),
    }[kind]
    return CustomerGoalV1(goal_type=kind, entities=raw.get("entities", {}), required_capabilities=caps)


def _candidate(kind: BusinessIntent) -> DagCandidateV1:
    if kind is BusinessIntent.ORDER_QUERY:
        nodes = (DagNodeV1(node_key="order", capability_ref="order/read@v1"),)
    elif kind is BusinessIntent.LOGISTICS_QUERY:
        nodes = (DagNodeV1(node_key="logistics", capability_ref="logistics/read@v1"),)
    else:
        nodes = (
            DagNodeV1(node_key="order", capability_ref="order/read@v1"),
            DagNodeV1(node_key="logistics", capability_ref="logistics/read@v1", depends_on=("order",),
                      input_sources={"carrier_code": "order.payload.carrier_code", "tracking_no": "order.payload.tracking_no"}),
        )
    return DagCandidateV1(nodes=nodes)


class GoalAwareCandidateProvider:
    """Deterministic contract smoke; it is never reported as model quality."""
    test_only = True
    def __call__(self, prompt: str, schema: type[Any]) -> Any:
        if schema is not DagCandidateV1:
            raise ValueError("unexpected schema")
        kind = next((item for item in (BusinessIntent.ORDER_QUERY, BusinessIntent.LOGISTICS_QUERY, BusinessIntent.ORDER_AND_LOGISTICS)
                     if f'"goal_type": "{item.value}"' in prompt), None)
        if kind is None:
            raise ValueError("goal_type missing")
        return ModelResult(_candidate(kind))


class FixedOrderLogisticsProvider:
    test_only = True
    def __call__(self, _prompt: str, schema: type[Any]) -> Any:
        return ModelResult(_candidate(BusinessIntent.ORDER_AND_LOGISTICS))


class AllToolsProvider:
    test_only = True
    def __call__(self, _prompt: str, schema: type[Any]) -> Any:
        return ModelResult(DagCandidateV1(nodes=(
            DagNodeV1(node_key="order", capability_ref="order/read@v1"),
            DagNodeV1(node_key="logistics", capability_ref="logistics/read@v1"),
        )))


class PolicyLookupProvider:
    test_only = True
    def __call__(self, _prompt: str, schema: type[Any]) -> Any:
        return ModelResult(DagCandidateV1(nodes=(DagNodeV1(node_key="policy", capability_ref="policy/read@v1"),)))


def _world(case: Mapping[str, Any], root: Path) -> tuple[Path, Path, CustomerGoalV1]:
    goal = _goal(case["goal"]); entities = goal.entity_map; world = case["world"]
    order_id = entities.get("order_id") or f"R2ORD-{case['case_id']}"
    owner_phone = str(world.get("owner_phone_last4") or entities.get("phone_last4") or "0000")
    carrier = entities.get("carrier_code") or "carrier_eval"
    tracking = entities.get("tracking_no") or f"R2TRK-{hashlib.sha256(case['case_id'].encode()).hexdigest()[:10]}"
    order_db, logistics_db = root / "orders.db", root / "logistics.db"
    conn = sqlite3.connect(order_db)
    conn.execute("CREATE TABLE orders (order_id TEXT PRIMARY KEY, phone_last4 TEXT NOT NULL, order_status TEXT, pay_status TEXT, shipment_status TEXT, created_at TEXT, shipped_at TEXT, delivered_at TEXT, carrier_code TEXT, tracking_no TEXT)")
    conn.execute("INSERT INTO orders VALUES (?,?,?,?,?,?,?,?,?,?)", (order_id, owner_phone, "PAID", "PAID", "IN_TRANSIT", "2026-01-01", "2026-01-02", "", carrier, tracking))
    conn.commit(); conn.close()
    quality = str(world.get("logistics", "FRESH"))
    if quality != "MISSING":
        seed_rows(logistics_db, [{"order_ref_hash": order_ref_hash(order_id), "carrier_code": carrier, "tracking_no": tracking,
            "delivery_state": "IN_TRANSIT", "shipment_status": "IN_TRANSIT", "observed_at": "2026-01-03T00:00:00Z",
            "data_quality": quality, "events": [{"event_code": "PICKED_UP", "event_time": "2026-01-02T00:00:00Z"}]}])
    else:
        seed_rows(logistics_db, [])
    return order_db, logistics_db, goal


def _topology(result: Any) -> tuple[list[str], list[list[str]]]:
    if not result.plan_revisions:
        return [], []
    tasks = result.plan_revisions[0].tasks
    names = {task.task_id: ("order" if task.agent_ref == "order-agent@v1" else "logistics" if task.agent_ref == "logistics-agent@v1" else task.agent_ref) for task in tasks}
    nodes = [names[task.task_id] for task in tasks]
    edges = [[names[dep], names[task.task_id]] for task in tasks for dep in task.depends_on]
    return nodes, edges


def _run_case(case: Mapping[str, Any], provider: Any, root: Path, *, allow_replan: bool) -> dict[str, Any]:
    case_root = root / str(case["case_id"]); case_root.mkdir(parents=True, exist_ok=True)
    order_db, logistics_db, goal = _world(case, case_root)
    fault = {"logistics/query@v1": {"code": "INFRA_TIMEOUT"}} if case["world"].get("fault") else None
    is_external = isinstance(provider, ChatOpenAIStructuredProvider)
    mode = ExecutionMode.FAULT if fault else (ExecutionMode.LIVE if is_external else ExecutionMode.SIMULATED)
    try:
        runtime = R2OrderLogisticsRuntime(db_path=order_db, logistics_db_path=logistics_db, provider=provider,
            artifact_root=case_root / "artifacts", mode=mode, failure_script=fault, allow_replan=allow_replan)
        result = runtime.run(user_id="r2-eval-principal", goal=goal)
        nodes, edges = _topology(result)
        return {"case_id": case["case_id"], "nodes": nodes, "edges": edges, "terminal": result.status, "code": result.code,
                "replan_count": result.replan_count, "bundle_verified": bool(result.bundle_verification.get("ok")),
                "model_calls": result.model_calls, "tool_calls": result.tool_calls, "error": None}
    except Exception as exc:
        return {"case_id": case["case_id"], "nodes": [], "edges": [], "terminal": "FAILED", "code": getattr(exc, "code", "EVALUATION_EXCEPTION"),
                "replan_count": 0, "bundle_verified": False, "model_calls": 0, "tool_calls": 0, "error": type(exc).__name__}


def _score(cases: Sequence[Mapping[str, Any]], observations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_id = {row["case_id"]: row for row in observations}; counters = {key: 0 for key in ("topology", "terminal", "code", "replan", "bundle", "case")}
    failures = []
    for case in cases:
        expected, row = case["expected"], by_id[case["case_id"]]
        checks = {
            "topology": row["nodes"] == expected["nodes"] and row["edges"] == expected["edges"],
            "terminal": row["terminal"] == expected["terminal"], "code": row["code"] == expected["code"],
            "replan": row["replan_count"] == int(expected.get("replan_count", 0)), "bundle": row["bundle_verified"],
        }
        for key, passed in checks.items(): counters[key] += int(passed)
        checks["case"] = all(checks.values()); counters["case"] += int(checks["case"])
        if not checks["case"]: failures.append({"case_id": case["case_id"], "failed_checks": [k for k, v in checks.items() if not v]})
    total = max(1, len(cases))
    return {"N": len(cases), **{f"{key}_accuracy": value / total for key, value in counters.items()}, "failed": failures}


def evaluate(dataset: str | Path, output: str | Path, artifact_root: str | Path, *, external_target: bool = False) -> dict[str, Any]:
    cases = _rows(dataset)
    artifact_parent = Path(artifact_root); artifact_parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="run-", dir=str(artifact_parent)))
    systems = {
        "contract_smoke": (GoalAwareCandidateProvider(), True, "deterministic_contract_smoke"),
        "fixed_order_logistics": (FixedOrderLogisticsProvider(), True, "executable_baseline"),
        "all_tools": (AllToolsProvider(), True, "executable_baseline"),
        "policy_lookup": (PolicyLookupProvider(), True, "executable_baseline"),
        "no_replan": (GoalAwareCandidateProvider(), False, "executable_baseline"),
    }
    config = None
    if external_target:
        config = LLMRuntimeConfig.from_environment(db_path="ecommerce.db")
        systems["external_planner"] = (ChatOpenAIStructuredProvider(config), True, "external_structured_model")
    results = {}
    for name, (provider, allow_replan, run_class) in systems.items():
        observations = [_run_case(case, provider, root / name, allow_replan=allow_replan) for case in cases]
        results[name] = {"run_class": run_class, "metrics": _score(cases, observations), "observations": observations}
    smoke_score = results["contract_smoke"]["metrics"]["case_accuracy"]
    external_score = results.get("external_planner", {}).get("metrics", {}).get("case_accuracy")
    if external_score is not None:
        status = "SUSPICIOUS_PERFECT" if external_score == 1.0 else "EXTERNAL_DEV_MEASURED"
    else:
        status = "ENGINEERING_PASS" if smoke_score == 1.0 else "ENGINEERING_FAIL"
    report = {"report_version": "r2.correction.eval.v1", "dataset": str(dataset), "dataset_sha256": hashlib.sha256(Path(dataset).read_bytes()).hexdigest(),
              "N": len(cases), "systems": results, "config_evidence": config.evidence() if config else None,
              "run_namespace": root.name,
              "status": status,
              "quality_claim": "NOT_VALIDATED" if not external_target else "EXTERNAL_DEV_ONLY"}
    target = Path(output); target.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    fd, tmp = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle: handle.write(encoded); handle.flush(); os.fsync(handle.fileno())
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--dataset", default="eval/datasets/r2_correction/dev-cases.jsonl")
    parser.add_argument("--output", default="artifacts/r2_correction/eval-report.json"); parser.add_argument("--artifact-root", default="artifacts/r2_correction/runs")
    parser.add_argument("--external-target", action="store_true"); parser.add_argument("--local-env-bootstrap", action="store_true")
    args = parser.parse_args(argv)
    if args.local_env_bootstrap: import agent.llm  # noqa: F401
    report = evaluate(args.dataset, args.output, args.artifact_root, external_target=args.external_target)
    print(json.dumps({"N": report["N"], "status": report["status"], "quality_claim": report["quality_claim"], "case_accuracy": {k: v["metrics"]["case_accuracy"] for k, v in report["systems"].items()}}, sort_keys=True))
    return 0


if __name__ == "__main__": raise SystemExit(main())
