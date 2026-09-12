"""Gold-independent R4-B evaluator shell and executable baselines.

The evaluator accepts executable R4-B inputs and, optionally, a separately
supplied gold document.  It never discovers a gold file by convention and it
does not derive expectations from the input.  Every mode receives the same
case object and an independently materialized temporary world.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Iterable, Mapping, Sequence

from agent.domain.objects import sha256_json
from agent.m2_context import InvocationContext
from agent.r4_a2a_contracts import A2AContractError
from agent.r4_a2a_runtime import (
    ROUTES,
    R4A2ARuntime,
    R4TraceEventV1,
    project_trace_event_id,
)
from agent.r4_b_contracts import (
    R4SupervisorProviderBoundary,
)
from agent.r2_logistics_repository import order_ref_hash, seed_rows
from eval.evaluator import wilson_interval

from .r4_b_inputs import (
    NORMAL_TOPOLOGIES,
    R4BInputV1,
    build_r4_b_dev_inputs,
    input_manifest,
    load_r4_b_inputs,
)


R4_B_EVALUATOR_VERSION = "r4-b.evaluator.v1"
R4_B_MODES = (
    "a2a",
    "direct_call",
    "fixed_order",
    "single_agent",
    "no_dedup",
    "no_retry",
    "disabled_specialist",
)
_READ_CAPABILITIES = {
    "order": "order/read@v1",
    "logistics": "logistics/read@v1",
    "policy": "policy/read@v1",
}
# Exact execution plans are accepted only through the explicit execution spec
# boundary.  They are intentionally absent from R4-B input JSONL.
_TASKS_FOR_TOPOLOGY = {
    "order_only": ("order",),
    "logistics_only": ("logistics",),
    "policy_only": ("policy",),
    "order->logistics": ("order", "logistics"),
    "order+policy": ("order", "policy"),
    "order->logistics+policy": ("order", "logistics", "policy"),
}
_FIXED_TASKS = ("order", "logistics", "policy")
_VALID_EXECUTION_TOPOLOGIES = frozenset(_TASKS_FOR_TOPOLOGY)
_COORDINATION_MODES = frozenset({"a2a", "direct_call", "fixed_order", "no_dedup", "no_retry", "disabled_specialist"})


def _metric(numerator: int, denominator: int, *, case_ids: Sequence[str]) -> dict[str, Any]:
    numerator = int(numerator)
    denominator = int(denominator)
    return {
        "value": (numerator / denominator) if denominator else 0.0,
        "numerator": numerator,
        "denominator": denominator,
        "unique_case_N": len(set(case_ids)),
        "wilson95": list(wilson_interval(numerator, denominator)),
    }


def _utc_clock() -> datetime:
    # Keep the deterministic scene after the host wall clock.  The runtime
    # still records this fixed value, while verifier deadline checks remain
    # valid on every machine running the shell.
    return datetime(2030, 1, 1, 0, 0, tzinfo=timezone.utc)


def _authority() -> dict[str, Any]:
    version_tuple = (
        "r3.manifest.v1",
        "r3.test.corpus.v1",
        "r3.5.runtime.v1",
        "r4-b.synthetic-policy.v1",
    )
    evidence = (
        {
            "evidence_id": "r4b-ev-1",
            "source_id": "r4b-policy-source",
            "version": "v1",
            "chunk_id": "r4b-chunk-1",
            "text_hash": "a" * 64,
            "locator": "r4b-policy-source#1",
        },
    )
    return {
        "source": "r3.5.project-authored-kb",
        "source_version": sha256_json(list(version_tuple)),
        "version_tuple": list(version_tuple),
        "strategy_checksum": "b" * 64,
        "evidence": list(evidence),
    }


def _policy_reader(query: str) -> dict[str, Any]:
    authority = _authority()
    evidence = list(authority["evidence"])
    return {
        "success": True,
        "code": "OK",
        "data": {
            "query": str(query),
            "status": "ANSWERED",
            "source": authority["source"],
            "source_version": authority["source_version"],
            "source_version_tuple": authority["version_tuple"],
            "strategy_checksum": authority["strategy_checksum"],
            "evidence": evidence,
            "claims": [{"claim_id": "r4b-claim-1", "text": "synthetic policy evidence", "evidence_ids": ["r4b-ev-1"]}],
        },
    }


@dataclass
class _CaseWorld:
    case: R4BInputV1
    root: Path
    order_db: Path = field(init=False)
    logistics_db: Path = field(init=False)
    ledger_db: Path = field(init=False)
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.order_db = self.root / "orders.sqlite"
        self.logistics_db = self.root / "logistics.sqlite"
        self.ledger_db = self.root / "a2a-ledger.sqlite"
        world = self.case.world
        self.fingerprint = sha256_json(world.model_dump(mode="json"))
        self._create_order_db()
        seed_rows(
            self.logistics_db,
            [
                {
                    "order_ref_hash": order_ref_hash(world.order_id),
                    "carrier_code": world.carrier_code,
                    "tracking_no": world.tracking_no,
                    "delivery_state": "IN_TRANSIT",
                    "shipment_status": "IN_TRANSIT",
                    "observed_at": "2026-09-09T00:00:00Z",
                    "data_quality": world.logistics_quality,
                    "events": [{"event_code": "PICKED_UP", "event_time": "2026-09-08T00:00:00Z"}],
                }
            ],
            dataset_version="r4-b.synthetic-world.v1",
        )

    def _create_order_db(self) -> None:
        world = self.case.world
        conn = sqlite3.connect(self.order_db)
        try:
            conn.execute(
                """
                CREATE TABLE orders (
                    order_id TEXT PRIMARY KEY,
                    phone_last4 TEXT NOT NULL,
                    product_name TEXT NOT NULL,
                    amount REAL NOT NULL,
                    order_status TEXT NOT NULL,
                    pay_status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    can_apply_aftersales INTEGER NOT NULL,
                    carrier_code TEXT,
                    tracking_no TEXT
                )
                """
            )
            conn.execute(
                "INSERT INTO orders VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    world.order_id,
                    world.phone_last4,
                    "synthetic item",
                    1.0,
                    "PAID",
                    "PAID",
                    "2026-09-01T00:00:00Z",
                    1,
                    world.carrier_code,
                    world.tracking_no,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def runtime(
        self,
        *,
        failure_script: Mapping[str, Any] | None = None,
        retry_budget: int = 2,
        ledger_path: str | Path | None = None,
        disabled_capabilities: Sequence[str] = (),
    ) -> R4A2ARuntime:
        return R4A2ARuntime(
            db_path=self.order_db,
            logistics_db_path=self.logistics_db,
            ledger_path=self.ledger_db if ledger_path is None else ledger_path,
            scene_clock=_utc_clock,
            policy_reader=_policy_reader,
            policy_authority=_authority(),
            failure_script=failure_script if failure_script is not None else self.case.failure_script,
            retry_budget=retry_budget,
            disabled_capabilities=disabled_capabilities,
        )


def _payload(case: R4BInputV1, task: str) -> dict[str, Any]:
    world = case.world
    if task == "order":
        return {"order_id": world.order_id, "phone_last4": world.phone_last4}
    if task == "logistics":
        return {"carrier_code": world.carrier_code, "tracking_no": world.tracking_no, "phone_last4": world.phone_last4}
    if task == "policy":
        return {"query": world.policy_query, "top_k": 5}
    raise ValueError(f"unknown R4-B task: {task}")


def _safe_trace_event(
    trace: list[dict[str, Any]],
    *,
    run_id: str,
    trace_id: str,
    event_type: str,
    task: str,
    attempt_no: int = 1,
    payload: Mapping[str, Any] | None = None,
    message_id: str | None = None,
) -> None:
    seq_no = len(trace) + 1
    safe_payload = {"task": task, "attempt_no": int(attempt_no), **dict(payload or {})}
    payload_hash = sha256_json(safe_payload)
    event_id = project_trace_event_id(
        seq_no=seq_no,
        run_id=run_id,
        trace_id=trace_id,
        event_type=event_type,
        message_id=message_id or f"{run_id}:{task}",
        attempt_id=f"{run_id}:{task}:attempt:{attempt_no}",
        actor="r4-b-baseline",
        scene_clock=_utc_clock(),
        payload_hash=payload_hash,
    )
    trace.append(
        {
            "seq_no": seq_no,
            "event_id": event_id,
            "run_id": run_id,
            "trace_id": trace_id,
            "event_type": event_type,
            "message_id": message_id or f"{run_id}:{task}",
            "attempt_id": f"{run_id}:{task}:attempt:{attempt_no}",
            "actor": "r4-b-baseline",
            "scene_clock": _utc_clock().isoformat().replace("+00:00", "Z"),
            "payload": safe_payload,
            "payload_hash": payload_hash,
        }
    )


def _status_from_values(values: Iterable[str]) -> str:
    statuses = list(values)
    if statuses and all(item == "SUCCEEDED" for item in statuses):
        return "SUCCEEDED"
    if any(item == "SUCCEEDED" for item in statuses):
        return "PARTIAL"
    if any(item in {"PENDING", "BLOCKED"} for item in statuses):
        return "BLOCKED"
    if any(item == "LATE" for item in statuses):
        return "LATE"
    return "FAILED"


def _trace_projection(trace: Sequence[Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for item in trace:
        if isinstance(item, R4TraceEventV1):
            output.append(item.model_dump(mode="json"))
        elif isinstance(item, Mapping):
            output.append(dict(item))
    return output


def _trace_complete(trace: Sequence[Mapping[str, Any]], dispatches: Mapping[str, Mapping[str, Any]]) -> bool:
    seqs = [int(item.get("seq_no", -1)) for item in trace]
    if seqs != list(range(1, len(seqs) + 1)) or len({item.get("event_id") for item in trace}) != len(trace):
        return False
    terminal_event_types = {"A2A_RESULT_VERIFIED", "A2A_MESSAGE_FAILED", "A2A_SUPERVISOR_DECISION_VERIFIED"}
    rejection_event_types = {"A2A_MESSAGE_REJECTED", "A2A_DEPENDENCY_REJECTED", "A2A_MESSAGE_BLOCKED"}
    for original_task, dispatch in dispatches.items():
        if original_task == "duplicate_replay":
            task = next((key for key in dispatches if key not in {"duplicate_replay", "late_replay"}), original_task)
        elif original_task.endswith("_replay") or original_task.endswith("_first"):
            task = original_task.rsplit("_", 1)[0]
        else:
            task = original_task
        status = str(dispatch.get("status"))
        message_id = dispatch.get("message_id")
        if message_id:
            events = [item for item in trace if str(item.get("message_id")) == str(message_id)]
        else:
            events = [item for item in trace if str(item.get("payload", {}).get("task")) == task]
        terminals = [item for item in events if item.get("event_type") in terminal_event_types]
        if status in {"SUCCEEDED", "FAILED"} and len(terminals) != 1:
            if not (status == "FAILED" and any(item.get("event_type") in rejection_event_types for item in events)):
                return False
        if status == "LATE" and terminals and not original_task.endswith("_replay"):
            return False
    return True


def _observation(
    *,
    case: R4BInputV1,
    mode: str,
    world: _CaseWorld,
    status: str,
    dispatches: Mapping[str, Mapping[str, Any]],
    trace: Sequence[Any],
    physical_by_capability: Mapping[str, int],
    logical_calls: int,
    provider: Mapping[str, Any] | None = None,
    late_canonical_mutation: bool = False,
    freeze_error: str | None = None,
    ledger_used: bool = True,
    verifier_enabled: bool = True,
    lifecycle_enabled: bool = True,
    dedup_enabled: bool = True,
    retry_enabled: bool = True,
    disabled_capability: str | None = None,
    capability_set: Sequence[str] | None = None,
    execution_spec_used: bool = True,
) -> dict[str, Any]:
    trace_projection = _trace_projection(trace)
    physical_total = sum(int(value) for value in physical_by_capability.values())
    duplicate_reexecuted = any(
        str(key).endswith("replay")
        and not bool(value.get("duplicate"))
        and int(value.get("physical_call_count", 0)) > 0
        for key, value in dispatches.items()
    )
    return {
        "case_id": case.case_id,
        "mode": mode,
        "world_fingerprint": world.fingerprint,
        "status": status,
        "dispatches": {str(key): dict(value) for key, value in dispatches.items()},
        "trace": trace_projection,
        "trace_event_count": len(trace_projection),
        "trace_complete": _trace_complete(trace_projection, dispatches),
        "call_counts": {
            "logical_calls": int(logical_calls),
            "physical_attempts": int(physical_total),
            "physical_by_capability": {str(key): int(value) for key, value in physical_by_capability.items()},
            "write_capability_calls": 0,
        },
        "late_canonical_mutation": bool(late_canonical_mutation),
        "freeze_error": freeze_error,
        "provider": dict(provider or {}),
        "execution": {
            "ledger_used": bool(ledger_used),
            "verifier_enabled": bool(verifier_enabled),
            "lifecycle_enabled": bool(lifecycle_enabled),
            "dedup_enabled": bool(dedup_enabled),
            "retry_enabled": bool(retry_enabled),
            "disabled_capability": disabled_capability,
            "capability_set": sorted(set(_READ_CAPABILITIES.values() if capability_set is None else capability_set)),
            "registry_version": "r4.registry.v1",
            "execution_spec_used": bool(execution_spec_used),
        },
    }


def _dispatch_record(result: Any, *, replay_reason: str | None = None) -> dict[str, Any]:
    record = {
        "message_id": str(result.request.message_id),
        "task_id": str(result.request.task_id),
        "status": str(result.status),
        "error_code": result.error_code,
        "attempt_count": int(result.attempt_count),
        "physical_call_count": int(result.physical_call_count),
        "duplicate": bool(result.duplicate),
        "late": bool(result.late),
        "pending": bool(result.pending),
        "blocked": bool(result.blocked),
    }
    if replay_reason is not None:
        record["replay_reason"] = str(replay_reason)
    return record


def _execution_spec_rows(spec: Any) -> dict[str, Mapping[str, Any]]:
    """Parse an explicitly supplied execution spec without discovering files."""

    if spec is None:
        return {}
    if isinstance(spec, (str, Path)):
        raw = json.loads(Path(spec).read_text(encoding="utf-8"))
    else:
        raw = spec
    if isinstance(raw, Mapping) and "cases" in raw:
        rows = raw["cases"]
    elif isinstance(raw, Mapping) and any(key in raw for key in ("topology", "tasks", "dependencies")):
        raise ValueError("execution spec requires case keyed rows")
    elif isinstance(raw, Mapping):
        rows = [{"case_id": key, **dict(value)} for key, value in raw.items()]
    else:
        rows = raw
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("execution spec must contain a cases sequence")
    output: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or not row.get("case_id"):
            raise ValueError("execution spec case rows require case_id")
        case_id = str(row["case_id"])
        if case_id in output:
            raise ValueError("duplicate execution spec case_id")
        output[case_id] = dict(row)
    return output


def _load_execution_spec(spec: Any, *, input_ids: Sequence[str] | None = None) -> dict[str, Mapping[str, Any]]:
    rows = _execution_spec_rows(spec)
    if input_ids is not None:
        expected = set(map(str, input_ids))
        actual = set(rows)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(f"execution spec IDs must exactly match inputs; missing={missing}, extra={extra}")
    for case_id, row in rows.items():
        topology = row.get("topology")
        tasks = row.get("tasks")
        if topology is not None:
            topology = str(topology).strip().lower().replace(" ", "")
            if topology not in _VALID_EXECUTION_TOPOLOGIES:
                raise ValueError(f"unsupported execution topology for {case_id}")
            expected_tasks = _TASKS_FOR_TOPOLOGY[topology]
            if tasks is None:
                tasks = expected_tasks
            elif tuple(map(str, tasks)) != expected_tasks:
                raise ValueError(f"execution topology/task mismatch for {case_id}")
        if tasks is None:
            raise ValueError(f"execution spec requires tasks or topology for {case_id}")
        normalized_tasks = tuple(str(task) for task in tasks)
        if not normalized_tasks or any(task not in _READ_CAPABILITIES for task in normalized_tasks):
            raise ValueError(f"execution spec contains unknown task for {case_id}")
        if len(set(normalized_tasks)) != len(normalized_tasks):
            raise ValueError(f"execution spec tasks must be unique for {case_id}")
        normalized = dict(row)
        normalized["tasks"] = normalized_tasks
        if topology is not None:
            normalized["topology"] = topology
        dependencies = normalized.get("dependencies", ())
        if not isinstance(dependencies, Sequence) or isinstance(dependencies, (str, bytes)):
            raise ValueError(f"execution dependencies must be a sequence for {case_id}")
        normalized_dependencies: list[tuple[str, str]] = []
        for edge in dependencies:
            if isinstance(edge, (list, tuple)) and len(edge) == 2:
                normalized_dependencies.append((str(edge[0]), str(edge[1])))
            elif isinstance(edge, Mapping) and edge.get("upstream_task_id") and edge.get("downstream_task_id"):
                normalized_dependencies.append((str(edge["upstream_task_id"]), str(edge["downstream_task_id"])))
            else:
                raise ValueError(f"invalid execution dependency for {case_id}")
        normalized["dependencies"] = tuple(normalized_dependencies)
        if any(up not in normalized_tasks or down not in normalized_tasks for up, down in normalized["dependencies"]):
            raise ValueError(f"execution dependency references unknown task for {case_id}")
        disabled = normalized.get("disabled_capability", normalized.get("disabled_specialist"))
        if disabled is not None:
            disabled = str(disabled)
            if disabled in _READ_CAPABILITIES.values():
                disabled = next(task for task, capability in _READ_CAPABILITIES.items() if capability == disabled)
            if disabled not in normalized_tasks:
                raise ValueError(f"disabled capability is outside execution plan for {case_id}")
            normalized["disabled_capability"] = disabled
        rows[case_id] = normalized
    return rows


def _execution_plan(case: R4BInputV1, execution_spec: Mapping[str, Mapping[str, Any]] | None) -> Mapping[str, Any]:
    if execution_spec is None or case.case_id not in execution_spec:
        raise ValueError(f"no external execution spec for {case.case_id}")
    return execution_spec[case.case_id]


def _failure_script_for(case: R4BInputV1, plan: Mapping[str, Any]) -> Mapping[str, Any]:
    supplied = plan.get("failure_script")
    if supplied is not None:
        if not isinstance(supplied, Mapping):
            raise ValueError(f"failure_script must be a mapping for {case.case_id}")
        if any(key in supplied for key in ("kind", "code")):
            tasks = tuple(plan.get("tasks", ()))
            target = plan.get("failure_task") or (tasks[0] if tasks else None)
            return {str(target): dict(supplied)} if target else {}
        return dict(supplied)
    # Exact fault targets and error kinds must be present in the independent
    # execution spec.  Input failure families are never interpreted as an
    # execution oracle.
    return {}


def _run_a2a(
    case: R4BInputV1,
    world: _CaseWorld,
    *,
    execution_spec: Mapping[str, Any],
    mode: str = "a2a",
    retry_budget: int = 2,
    dedup_enabled: bool = True,
    disabled_specialist: str | None = None,
) -> dict[str, Any]:
    plan = execution_spec
    tasks = tuple(plan["tasks"])
    failure_script = _failure_script_for(case, plan)
    disabled_capability = disabled_specialist or plan.get("disabled_capability")
    if disabled_capability in _READ_CAPABILITIES.values():
        disabled_capability = next(task for task, capability in _READ_CAPABILITIES.items() if capability == disabled_capability)
    disabled_refs = (_READ_CAPABILITIES[str(disabled_capability)],) if disabled_capability else ()
    runtime = world.runtime(
        failure_script=failure_script,
        retry_budget=retry_budget,
        disabled_capabilities=disabled_refs,
    )
    run_id = f"r4b-{case.case_id}-{mode}"
    plan_id = f"plan-{case.case_id}"
    trace: Sequence[Any] = ()
    dispatches: dict[str, dict[str, Any]] = {}
    requests: dict[str, Any] = {}
    try:
        runtime.start_run(run_id=run_id, plan_revision_id=plan_id)

        def submit(task: str, *, dependency_ids: tuple[str, ...] = ()) -> Any:
            request = runtime.build_request(
                run_id=run_id,
                plan_revision_id=plan_id,
                task_id=task,
                capability_ref=_READ_CAPABILITIES[task],
                payload=_payload(case, task),
                dependency_message_ids=dependency_ids,
                idempotency_key=f"{case.case_id}:{task}",
            )
            requests[task] = request
            result = runtime.dispatch(request)
            dispatches[task] = _dispatch_record(result)
            return result

        action = str(plan.get("action") or (case.actions[0] if case.actions else ""))
        if case.failure_family == "schema_version":
            request = runtime.build_request(
                run_id=run_id,
                plan_revision_id=plan_id,
                task_id="order",
                capability_ref=_READ_CAPABILITIES["order"],
                payload=_payload(case, "order"),
                idempotency_key=f"{case.case_id}:order",
            )
            requests["order"] = request
            bad_request = request.model_copy(update={"schema_version": "r4.a2a.message.v0"})
            dispatches["order"] = _dispatch_record(runtime.dispatch(bad_request))
        elif action in {"exercise_ordering", "exercise_dependency_ordering", "dispatch_dependent_first"}:
            edges = tuple(plan.get("dependencies", ()))
            if not edges:
                raise ValueError(f"dependency ordering requires external dependency edges for {case.case_id}")
            upstream, downstream = edges[0]
            if upstream not in tasks or downstream not in tasks:
                raise ValueError(f"dependency edge is outside execution plan for {case.case_id}")
            order = runtime.build_request(
                run_id=run_id,
                plan_revision_id=plan_id,
                task_id=upstream,
                capability_ref=_READ_CAPABILITIES[upstream],
                payload=_payload(case, upstream),
                idempotency_key=f"{case.case_id}:{upstream}",
            )
            requests[upstream] = order
            logistics = runtime.build_request(
                run_id=run_id,
                plan_revision_id=plan_id,
                task_id=downstream,
                capability_ref=_READ_CAPABILITIES[downstream],
                payload=_payload(case, downstream),
                dependency_message_ids=(order.message_id,),
                idempotency_key=f"{case.case_id}:{downstream}",
            )
            requests[downstream] = logistics
            dispatches[f"{downstream}_first"] = _dispatch_record(runtime.dispatch(logistics), replay_reason="dependency_pending")
            order_result = runtime.dispatch(order)
            dispatches[upstream] = _dispatch_record(order_result)
            dispatches[f"{downstream}_replay"] = _dispatch_record(runtime.dispatch(logistics), replay_reason="dependency_release")
        else:
            for task in tasks:
                deps = ()
                for upstream, downstream in tuple(plan.get("dependencies", ())):
                    if downstream == task and upstream in requests:
                        deps = (requests[upstream].message_id,)
                        break
                submit(task, dependency_ids=deps)
                if action in {"replay_request", "replay_same_request"} and task == tasks[0]:
                    if dedup_enabled:
                        replay_result = runtime.dispatch(requests[task])
                    else:
                        replay_request = runtime.build_request(
                            run_id=run_id,
                            plan_revision_id=plan_id,
                            task_id=task,
                            capability_ref=_READ_CAPABILITIES[task],
                            payload=_payload(case, task),
                            idempotency_key=f"{case.case_id}:{task}:replay:2",
                        )
                        replay_result = runtime.dispatch(replay_request)
                    dispatches["duplicate_replay"] = _dispatch_record(replay_result, replay_reason="duplicate")
                if action in {"freeze_and_replay", "freeze_before_replay"} and task == tasks[-1]:
                    try:
                        runtime.freeze(run_id)
                    except A2AContractError:
                        pass
                    dispatches["late_replay"] = _dispatch_record(runtime.dispatch(requests[task]))
        if case.failure_family == "partial_branch" and "policy" not in requests:
            submit("policy")
        latest_by_task: dict[str, str] = {}
        for item in dispatches.values():
            latest_by_task[str(item.get("task_id"))] = str(item.get("status"))
        run_status = _status_from_values(latest_by_task.values())
        freeze_error = None
        if not any(item["status"] in {"PENDING", "BLOCKED", "LATE"} for item in dispatches.values()) and case.failure_family != "schema_version":
            try:
                runtime.freeze(run_id)
            except A2AContractError as exc:
                freeze_error = exc.code
        trace = runtime.trace(run_id)
        attempts = runtime.ledger.conn.execute(
            "SELECT COUNT(*) FROM a2a_attempts AS a JOIN a2a_messages AS m ON m.message_id=a.message_id WHERE m.run_id=?",
            (run_id,),
        ).fetchone()[0]
        physical = dict(runtime.physical_call_counts)
        return _observation(
            case=case,
            mode=mode,
            world=world,
            status=run_status,
            dispatches=dispatches,
            trace=trace,
            physical_by_capability=physical,
            logical_calls=int(attempts),
            late_canonical_mutation=False,
            freeze_error=freeze_error,
            ledger_used=True,
            verifier_enabled=True,
            lifecycle_enabled=True,
            dedup_enabled=dedup_enabled,
            retry_enabled=retry_budget > 1,
            disabled_capability=disabled_capability,
            capability_set=tuple(
                _READ_CAPABILITIES[task]
                for task in tasks
                if disabled_capability is None or _READ_CAPABILITIES[task] != _READ_CAPABILITIES[str(disabled_capability)]
            ),
        )
    finally:
        runtime.close()


def _direct_invoke(
    runtime: R4A2ARuntime,
    case: R4BInputV1,
    task: str,
    *,
    trace: list[dict[str, Any]],
    physical: dict[str, int],
    failure_script: Mapping[str, Any] | None = None,
    attempt_no: int = 1,
    disabled: str | None = None,
) -> dict[str, Any]:
    capability = _READ_CAPABILITIES[task]
    message_id = f"direct:{case.case_id}:{task}:{attempt_no}"
    idempotency_key = f"direct:{case.case_id}:{task}"
    if disabled == task:
        _safe_trace_event(trace, run_id=f"direct:{case.case_id}", trace_id=f"trace:{case.case_id}", event_type="A2A_MESSAGE_REJECTED", task=task, attempt_no=attempt_no, payload={"error_class": "SPECIALIST_DISABLED"}, message_id=message_id)
        return {"message_id": message_id, "idempotency_key": idempotency_key, "status": "FAILED", "error_code": "SPECIALIST_DISABLED", "attempt_count": attempt_no, "physical_call_count": 0, "duplicate": False, "late": False, "pending": False, "blocked": False}
    _safe_trace_event(trace, run_id=f"direct:{case.case_id}", trace_id=f"trace:{case.case_id}", event_type="A2A_ATTEMPT_STARTED", task=task, attempt_no=attempt_no, payload={"capability_ref": capability}, message_id=message_id)
    script = failure_script if failure_script is not None else case.failure_script
    rule = script.get(task) if isinstance(script, Mapping) else None
    kind = str(rule.get("kind") if isinstance(rule, Mapping) else rule or "").upper()
    if kind == "NON_RETRYABLE":
        _safe_trace_event(trace, run_id=f"direct:{case.case_id}", trace_id=f"trace:{case.case_id}", event_type="A2A_MESSAGE_FAILED", task=task, attempt_no=attempt_no, payload={"error_class": "NON_RETRYABLE_FAULT", "physical_call": False}, message_id=message_id)
        return {"message_id": message_id, "idempotency_key": idempotency_key, "status": "FAILED", "error_code": "NON_RETRYABLE_FAULT", "attempt_count": attempt_no, "physical_call_count": 0, "duplicate": False, "late": False, "pending": False, "blocked": False}
    if kind in {"TIMEOUT", "LOST"}:
        error = "INFRA_TIMEOUT" if kind == "TIMEOUT" else "INFRA_UNAVAILABLE"
        _safe_trace_event(trace, run_id=f"direct:{case.case_id}", trace_id=f"trace:{case.case_id}", event_type="A2A_MESSAGE_FAILED", task=task, attempt_no=attempt_no, payload={"error_class": error, "physical_call": False}, message_id=message_id)
        return {"message_id": message_id, "idempotency_key": idempotency_key, "status": "FAILED", "error_code": error, "attempt_count": attempt_no, "physical_call_count": 0, "duplicate": False, "late": False, "pending": False, "blocked": False}
    physical[capability] = physical.get(capability, 0) + 1
    try:
        route = ROUTES[capability]
        context = InvocationContext(
            session_id=f"r4b-session-{case.case_id}",
            user_id="r4b-evaluator",
            run_id=f"direct:{case.case_id}",
            plan_revision_id=f"plan-{case.case_id}",
            task_id=task,
            attempt_id=message_id,
            agent_ref=route.agent_ref,
            auth_scope=capability,
            idempotency_key=message_id,
            deadline=_utc_clock(),
            cancellation=False,
            config_version="r4-b.direct.v1",
            registry_version="r4.registry.v1",
            dataset_version="r4-b.synthetic-world.v1",
            trace_id=f"trace:{case.case_id}",
        )
        result = runtime.adapters[capability].invoke(_payload(case, task), context=context)
        if kind in {"SEMANTIC_WRONG", "SEMANTIC_MISMATCH"}:
            result = runtime._mutate_semantic_result(result)
        if result.ok and kind not in {"SEMANTIC_WRONG", "SEMANTIC_MISMATCH"}:
            _safe_trace_event(trace, run_id=f"direct:{case.case_id}", trace_id=f"trace:{case.case_id}", event_type="A2A_RESULT_VERIFIED", task=task, attempt_no=attempt_no, payload={"physical_call": True}, message_id=message_id)
            status = "SUCCEEDED"
            error_code = None
        else:
            error_code = "A2A_SEMANTIC_WRONG" if kind in {"SEMANTIC_WRONG", "SEMANTIC_MISMATCH"} else (result.error_code or "TOOL_EXECUTION_FAILED")
            _safe_trace_event(trace, run_id=f"direct:{case.case_id}", trace_id=f"trace:{case.case_id}", event_type="A2A_MESSAGE_FAILED", task=task, attempt_no=attempt_no, payload={"error_class": error_code, "physical_call": True}, message_id=message_id)
            status = "FAILED"
        return {"message_id": message_id, "idempotency_key": idempotency_key, "status": status, "error_code": error_code, "attempt_count": attempt_no, "physical_call_count": 1, "duplicate": False, "late": False, "pending": False, "blocked": False}
    except Exception as exc:
        error_code = getattr(exc, "code", None) or "TOOL_EXECUTION_FAILED"
        _safe_trace_event(trace, run_id=f"direct:{case.case_id}", trace_id=f"trace:{case.case_id}", event_type="A2A_MESSAGE_FAILED", task=task, attempt_no=attempt_no, payload={"error_class": error_code, "physical_call": True}, message_id=message_id)
        return {"message_id": message_id, "idempotency_key": idempotency_key, "status": "FAILED", "error_code": str(error_code), "attempt_count": attempt_no, "physical_call_count": 1, "duplicate": False, "late": False, "pending": False, "blocked": False}


def _run_direct(
    case: R4BInputV1,
    world: _CaseWorld,
    *,
    mode: str,
    execution_spec: Mapping[str, Any],
    supervisor_provider: Any = None,
    disabled_specialist: str = "policy",
) -> dict[str, Any]:
    # Baselines use the same registry/ports/tools but have no durable A2A
    # ledger.  The in-memory runtime object only supplies the admitted ports.
    plan = execution_spec
    failure_script = _failure_script_for(case, plan)
    runtime = world.runtime(failure_script=failure_script, ledger_path=":memory:")
    trace: list[dict[str, Any]] = []
    physical: dict[str, int] = {}
    dispatches: dict[str, dict[str, Any]] = {}
    try:
        if mode == "single_agent":
            if supervisor_provider is None:
                return _observation(
                    case=case,
                    mode=mode,
                    world=world,
                    status="NOT_RUN_NO_PROVIDER",
                    dispatches={},
                    trace=(),
                    physical_by_capability=physical,
                    logical_calls=0,
                    provider={"status": "NOT_RUN_NO_PROVIDER", "provider_called": False},
                    ledger_used=False,
                    verifier_enabled=False,
                    lifecycle_enabled=False,
                    dedup_enabled=False,
                    retry_enabled=False,
                    capability_set=(),
                    execution_spec_used=False,
                )
            boundary = R4SupervisorProviderBoundary(
                supervisor_provider,
                model_ref=getattr(supervisor_provider, "model_ref", None),
                clock=_utc_clock,
            )
            decision = boundary.decide(case.user_text)
            provider = decision.evidence.model_dump(mode="json")
            provider["decision_schema_valid"] = decision.decision is not None
            dispatches = {
                "supervisor": {
                    "status": "DECISION_RETURNED" if decision.decision is not None else "FAILED",
                    "error_code": decision.evidence.error_code,
                    "attempt_count": 1,
                    "physical_call_count": 0,
                    "duplicate": False,
                    "late": False,
                    "pending": False,
                    "blocked": False,
                }
            }
            _safe_trace_event(
                trace,
                run_id=f"single:{case.case_id}",
                trace_id=f"trace:{case.case_id}",
                event_type="A2A_SUPERVISOR_DECISION_VERIFIED" if decision.decision is not None else "A2A_MESSAGE_FAILED",
                task="supervisor",
                payload={"schema_valid": bool(decision.decision)},
            )
            return _observation(
                case=case,
                mode=mode,
                world=world,
                status=dispatches["supervisor"]["status"],
                dispatches=dispatches,
                trace=trace,
                physical_by_capability=physical,
                logical_calls=1,
                provider=provider,
                late_canonical_mutation=False,
                ledger_used=False,
                verifier_enabled=False,
                lifecycle_enabled=False,
                dedup_enabled=False,
                retry_enabled=False,
                capability_set=(),
                execution_spec_used=False,
            )
        if mode == "fixed_order":
            tasks = _FIXED_TASKS
        else:
            tasks = tuple(plan["tasks"])
        disabled = None
        for task in tasks:
            dispatches[task] = _direct_invoke(runtime, case, task, trace=trace, physical=physical, disabled=disabled, failure_script=failure_script)
            if case.failure_family == "duplicate" and task == tasks[0]:
                if mode in {"direct_call", "no_dedup"}:
                    dispatches["duplicate_replay"] = _direct_invoke(runtime, case, task, trace=trace, physical=physical, attempt_no=2, disabled=disabled)
                else:
                    _safe_trace_event(trace, run_id=f"direct:{case.case_id}", trace_id=f"trace:{case.case_id}", event_type="A2A_DUPLICATE", task=task, attempt_no=1, payload={"physical_call": False})
                    dispatches["duplicate_replay"] = dict(dispatches[task], duplicate=True, physical_call_count=0)
        if case.failure_family == "late":
            if mode == "direct_call":
                dispatches["late_replay"] = _direct_invoke(runtime, case, tasks[-1], trace=trace, physical=physical, attempt_no=2, disabled=disabled)
            else:
                dispatches["late_replay"] = {"status": "LATE", "error_code": "A2A_LATE_MESSAGE", "attempt_count": 0, "physical_call_count": 0, "duplicate": False, "late": True, "pending": False, "blocked": False}
        status = _status_from_values(item["status"] for item in dispatches.values())
        return _observation(
            case=case,
            mode=mode,
            world=world,
            status=status,
            dispatches=dispatches,
            trace=trace,
            physical_by_capability=physical,
            logical_calls=sum(int(item.get("attempt_count", 0)) for item in dispatches.values()),
            provider=None,
            late_canonical_mutation=bool(case.failure_family == "late" and mode == "direct_call"),
            ledger_used=False,
            verifier_enabled=False,
            lifecycle_enabled=False,
            dedup_enabled=(mode != "direct_call"),
            retry_enabled=False,
            disabled_capability=None,
            capability_set=tuple(_READ_CAPABILITIES[task] for task in tasks),
        )
    finally:
        runtime.close()


def run_r4_b_case(
    case: R4BInputV1 | Mapping[str, Any],
    mode: str,
    *,
    work_root: str | Path,
    execution_spec: Mapping[str, Any] | None = None,
    supervisor_provider: Any = None,
    disabled_specialist: str = "policy",
) -> dict[str, Any]:
    """Execute one case in one named mode using the shared case/world interface."""

    normalized = str(mode).lower()
    if normalized not in R4_B_MODES:
        raise ValueError(f"unknown R4-B mode: {mode}")
    parsed = case if isinstance(case, R4BInputV1) else R4BInputV1.model_validate(case)
    root = Path(work_root) / normalized / parsed.case_id
    started = perf_counter()
    world = _CaseWorld(parsed, root)
    spec_rows: dict[str, Mapping[str, Any]] = {}
    if execution_spec is not None:
        if isinstance(execution_spec, Mapping) and parsed.case_id in execution_spec and isinstance(execution_spec[parsed.case_id], Mapping):
            spec_rows = _load_execution_spec(execution_spec)
        elif isinstance(execution_spec, Mapping) and any(key in execution_spec for key in ("topology", "tasks", "dependencies")):
            spec_rows = _load_execution_spec({parsed.case_id: execution_spec})
        else:
            spec_rows = _load_execution_spec(execution_spec)
    plan = spec_rows.get(parsed.case_id)
    if normalized in _COORDINATION_MODES and plan is None:
        observation = _observation(
            case=parsed,
            mode=normalized,
            world=world,
            status="NOT_RUN_NO_EXECUTION_SPEC",
            dispatches={},
            trace=(),
            physical_by_capability={},
            logical_calls=0,
            provider={"status": "NOT_RUN_NO_EXECUTION_SPEC"},
            ledger_used=False,
            verifier_enabled=False,
            lifecycle_enabled=False,
            dedup_enabled=False,
            retry_enabled=False,
            capability_set=(),
            execution_spec_used=False,
        )
    elif normalized == "a2a":
        observation = _run_a2a(parsed, world, execution_spec=plan, mode=normalized)
    elif normalized == "no_retry":
        observation = _run_a2a(parsed, world, execution_spec=plan, mode=normalized, retry_budget=1)
    elif normalized == "single_agent":
        observation = _run_direct(parsed, world, mode=normalized, execution_spec=plan or {}, supervisor_provider=supervisor_provider, disabled_specialist=disabled_specialist)
    elif normalized == "disabled_specialist":
        observation = _run_a2a(parsed, world, execution_spec=plan, mode=normalized, disabled_specialist=disabled_specialist)
    elif normalized == "no_dedup":
        observation = _run_a2a(parsed, world, execution_spec=plan, mode=normalized, dedup_enabled=False)
    else:
        observation = _run_direct(parsed, world, mode=normalized, execution_spec=plan, disabled_specialist=disabled_specialist)
    observation["latency_ms"] = max(0.0, (perf_counter() - started) * 1000.0)
    observation["usage"] = {"input_tokens": 0, "output_tokens": 0}
    return observation


def _load_external_gold(
    gold: Any,
    *,
    input_path: str | Path | None = None,
    input_ids: Sequence[str] | None = None,
) -> tuple[dict[str, Mapping[str, Any]], str | None]:
    if gold is None:
        return {}, None
    if isinstance(gold, (str, Path)):
        target = Path(gold)
        if input_path is not None and target.resolve() == Path(input_path).resolve():
            raise ValueError("gold must be a separate file from the executable input")
        raw_bytes = target.read_bytes()
        raw = json.loads(raw_bytes.decode("utf-8"))
        digest = hashlib.sha256(raw_bytes).hexdigest()
    else:
        raw = gold
        digest = None
    if isinstance(raw, Mapping) and "cases" in raw:
        rows = raw["cases"]
    elif isinstance(raw, Mapping):
        rows = [{"case_id": key, **dict(value)} for key, value in raw.items()]
    else:
        rows = raw
    if not isinstance(rows, Sequence):
        raise ValueError("separate gold must contain a cases sequence")
    by_id: dict[str, Mapping[str, Any]] = {}
    seen: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping) or not row.get("case_id"):
            raise ValueError("gold case rows require case_id")
        case_id = str(row["case_id"])
        if case_id in by_id:
            raise ValueError("duplicate gold case_id")
        seen.append(case_id)
        by_id[case_id] = dict(row)
    if input_ids is not None:
        expected = set(map(str, input_ids))
        actual = set(by_id)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(f"gold IDs must exactly match inputs; missing={missing}, extra={extra}")
    return by_id, digest


def _actual_error_code(observation: Mapping[str, Any]) -> str | None:
    values = [row.get("error_code") for row in observation.get("dispatches", {}).values() if row.get("error_code")]
    return str(values[-1]) if values else None


def _actual_duplicate_reexecution(observation: Mapping[str, Any]) -> bool:
    return any(
        row.get("replay_reason") == "duplicate" and not bool(row.get("duplicate")) and int(row.get("physical_call_count", 0)) > 0
        for row in observation.get("dispatches", {}).values()
    )


def _protocol_match(observation: Mapping[str, Any], expected: Mapping[str, Any]) -> bool | None:
    """Derive protocol safety from explicit gold components only."""

    aliases = {
        "terminal_status": "status",
        "terminal_class": "status",
        "error_class": "error_code",
        "canonical_result_mutation": "canonical_mutation",
        "late_mutation": "late_canonical_mutation",
    }
    actual_values: dict[str, Any] = {
        "status": observation.get("status"),
        "error_code": _actual_error_code(observation),
        "physical_attempts": observation.get("call_counts", {}).get("physical_attempts"),
        "canonical_mutation": bool(observation.get("canonical_mutation", False)),
        "late_canonical_mutation": bool(observation.get("late_canonical_mutation", False)),
        "trace_complete": bool(observation.get("trace_complete", False)),
        "branch_status": tuple(sorted(str(item.get("status")) for item in observation.get("dispatches", {}).values())),
        "duplicate_physical_reexecution": _actual_duplicate_reexecution(observation),
    }
    component_keys = {
        "status",
        "terminal_status",
        "terminal_class",
        "error_code",
        "error_class",
        "physical_attempts",
        "canonical_mutation",
        "canonical_result_mutation",
        "late_canonical_mutation",
        "late_mutation",
        "trace_complete",
        "branch_status",
        "duplicate_physical_reexecution",
    }
    present = [key for key in component_keys if key in expected]
    if not present:
        return None
    for key in present:
        actual_key = aliases.get(key, key)
        actual = actual_values.get(actual_key)
        wanted = expected[key]
        if actual_key == "branch_status":
            if isinstance(wanted, (list, tuple)):
                wanted = tuple(sorted(str(item) for item in wanted))
        if actual != wanted:
            return False
    return True


def _score_gold(observations: Sequence[Mapping[str, Any]], gold: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if not gold:
        return {}
    # A provider-free single-agent observation is a deliberate NOT_RUN state,
    # never a quality failure or a success denominator.
    by_id = {
        str(row["case_id"]): row
        for row in observations
        if str(row.get("status")) not in {"NOT_RUN_NO_PROVIDER", "NOT_RUN_NO_EXECUTION_SPEC"}
    }
    metrics: dict[str, dict[str, Any]] = {}
    fields = ("status", "error_code", "physical_attempts", "trace_complete", "late_canonical_mutation")
    for field_name in fields:
        eligible = [case_id for case_id in gold if case_id in by_id and field_name in gold[case_id]]
        if not eligible:
            continue
        numerator = 0
        for case_id in eligible:
            observation = by_id[case_id]
            expected = gold[case_id]
            if field_name == "physical_attempts":
                actual = observation["call_counts"]["physical_attempts"]
            elif field_name == "trace_complete":
                actual = observation["trace_complete"]
            elif field_name == "late_canonical_mutation":
                actual = observation["late_canonical_mutation"]
            else:
                actual = observation.get("status") if field_name == "status" else _actual_error_code(observation)
            numerator += int(actual == expected[field_name])
        metrics[field_name] = _metric(numerator, len(eligible), case_ids=eligible)

    bool_metrics: dict[str, Callable[[Mapping[str, Any]], bool]] = {
        "normal_e2e_success": lambda obs: str(obs.get("status")) == "SUCCEEDED",
        "invalid_message_blocking": lambda obs: str(obs.get("status")) == "FAILED" and int(obs.get("call_counts", {}).get("physical_attempts", 0)) == 0,
        "duplicate_physical_reexecution": _actual_duplicate_reexecution,
        "recoverable_fault_recovery": lambda obs: str(obs.get("status")) == "SUCCEEDED" and any(int(row.get("attempt_count", 0)) > 1 for row in obs.get("dispatches", {}).values()),
        "failure_containment": lambda obs: str(obs.get("status")) in {"FAILED", "BLOCKED", "LATE"},
        "independent_branch_partial_preservation": lambda obs: str(obs.get("status")) == "PARTIAL",
        "partial_preservation": lambda obs: str(obs.get("status")) == "PARTIAL",
        "semantic_rejection": lambda obs: _actual_error_code(obs) == "A2A_SEMANTIC_WRONG",
        "semantic_wrong_rejection": lambda obs: _actual_error_code(obs) == "A2A_SEMANTIC_WRONG",
        "late_mutation": lambda obs: bool(obs.get("late_canonical_mutation", False)),
        "late_canonical_mutation": lambda obs: bool(obs.get("late_canonical_mutation", False)),
        "trace_completeness": lambda obs: bool(obs.get("trace_complete", False)),
    }
    for metric_name, actual_fn in bool_metrics.items():
        eligible = [case_id for case_id in gold if case_id in by_id and metric_name in gold[case_id]]
        if not eligible:
            continue
        numerator = sum(int(bool(actual_fn(by_id[case_id])) == bool(gold[case_id][metric_name])) for case_id in eligible)
        metrics[metric_name] = _metric(numerator, len(eligible), case_ids=eligible)

    protocol_ids = [case_id for case_id in gold if case_id in by_id and "protocol_safe_success" in gold[case_id] and _protocol_match(by_id[case_id], gold[case_id]) is not None]
    if protocol_ids:
        numerator = sum(int(bool(_protocol_match(by_id[case_id], gold[case_id])) == bool(gold[case_id]["protocol_safe_success"])) for case_id in protocol_ids)
        metrics["protocol_safe_success"] = _metric(numerator, len(protocol_ids), case_ids=protocol_ids)
    return metrics


def evaluate_r4_b_supervisor(
    inputs: Sequence[R4BInputV1 | Mapping[str, Any]],
    provider: Any,
    *,
    gold: Any = None,
) -> dict[str, Any]:
    """Evaluate the provider boundary without constructing A2A envelopes.

    A provider is supplied explicitly by the caller.  This function has no
    default model and therefore cannot accidentally read configuration or
    call an external service during deterministic R4-B preparation.
    """

    cases = tuple(item if isinstance(item, R4BInputV1) else R4BInputV1.model_validate(item) for item in inputs)
    gold_by_id, gold_sha256 = _load_external_gold(gold, input_ids=[item.case_id for item in cases])
    observations: list[dict[str, Any]] = []
    for case in cases:
        if case.failure_family != "normal":
            observations.append(
                {
                    "case_id": case.case_id,
                    "eligible": False,
                    "ineligible_reason": "fault_only_generic_case",
                    "provider": {"provider_called": False, "provider_returned": False},
                    "schema_valid": False,
                    "first_attempt_schema_valid": False,
                    "retried": False,
                    "provider_attempts": 0,
                    "decision": None,
                }
            )
            continue
        boundary = R4SupervisorProviderBoundary(provider, model_ref=getattr(provider, "model_ref", None), clock=_utc_clock)
        outcome = boundary.decide(case.user_text)
        provider_evidence = outcome.evidence.model_dump(mode="json")
        row: dict[str, Any] = {
            "case_id": case.case_id,
            "eligible": True,
            "provider": provider_evidence,
            "schema_valid": bool(outcome.decision is not None),
            "first_attempt_schema_valid": bool(provider_evidence.get("first_attempt_schema_valid", False)),
            "retried": bool(provider_evidence.get("retried", False)),
            "provider_attempts": int(provider_evidence.get("total_attempts", 1)),
            "decision": outcome.decision.model_dump(mode="json") if outcome.decision is not None else None,
        }
        observations.append(row)
    ids = [str(item["case_id"]) for item in observations if item.get("eligible")]
    eligible_observations = [item for item in observations if item.get("eligible")]
    eventual_schema_valid = _metric(
        sum(bool(item["schema_valid"]) for item in eligible_observations),
        len(ids),
        case_ids=ids,
    )
    metrics: dict[str, Any] = {
        "provider_return_rate": _metric(sum(bool(item["provider"].get("provider_returned")) for item in eligible_observations), len(ids), case_ids=ids),
        "first_attempt_schema_valid_rate": _metric(
            sum(bool(item["first_attempt_schema_valid"]) for item in eligible_observations),
            len(ids),
            case_ids=ids,
        ),
        # Keep schema_valid_rate as the established eventual-validity field.
        "schema_valid_rate": eventual_schema_valid,
        "eventual_schema_valid_rate": eventual_schema_valid,
        "retried_case_count": sum(bool(item["retried"]) for item in eligible_observations),
        "total_provider_attempts": sum(int(item["provider_attempts"]) for item in eligible_observations),
    }
    if gold_by_id:
        field_aliases = {
            "topology": ("topology",),
            "requested_capabilities": ("requested_capabilities", "capabilities"),
            "agents": ("agents", "expected_agents"),
            "dependency_edges": ("dependency_edges", "dependencies"),
            "entity_binding": ("entity_binding", "entities"),
            "needs_clarification": ("needs_clarification",),
        }
        for field_name, aliases in field_aliases.items():
            eligible = [
                item
                for item in observations
                if item.get("eligible") and item["case_id"] in gold_by_id and any(alias in gold_by_id[item["case_id"]] for alias in aliases)
            ]
            numerator = 0
            for item in eligible:
                decision = item["decision"] or {}
                expected_row = gold_by_id[item["case_id"]]
                expected_key = next(alias for alias in aliases if alias in expected_row)
                expected = expected_row[expected_key]
                if field_name == "agents":
                    actual = sorted(
                        {
                            f"{str(cap).split('/', 1)[0]}-agent@v1"
                            for cap in decision.get("requested_capabilities", [])
                        }
                    )
                elif field_name == "dependency_edges":
                    actual = sorted(
                        (str(edge.get("upstream_task_id")), str(edge.get("downstream_task_id")))
                        for edge in decision.get("dependencies", [])
                    )
                    expected = sorted(
                        (str(edge[0]), str(edge[1])) if isinstance(edge, (list, tuple)) else (str(edge.get("upstream_task_id")), str(edge.get("downstream_task_id")))
                        for edge in expected
                    )
                elif field_name == "entity_binding":
                    actual = decision.get("entities", {})
                else:
                    actual = decision.get(field_name)
                if field_name in {"requested_capabilities", "agents"}:
                    actual = sorted(actual or [])
                    expected = sorted(expected or [])
                numerator += int(actual == expected)
            metrics[field_name] = _metric(numerator, len(eligible), case_ids=[item["case_id"] for item in eligible])
        joint_eligible = []
        for item in observations:
            row = gold_by_id.get(item["case_id"])
            if not item.get("eligible") or row is None:
                continue
            if not (
                ("agents" in row or "expected_agents" in row)
                and ("requested_capabilities" in row or "capabilities" in row)
                and ("dependency_edges" in row or "dependencies" in row)
            ):
                continue
            joint_eligible.append(item)

        def _edge_set(value: Any) -> tuple[tuple[str, str], ...]:
            normalized_edges = []
            for edge in value or ():
                if isinstance(edge, Mapping):
                    normalized_edges.append((str(edge.get("upstream_task_id")), str(edge.get("downstream_task_id"))))
                elif isinstance(edge, (list, tuple)) and len(edge) == 2:
                    normalized_edges.append((str(edge[0]), str(edge[1])))
            return tuple(sorted(normalized_edges))

        joint_numerator = 0
        joint_ids = []
        for item in joint_eligible:
            row = gold_by_id[item["case_id"]]
            decision = item["decision"] or {}
            actual_agents = tuple(sorted({f"{str(cap).split('/', 1)[0]}-agent@v1" for cap in decision.get("requested_capabilities", [])}))
            actual_capabilities = tuple(sorted(str(cap) for cap in decision.get("requested_capabilities", [])))
            actual_edges = _edge_set(decision.get("dependencies", []))
            expected_agents = row.get("agents", row.get("expected_agents"))
            expected_capabilities = row.get("requested_capabilities", row.get("capabilities"))
            expected_edges = row.get("dependency_edges", row.get("dependencies"))
            expected_agents = tuple(sorted(str(agent) for agent in (expected_agents or ())))
            expected_capabilities = tuple(sorted(str(cap) for cap in (expected_capabilities or ())))
            joint_numerator += int(
                actual_agents == expected_agents
                and actual_capabilities == expected_capabilities
                and actual_edges == _edge_set(expected_edges)
            )
            joint_ids.append(item["case_id"])
        if joint_ids:
            metrics["exact_handoff_joint"] = _metric(joint_numerator, len(joint_ids), case_ids=joint_ids)
            metrics["exact_handoff"] = metrics["exact_handoff_joint"]
        # Stable names for the acceptance matrix.  The shorter names above
        # remain for compatibility with the initial evaluator shell.
        aliases = {
            "agents": "handoff_agent_set",
            "requested_capabilities": "handoff_capability_set",
            "needs_clarification": "clarification",
        }
        for source, alias in aliases.items():
            if source in metrics:
                metrics[alias] = metrics[source]
    return {
        "status": "SCORED_WITH_EXTERNAL_GOLD" if gold_by_id else "ENGINEERING_EVIDENCE_ONLY",
        "gold": {"used": bool(gold_by_id), "sha256": gold_sha256, "case_N": len(gold_by_id)},
        "unique_case_N": len(cases),
        "observations": observations,
        "metrics": metrics,
    }


def _evidence_metrics(observations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ids = [str(item["case_id"]) for item in observations]
    return {
        "trace_completeness": _metric(sum(bool(item["trace_complete"]) for item in observations), len(observations), case_ids=ids),
        "write_capability_calls_zero": _metric(sum(int(item["call_counts"]["write_capability_calls"]) == 0 for item in observations), len(observations), case_ids=ids),
        "call_count_evidence": _metric(sum("call_counts" in item for item in observations), len(observations), case_ids=ids),
        "world_reuse_guard": _metric(sum(bool(item.get("world_fingerprint")) for item in observations), len(observations), case_ids=ids),
    }


def evaluate_r4_b(
    inputs: Sequence[R4BInputV1 | Mapping[str, Any]],
    *,
    work_root: str | Path | None = None,
    modes: Sequence[str] = R4_B_MODES,
    gold: Any = None,
    input_path: str | Path | None = None,
    overlap_case_ids: Sequence[str] = (),
    supervisor_provider: Any = None,
    execution_spec: Any = None,
) -> dict[str, Any]:
    """Run R4-B modes using an explicit execution spec and optional gold.

    Without the independent execution spec this function performs only input
    manifest/schema smoke and the provider-free single-agent NOT_RUN record;
    it never infers a task graph from user text or coarse input metadata.
    """

    cases = tuple(item if isinstance(item, R4BInputV1) else R4BInputV1.model_validate(item) for item in inputs)
    if len({item.case_id for item in cases}) != len(cases):
        raise ValueError("R4-B input case IDs must be unique")
    root = Path(work_root) if work_root is not None else Path(tempfile.mkdtemp(prefix="r4-b-eval-"))
    root.mkdir(parents=True, exist_ok=True)
    input_ids = [item.case_id for item in cases]
    gold_by_id, gold_sha256 = _load_external_gold(gold, input_path=input_path, input_ids=input_ids)
    spec_rows = _load_execution_spec(execution_spec, input_ids=input_ids) if execution_spec is not None else {}
    manifest = input_manifest(cases)
    current_ids = set(item.case_id for item in cases)
    reference_ids = set(map(str, overlap_case_ids))
    intersection = sorted(current_ids & reference_ids)
    manifest["overlap"] = {
        "reference_case_N": len(reference_ids),
        "intersection_N": len(intersection),
        "intersection_rate": (len(intersection) / len(current_ids)) if current_ids else 0.0,
        "intersection_case_ids_sha256": sha256_json(intersection),
    }
    systems: dict[str, Any] = {}
    for mode in modes:
        normalized = str(mode).lower()
        if normalized in _COORDINATION_MODES and not spec_rows:
            observations: list[Mapping[str, Any]] = []
            mode_status = "NOT_RUN_NO_EXECUTION_SPEC"
        else:
            observations = [
                run_r4_b_case(
                    case,
                    normalized,
                    work_root=root,
                    execution_spec=spec_rows,
                    supervisor_provider=supervisor_provider,
                )
                for case in cases
            ]
            if normalized == "single_agent" and observations and all(item.get("status") == "NOT_RUN_NO_PROVIDER" for item in observations):
                mode_status = "NOT_RUN_NO_PROVIDER"
            else:
                mode_status = "EXECUTED"
        scored_metrics = _score_gold(observations, gold_by_id) if spec_rows and gold_by_id else {}
        systems[normalized] = {
            "mode": normalized,
            "independent_execution_path": normalized,
            "status": mode_status,
            "unique_case_N": len(cases),
            "observations": observations,
            "metrics": {**_evidence_metrics(observations), **scored_metrics},
            "call_count_evidence": {
                "logical_calls": sum(item["call_counts"]["logical_calls"] for item in observations),
                "physical_attempts": sum(item["call_counts"]["physical_attempts"] for item in observations),
                "write_capability_calls": sum(item["call_counts"]["write_capability_calls"] for item in observations),
                "latency_ms_total": sum(float(item.get("latency_ms", 0.0)) for item in observations),
                "usage": {
                    "input_tokens": sum(int(item.get("usage", {}).get("input_tokens", 0)) for item in observations),
                    "output_tokens": sum(int(item.get("usage", {}).get("output_tokens", 0)) for item in observations),
                },
            },
        }
    supervisor = evaluate_r4_b_supervisor(cases, supervisor_provider, gold=gold) if supervisor_provider is not None else {
        "status": "NOT_RUN",
        "gold": {"used": False, "sha256": None, "case_N": 0},
        "unique_case_N": 0,
        "observations": [],
        "metrics": {},
    }
    if not spec_rows and any(str(mode).lower() in _COORDINATION_MODES for mode in modes):
        report_status = "NOT_RUN_NO_EXECUTION_SPEC"
    elif systems and all(system.get("status") == "NOT_RUN_NO_PROVIDER" for system in systems.values()):
        report_status = "NOT_RUN_NO_PROVIDER"
    elif spec_rows and gold_by_id:
        report_status = "SCORED_WITH_EXTERNAL_GOLD"
    else:
        report_status = "ENGINEERING_EVIDENCE_ONLY"
    return {
        "report_version": R4_B_EVALUATOR_VERSION,
        "input_path": str(input_path) if input_path is not None else None,
        "input_manifest": manifest,
        "gold": {"used": bool(gold_by_id), "sha256": gold_sha256, "case_N": len(gold_by_id)},
        "execution_spec": {"used": bool(spec_rows), "case_N": len(spec_rows)},
        "supervisor": supervisor,
        "modes": systems,
        "status": report_status,
        "quality_claim": "NOT_VALIDATED",
    }


def evaluate_r4_b_inputs(
    input_path: str | Path,
    *,
    work_root: str | Path | None = None,
    modes: Sequence[str] = R4_B_MODES,
    gold: Any = None,
    overlap_case_ids: Sequence[str] = (),
    supervisor_provider: Any = None,
    execution_spec: Any = None,
) -> dict[str, Any]:
    cases = load_r4_b_inputs(input_path)
    return evaluate_r4_b(cases, work_root=work_root, modes=modes, gold=gold, input_path=input_path, overlap_case_ids=overlap_case_ids, supervisor_provider=supervisor_provider, execution_spec=execution_spec)


__all__ = [
    "R4_B_EVALUATOR_VERSION",
    "R4_B_MODES",
    "evaluate_r4_b",
    "evaluate_r4_b_inputs",
    "evaluate_r4_b_supervisor",
    "run_r4_b_case",
]
