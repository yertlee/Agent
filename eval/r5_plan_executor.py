"""Execute a model-produced R5 plan against the real R5 A2A runtime.

This is a thin plan-scheduling adapter over the production runtime
(``R5A2ARuntime`` / ``R5AfterSalesWriteService``), not a separate evaluation
executor: read nodes dispatch through the same A2A message contract used by
the application, and write nodes are never auto-approved.

Guarantees:
- an invalid plan is rejected and not executed;
- gold is never used to rebuild or replace the plan;
- a write capability stops at ``WAITING_CONFIRMATION`` unless an explicit,
  separate confirmation script is supplied by the caller;
- an unrequested write (not allowed by the caller's stage policy) is rejected
  without execution.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from agent.r5_a2a_runtime import R5A2ARuntime
from agent.r5_plan_contracts import (
    R5_ALL_CAPABILITIES,
    R5_ESCALATION_CAPABILITIES,
    R5_READ_CAPABILITIES,
    R5_WRITE_CAPABILITIES,
    R5PlanV1,
)
from agent.r5_replan import BoundedReplanPolicy

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class NodeOutcome:
    node_id: str
    capability_ref: str
    ok: bool
    error_code: str | None = None
    payload: Mapping[str, Any] | None = None
    status: str = ""
    message_id: str | None = None
    dependency_message_ids: tuple[str, ...] = ()


@dataclass
class ExecutionOutcome:
    status: str
    terminal: str
    executed: bool
    node_results: list[NodeOutcome] = field(default_factory=list)
    successful_reads: list[str] = field(default_factory=list)
    unrequested_write: bool = False
    physical_calls: int = 0
    error_code: str | None = None
    revisions: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "terminal": self.terminal,
            "executed": self.executed,
            "node_results": [
                {
                    **n.__dict__,
                    "dependency_message_ids": list(n.dependency_message_ids),
                }
                for n in self.node_results
            ],
            "successful_reads": sorted(set(self.successful_reads)),
            "unrequested_write": self.unrequested_write,
            "physical_calls": self.physical_calls,
            "error_code": self.error_code,
            "revisions": list(self.revisions),
        }


def _resolve_path(payload: Any, path: str) -> Any:
    if path.startswith("$."):
        path = path[2:]
    elif path.startswith("$"):
        path = path[1:]
    current = payload
    for part in [p for p in str(path).split(".") if p]:
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            raise KeyError(path)
    return current


def _policy_fixture() -> tuple[Any, dict[str, Any]]:
    """Minimal project-authored policy authority fixture (not gold)."""
    from agent.domain.objects import sha256_json

    version_tuple = ["r5.manifest.v1", "r5.exec.corpus.v1"] + [f"v{i}" for i in range(2, 14)] + ["a" * 64, "r5.exec.v1", "b" * 64, "c" * 64, "d" * 64, "e" * 64]
    source_version = sha256_json(version_tuple)
    authority = {
        "source": "r3.5.project-authored-kb",
        "source_version": source_version,
        "version_tuple": version_tuple,
        "strategy_checksum": "d" * 64,
        "evidence": [{"evidence_id": "ev-1", "source_id": "source-1", "version": "v1", "chunk_id": "chunk-1", "text_hash": "f" * 64, "locator": "chunk-1"}],
    }

    def reader(query: str) -> dict[str, Any]:
        return {
            "query": query,
            "status": "ANSWERED",
            "source": "r3.5.project-authored-kb",
            "source_version": source_version,
            "source_version_tuple": version_tuple,
            "evidence": [{"evidence_id": "ev-1", "source_id": "source-1", "version": "v1", "chunk_id": "chunk-1", "text_hash": "f" * 64, "locator": "chunk-1"}],
            "claims": [{"claim_id": "claim-1", "text": "verified", "evidence_ids": ["ev-1"]}],
        }

    return reader, authority


class R5PlanExecutor:
    """Plan scheduler over the real R5 runtime, with demo-world data paths."""

    def __init__(self, *, orders_db: str | Path, product_db: str | Path, aftersales_db: str | Path, logistics_db: str | Path | None = None):
        self.orders_db = str(orders_db)
        self.product_db = str(product_db)
        self.aftersales_db = str(aftersales_db)
        self.logistics_db = str(logistics_db) if logistics_db is not None else None

    def _runtime(self) -> R5A2ARuntime:
        reader, authority = _policy_fixture()
        return R5A2ARuntime(
            db_path=self.orders_db,
            product_db_path=self.product_db,
            aftersales_db_path=self.aftersales_db,
            logistics_db_path=self.logistics_db,
            ledger_path=":memory:",
            policy_reader=reader,
            policy_authority=authority,
        )

    def execute(
        self,
        plan_raw: Any,
        *,
        trusted_context: Mapping[str, str],
        allows_write: bool,
        confirmation_script: Mapping[str, Any] | None = None,
        replan_policy: BoundedReplanPolicy | None = None,
        allows_escalation: bool = False,
    ) -> ExecutionOutcome:
        try:
            plan = plan_raw if isinstance(plan_raw, R5PlanV1) else R5PlanV1.model_validate(plan_raw)
        except Exception as exc:
            return ExecutionOutcome(status="INVALID_PLAN", terminal="REJECT", executed=False, error_code=f"PLAN_INVALID:{type(exc).__name__}")

        if plan.needs_clarification:
            return ExecutionOutcome(status="CLARIFY", terminal="CLARIFY", executed=False)

        write_nodes = [n for n in plan.nodes if n.capability_ref in R5_WRITE_CAPABILITIES]
        if write_nodes and not allows_write:
            # Unrequested write: reject, never execute any node.
            return ExecutionOutcome(status="REJECT", terminal="REJECT", executed=False, unrequested_write=True, error_code="UNREQUESTED_WRITE")
        escalation_nodes = [n for n in plan.nodes if n.capability_ref in R5_ESCALATION_CAPABILITIES]
        if escalation_nodes and not allows_escalation:
            # Unrequested escalation: reject, never execute any node.
            return ExecutionOutcome(status="REJECT", terminal="REJECT", executed=False, error_code="UNREQUESTED_ESCALATION")

        runtime = self._runtime()
        order = plan.derived_dependency_order()
        if len(order) != len(plan.nodes):
            return ExecutionOutcome(status="INVALID_PLAN", terminal="REJECT", executed=False, error_code="PLAN_INVALID:CYCLIC")
        nodes_by_id = {n.node_id: n for n in plan.nodes}
        # All messages in one plan execution share the same protocol identity;
        # the A2A runtime requires dependency rows to have matching run and
        # plan IDs.  A fresh in-memory ledger is created for each execution.
        execution_run_id = "exec-r5-run"
        execution_plan_id = "exec-plan"
        dependency_nodes: dict[str, set[str]] = {node.node_id: set() for node in plan.nodes}
        for edge in plan.edges:
            dependency_nodes[edge.downstream_node_id].add(edge.upstream_node_id)
        for node in plan.nodes:
            for binding in node.bindings.values():
                if binding.kind == "result" and binding.source_node_id:
                    dependency_nodes[node.node_id].add(str(binding.source_node_id))

        outcomes: list[NodeOutcome] = []
        outputs: dict[str, Mapping[str, Any]] = {}
        message_ids: dict[str, str] = {}
        successful_reads: list[str] = []
        physical = 0
        revisions: list[dict[str, Any]] = []
        has_write = bool(write_nodes)
        try:
            for node_id in order:
                node = nodes_by_id[node_id]
                if node.capability_ref in R5_WRITE_CAPABILITIES or node.capability_ref in R5_ESCALATION_CAPABILITIES:
                    # Deferred: all reads run first so the pending action / escalation
                    # is built on complete facts.
                    continue

                payload: dict[str, Any] = {}
                binding_error: str | None = None
                for name, binding in node.bindings.items():
                    try:
                        if binding.kind == "context":
                            payload[name] = str(trusted_context[str(binding.context_key)])
                        else:
                            payload[name] = _resolve_path(outputs.get(str(binding.source_node_id)), str(binding.path))
                    except (KeyError, TypeError):
                        if binding.required:
                            binding_error = f"MISSING_BINDING:{name}"
                            break
                if binding_error:
                    outcomes.append(NodeOutcome(node_id=node_id, capability_ref=node.capability_ref, ok=False, status="FAILED", error_code=binding_error, dependency_message_ids=tuple(sorted(message_ids.get(dep, "") for dep in dependency_nodes[node_id] if message_ids.get(dep)))))
                    return ExecutionOutcome(status="FAILED", terminal="FAILED", executed=True, node_results=outcomes, successful_reads=successful_reads, physical_calls=physical, error_code=binding_error)
                for key, value in node.args.items():
                    payload.setdefault(str(key), value)

                if node.capability_ref not in R5_READ_CAPABILITIES:
                    outcomes.append(NodeOutcome(node_id=node_id, capability_ref=node.capability_ref, ok=False, status="FAILED", error_code="CAPABILITY_NOT_EXECUTABLE", dependency_message_ids=tuple(sorted(message_ids.get(dep, "") for dep in dependency_nodes[node_id] if message_ids.get(dep)))))
                    return ExecutionOutcome(status="FAILED", terminal="FAILED", executed=True, node_results=outcomes, successful_reads=successful_reads, physical_calls=physical, error_code="CAPABILITY_NOT_EXECUTABLE")

                dependency_message_ids = tuple(
                    sorted(message_ids[dep] for dep in dependency_nodes[node_id] if dep in message_ids)
                )
                request = runtime.build_request(
                    run_id=execution_run_id, plan_revision_id=execution_plan_id, task_id=node_id,
                    capability_ref=node.capability_ref, payload=payload,
                    dependency_message_ids=dependency_message_ids,
                )
                message_ids[node_id] = request.message_id
                dispatch = runtime.dispatch(request)
                physical += int(dispatch.physical_call_count or 0)
                if dispatch.status == "SUCCEEDED" and dispatch.canonical_result is not None:
                    payload_out = dispatch.canonical_result.payload or {}
                    outputs[node_id] = payload_out
                    successful_reads.append(node.capability_ref)
                    outcomes.append(NodeOutcome(node_id=node_id, capability_ref=node.capability_ref, ok=True, payload=payload_out, status="SUCCEEDED", message_id=request.message_id, dependency_message_ids=dependency_message_ids))
                    continue

                # Bounded replan: only onto a declared, argument-compatible
                # alternative capability, only for retryable read failures.
                decision = replan_policy.decide(capability_ref=node.capability_ref, error_code=dispatch.error_code, side_effect="READ_ONLY") if replan_policy is not None else None
                if decision is not None and decision.replan and decision.alternative_capability:
                    alt_capability = decision.alternative_capability
                    alt_request = runtime.build_request(
                        run_id=execution_run_id, plan_revision_id=execution_plan_id, task_id=f"{node_id}_r1",
                        capability_ref=alt_capability, payload=payload,
                        dependency_message_ids=dependency_message_ids,
                    )
                    message_ids[node_id] = alt_request.message_id
                    alt_dispatch = runtime.dispatch(alt_request)
                    physical += int(alt_dispatch.physical_call_count or 0)
                    revisions.append({"node_id": node_id, "from": node.capability_ref, "to": alt_capability, "reason": decision.reason})
                    if alt_dispatch.status == "SUCCEEDED" and alt_dispatch.canonical_result is not None:
                        payload_out = alt_dispatch.canonical_result.payload or {}
                        outputs[node_id] = payload_out
                        successful_reads.append(alt_capability)
                        outcomes.append(NodeOutcome(node_id=node_id, capability_ref=alt_capability, ok=True, payload=payload_out, status="SUCCEEDED", message_id=alt_request.message_id, dependency_message_ids=dependency_message_ids))
                        continue
                    outcomes.append(NodeOutcome(node_id=node_id, capability_ref=alt_capability, ok=False, status=alt_dispatch.status, error_code=alt_dispatch.error_code, message_id=alt_request.message_id, dependency_message_ids=dependency_message_ids))
                    return ExecutionOutcome(status="FAILED", terminal="FAILED", executed=True, node_results=outcomes, successful_reads=successful_reads, physical_calls=physical, error_code=alt_dispatch.error_code or alt_dispatch.status, revisions=revisions)
                outcomes.append(NodeOutcome(node_id=node_id, capability_ref=node.capability_ref, ok=False, status=dispatch.status, error_code=dispatch.error_code, message_id=request.message_id, dependency_message_ids=dependency_message_ids))
                return ExecutionOutcome(status="FAILED", terminal="FAILED", executed=True, node_results=outcomes, successful_reads=successful_reads, physical_calls=physical, error_code=dispatch.error_code or dispatch.status, revisions=revisions)
        finally:
            runtime.close()

        if escalation_nodes:
            # Reads (fact collection) completed; the run escalates to a human.
            node = escalation_nodes[0]
            outcomes.append(NodeOutcome(node_id=node.node_id, capability_ref=node.capability_ref, ok=True, status="ESCALATED"))
            return ExecutionOutcome(status="ESCALATED", terminal="HUMAN", executed=True, node_results=outcomes, successful_reads=successful_reads, physical_calls=physical, revisions=revisions)

        if has_write:
            # Guarded write reached with all reads complete: stop at the
            # pending-confirmation stage goal. The evaluator never approves.
            write_node = write_nodes[0]
            if confirmation_script is None:
                outcomes.append(NodeOutcome(node_id=write_node.node_id, capability_ref=write_node.capability_ref, ok=False, status="WAITING_CONFIRMATION", error_code="CONFIRMATION_REQUIRED"))
                return ExecutionOutcome(status="WAITING_CONFIRMATION", terminal="PENDING_CONFIRMATION", executed=True, node_results=outcomes, successful_reads=successful_reads, physical_calls=physical)
            outcomes.append(NodeOutcome(node_id=write_node.node_id, capability_ref=write_node.capability_ref, ok=False, status="CONFIRMATION_SCRIPT_UNSUPPORTED", error_code="CONFIRMATION_SCRIPT_UNSUPPORTED"))
            return ExecutionOutcome(status="FAILED", terminal="FAILED", executed=True, node_results=outcomes, successful_reads=successful_reads, physical_calls=physical, error_code="CONFIRMATION_SCRIPT_UNSUPPORTED")

        return ExecutionOutcome(status="COMPLETED", terminal="ANSWER", executed=True, node_results=outcomes, successful_reads=successful_reads, physical_calls=physical)


__all__ = ["ExecutionOutcome", "NodeOutcome", "R5PlanExecutor"]
