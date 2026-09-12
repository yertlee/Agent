"""R2 order-to-logistics dependency runtime.

This module is deliberately separate from the R1 single-tool runtime.  It
uses the canonical M2/M3 Registry, AgentPort, Executor, PlanRevision,
ResultBinding and TraceRecorder owners. Orders are read from the production
SQLite projection, while logistics facts are read from an explicit,
versioned R2 snapshot database; the runtime never calls the legacy simulator.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Optional, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from .agents.manifest import AgentManifest
from .agents.ports import LogisticsAgent, OrderAgent, TypedAgentResult
from .domain.objects import (
    AttemptStatus,
    InputBinding,
    PlanRevision,
    Result,
    ResultStatus,
    Run,
    Task,
    TaskAttempt,
    TaskStatus,
    Usage,
    sha256_json,
)
from .domain.plan_validator import PlanLimits, PlanValidator
from .interactive_runtime import (
    ChatOpenAIStructuredProvider,
    LLMRuntimeConfig,
    ModelClient,
    ModelResult,
    ModelUsage,
)
from .m2_context import InvocationContext
from .m1_runtime import Supervisor as M1Supervisor
from .m2_registry import Registry, ToolSpec
from .m3_registry import M3Registry
from .m3_bindings import BindingResolver, ResultBinding
from .m3_supervisor import CandidateIntent, M3Supervisor, PlanDraft
from .r1_5_router import (
    BusinessIntent,
    CustomerGoalV1,
    RouterContractError,
    normalize_customer_goal,
)
from .r2_logistics_repository import R2_LOGISTICS_SOURCE_VERSION, R2LogisticsRepository, order_ref_hash
from .storage.m2 import M2Repository
from .trace.events import sensitive_surface_scan
from eval.harness.contracts import ExecutionMode, FailureScript, FailureTrigger, FreezeBundle, VersionTuple, WorldSnapshot
from eval.harness.trace_recorder import TraceRecorder, verify_bundle


def _safe_execution_id(prefix: str) -> str:
    """Use identifiers safe for canonical artifact path scanning."""
    return prefix + uuid4().hex.translate(str.maketrans("0123456789", "abcdefghij"))


class R2RuntimeError(RuntimeError):
    def __init__(self, code: str, message: str = "R2 runtime failed") -> None:
        self.code = code
        super().__init__(message)


class DagNodeV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    node_key: str = Field(min_length=1)
    capability_ref: str = Field(min_length=1)
    depends_on: tuple[str, ...] = ()
    input_sources: dict[str, str] = Field(default_factory=dict)


class DagCandidateV1(BaseModel):
    """Planner output: nodes and edges only, never execution outcomes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["r2.dag-candidate.v1"] = "r2.dag-candidate.v1"
    nodes: tuple[DagNodeV1, ...] = Field(min_length=1)


class R2ProviderProtocol(Protocol):
    def __call__(self, prompt: str, output_schema: type[Any]) -> Any: ...


@dataclass(frozen=True)
class RepositoryRead:
    success: bool
    code: str
    data: Optional[dict[str, Any]] = None


class OrderLogisticsRepository:
    """Read-only owner lookup plus a versioned order-derived logistics snapshot."""

    source_version = "versioned_order_derived_logistics_snapshot.v1"

    def __init__(self, db_path: str | Path, *, logistics_db_path: str | Path | None = None, **_compatibility_options: Any):
        self.db_path = str(db_path)
        self.logistics_db_path = str(logistics_db_path) if logistics_db_path else None
        self.logistics_repository = R2LogisticsRepository(logistics_db_path) if logistics_db_path else None

    def _connect(self) -> sqlite3.Connection:
        path = Path(self.db_path)
        if not path.is_file():
            raise FileNotFoundError(self.db_path)
        conn = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only=ON")
        return conn

    @staticmethod
    def _columns(conn: sqlite3.Connection) -> set[str]:
        return {str(row[1]) for row in conn.execute("PRAGMA table_info(orders)").fetchall()}

    def _fetch_order(self, order_id: str, phone_last4: str | None = None) -> Optional[dict[str, Any]]:
        conn = self._connect()
        try:
            columns = self._columns(conn)
            wanted = [
                "order_id", "phone_last4", "order_status", "pay_status", "shipment_status",
                "created_at", "shipped_at", "delivered_at", "carrier_code", "tracking_no",
            ]
            selected = [name for name in wanted if name in columns]
            sql = f"SELECT {', '.join(selected)} FROM orders WHERE order_id = ?"
            params: list[Any] = [order_id]
            if phone_last4 is not None and "phone_last4" in columns:
                sql += " AND phone_last4 = ?"
                params.append(phone_last4)
            row = conn.execute(sql, tuple(params)).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def _fetch_by_tracking(self, carrier_code: str, tracking_no: str, phone_last4: str | None) -> Optional[dict[str, Any]]:
        conn = self._connect()
        try:
            columns = self._columns(conn)
            if "tracking_no" not in columns or "carrier_code" not in columns:
                return None
            wanted = [
                "order_id", "phone_last4", "order_status", "pay_status", "shipment_status",
                "created_at", "shipped_at", "delivered_at", "carrier_code", "tracking_no",
            ]
            selected = [name for name in wanted if name in columns]
            sql = f"SELECT {', '.join(selected)} FROM orders WHERE carrier_code = ? AND tracking_no = ?"
            params: list[Any] = [carrier_code, tracking_no]
            if phone_last4 is not None and "phone_last4" in columns:
                sql += " AND phone_last4 = ?"
                params.append(phone_last4)
            row = conn.execute(sql, tuple(params)).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    @staticmethod
    def _order_snapshot(row: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "order_status": str(row.get("order_status") or ""),
            "pay_status": str(row.get("pay_status") or ""),
            "shipment_status": str(row.get("shipment_status") or ""),
            "carrier_code": str(row.get("carrier_code") or ""),
            "tracking_no": str(row.get("tracking_no") or ""),
            "created_at": str(row.get("created_at") or ""),
            "shipped_at": str(row.get("shipped_at") or ""),
            "delivered_at": str(row.get("delivered_at") or ""),
        }

    def _quality(self, row: Mapping[str, Any]) -> str:
        order_status = str(row.get("order_status") or "").upper()
        shipment_status = str(row.get("shipment_status") or "").upper()
        delivered_at = str(row.get("delivered_at") or "")
        if order_status == "DELIVERED" and shipment_status not in {"DELIVERED", "SIGNED", ""}:
            return "CONFLICT"
        if not row.get("carrier_code") or not row.get("tracking_no"):
            return "MISSING"
        # The source has no updated_at column.  It is therefore fresh only as
        # a source snapshot, never by wall-clock inference.
        return "FRESH" if delivered_at or row.get("shipped_at") or row.get("created_at") else "MISSING"

    def read_order(self, order_id: str, phone_last4: str) -> RepositoryRead:
        try:
            row = self._fetch_order(order_id)
        except (FileNotFoundError, sqlite3.Error):
            return RepositoryRead(False, "INFRA_UNAVAILABLE")
        if row is None:
            return RepositoryRead(False, "ORDER_NOT_FOUND")
        if str(row.get("phone_last4") or "") != str(phone_last4):
            return RepositoryRead(False, "AUTH_IDENTITY_MISMATCH")
        quality = self._quality(row)
        snapshot = self._order_snapshot(row)
        data = {
            "order_status": snapshot["order_status"],
            "pay_status": snapshot["pay_status"],
            "shipment_status": snapshot["shipment_status"],
            "carrier_code": snapshot["carrier_code"],
            "tracking_no": snapshot["tracking_no"],
            "source": "system",
            "source_version": self.source_version,
            "data_quality": quality,
            "snapshot_hash": sha256_json(snapshot),
        }
        return RepositoryRead(True, "OK", data)

    def read_logistics(self, carrier_code: str, tracking_no: str, phone_last4: str | None = None) -> RepositoryRead:
        if not carrier_code or not tracking_no:
            return RepositoryRead(False, "DATA_MISSING")
        if self.logistics_repository is None:
            return RepositoryRead(False, "INFRA_UNAVAILABLE")
        try:
            row = self._fetch_by_tracking(carrier_code, tracking_no, phone_last4)
        except (FileNotFoundError, sqlite3.Error):
            return RepositoryRead(False, "INFRA_UNAVAILABLE")
        if row is None:
            if phone_last4 and self._fetch_by_tracking(carrier_code, tracking_no, None) is not None:
                return RepositoryRead(False, "AUTH_IDENTITY_MISMATCH")
            return RepositoryRead(False, "ORDER_NOT_FOUND")
        try:
            logistics = self.logistics_repository.read(
                order_ref=order_ref_hash(str(row.get("order_id") or "")),
                carrier_code=carrier_code,
                tracking_no=tracking_no,
            )
        except (FileNotFoundError, sqlite3.Error):
            return RepositoryRead(False, "INFRA_UNAVAILABLE")
        if logistics is None:
            return RepositoryRead(False, "DATA_MISSING")
        quality = str(logistics.get("data_quality") or "MISSING").upper()
        if quality == "MISSING":
            return RepositoryRead(False, "DATA_MISSING")
        if quality == "STALE":
            return RepositoryRead(False, "DATA_STALE")
        if quality == "CONFLICT":
            return RepositoryRead(False, "DATA_CONFLICT")
        data = {
            "carrier_code": str(logistics["carrier_code"]),
            "tracking_no": str(logistics["tracking_no"]),
            "delivery_state": str(logistics["delivery_state"]),
            "shipment_status": str(logistics["shipment_status"]),
            "observed_at": str(logistics["observed_at"]),
            "events": [
                {"event_code": str(event["event_code"]), "event_time": str(event["event_time"]), "event_hash": str(event["event_hash"])}
                for event in (logistics.get("events") or [])
            ],
            "source": str(logistics["source_name"]),
            "source_version": str(logistics["source_version"]),
            "data_quality": quality,
            "snapshot_hash": str(logistics["snapshot_hash"]),
        }
        return RepositoryRead(True, "OK", data)


class DeterministicR2Provider:
    """Explicit goal/candidate test adapter; never a live provider.

    It accepts preconstructed structured values only.  It deliberately has no
    message or keyword interpretation so it cannot masquerade as R1.5.
    """

    test_only = True

    def __init__(self, *, goal: CustomerGoalV1 | None = None, candidate: DagCandidateV1 | None = None):
        self.goal = goal
        self.candidate = candidate

    def __call__(self, prompt: str, output_schema: type[Any]) -> Any:
        if output_schema is DagCandidateV1:
            if self.candidate is None:
                raise RouterContractError("deterministic R2 adapter requires an explicit DAG candidate")
            return ModelResult(self.candidate)
        raise RouterContractError("deterministic R2 adapter received unsupported schema")


@dataclass(frozen=True)
class R2RunResult:
    run_id: str
    session_id: str
    route: str
    status: str
    code: str
    response: str
    plan_revisions: tuple[PlanRevision, ...]
    results: tuple[Result, ...]
    trace_path: str
    freeze_bundle: FreezeBundle
    bundle_verification: dict[str, Any]
    model_calls: int
    tool_calls: int
    replan_count: int


class R2OrderLogisticsRuntime:
    """Conditional order/logistics DAG runtime with bounded local replan."""

    def __init__(self, *, db_path: str | Path, logistics_db_path: str | Path | None = None, provider: R2ProviderProtocol | None = None, artifact_root: str | Path | None = None, mode: ExecutionMode | str = ExecutionMode.SIMULATED, failure_script: Mapping[str, Mapping[str, Any]] | None = None, timeout_seconds: float = 15.0, max_retries: int = 0, allow_replan: bool = True, **_compatibility_options: Any):
        self.db_path = str(db_path)
        self.mode = ExecutionMode(str(mode)) if not isinstance(mode, ExecutionMode) else mode
        if self.mode not in {ExecutionMode.LIVE, ExecutionMode.SIMULATED, ExecutionMode.FAULT}:
            raise R2RuntimeError("UNSUPPORTED_EXECUTION_MODE")
        if self.mode is ExecutionMode.FAULT and not failure_script:
            raise R2RuntimeError("FAULT_REQUIRES_EXPLICIT_SCRIPT")
        if self.mode is ExecutionMode.LIVE and getattr(provider, "test_only", False):
            raise R2RuntimeError("LIVE_REQUIRES_EXTERNAL_PROVIDER")
        self.artifact_root = Path(artifact_root) if artifact_root else Path(tempfile.mkdtemp(prefix="r2-runtime-"))
        self.provider = provider or (DeterministicR2Provider() if self.mode is not ExecutionMode.LIVE else None)
        self.repo_source = OrderLogisticsRepository(self.db_path, logistics_db_path=logistics_db_path)
        self.failure_script = {str(k): dict(v) for k, v in (failure_script or {}).items()}
        self.allow_replan = bool(allow_replan)
        triggers = []
        for index, (tool_ref, rule) in enumerate(sorted(self.failure_script.items())):
            triggers.append({
                "trigger_id": f"r2-trigger-{index + 1}",
                "tool_ref": tool_ref,
                "action": "RETURN_ERROR",
                "error_code": str(rule.get("code") or "INFRA_TIMEOUT"),
                "layer": str(rule.get("layer") or "infra"),
                "count": int(rule.get("count") or 1),
            })
        self.failure_script_contract = FailureScript.model_validate({"version": "r2.failure-script.v1", "triggers": triggers})
        self.config = LLMRuntimeConfig.from_environment(db_path=self.db_path, env={"ECOMMERCE_DB_PATH": self.db_path, "LLM_TIMEOUT_SECONDS": str(timeout_seconds), "LLM_MAX_RETRIES": str(max_retries)})

    def _version_tuple(self) -> VersionTuple:
        tool_impl = R2_LOGISTICS_SOURCE_VERSION if self.repo_source.logistics_repository is not None else self.repo_source.source_version
        return VersionTuple(schema="r2.schema.v1", model=self.config.model if self.mode is ExecutionMode.LIVE else "r2-deterministic", prompt="r2.prompt.v1", code="r2.order-logistics.v1", registry="r2.registry.v1", tool_impl=tool_impl, config="r2.config." + self.config.config_hash, policy_catalog="r2.policy.none.v1", kb="r2.kb.none.v1", dataset="dev-order-logistics-dag-v1", harness="r2.harness.v1", trace_schema="m1.trace.v1", evaluator="r2.evaluator.v1", simulator="r2.no-simulator-live.v1", world_template="r2.world.v1", seed=0)

    @staticmethod
    def _manifests() -> tuple[AgentManifest, ...]:
        return (
            AgentManifest(agent_ref="order-agent@v1", owner="order-agent", role="R2 order read", allowed_capabilities=("order/read@v1",), input_types=("OrderQuery",), output_types=("OrderFact", "Result")),
            AgentManifest(agent_ref="logistics-agent@v1", owner="logistics-agent", role="R2 logistics read", allowed_capabilities=("logistics/read@v1",), input_types=("LogisticsQuery",), output_types=("LogisticsFact", "Result")),
        )

    def _registry(self) -> M3Registry:
        # Build the narrow R2 registry directly.  Importing the legacy full
        # registry also imports the optional RAG index stack; R2 must not
        # initialize or depend on R3 retrieval just to read orders/logistics.
        specs: list[ToolSpec] = []
        calls: dict[str, int] = {ref: 0 for ref in self.failure_script}

        def maybe_fail(tool_ref: str, fn: Callable[..., dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
            rule = self.failure_script.get(tool_ref)
            if rule and calls.get(tool_ref, 0) == 0:
                calls[tool_ref] = 1
                return {"success": False, "code": str(rule.get("code") or "INFRA_TIMEOUT"), "message": "declared R2 fault", "data": None}
            return fn(**kwargs)

        def order_call(*, order_id: str, phone_last4: str) -> dict[str, Any]:
            read = self.repo_source.read_order(order_id, phone_last4)
            return {"success": read.success, "code": read.code, "message": "order read", "data": read.data}

        def logistics_call(*, carrier_code: str, tracking_no: str, phone_last4: str | None = None) -> dict[str, Any]:
            read = self.repo_source.read_logistics(carrier_code, tracking_no, phone_last4)
            return {"success": read.success, "code": read.code, "message": "logistics read", "data": read.data}

        implementation_mode = "LIVE" if self.mode is ExecutionMode.LIVE else ("FAULT" if self.mode is ExecutionMode.FAULT else "SIMULATED")
        specs.append(ToolSpec(tool_ref="order/get_info@v1", capability_ref="order/read@v1", owner="order-agent", args_schema="order.read.v1", result_schema="order.result.v1", risk="READ", implementation_mode=implementation_mode, side_effect="READ_ONLY", timeout_ms=5000, allowed_error_codes=("AUTH_IDENTITY_MISMATCH", "ORDER_NOT_FOUND", "DATA_MISSING", "INFRA_UNAVAILABLE", "INFRA_TIMEOUT", "DATA_STALE", "DATA_CONFLICT"), callable=lambda **kwargs: maybe_fail("order/get_info@v1", order_call, **kwargs)))
        specs.append(ToolSpec(tool_ref="logistics/query@v1", capability_ref="logistics/read@v1", owner="logistics-agent", args_schema="logistics.query.v1", result_schema="logistics.result.v1", risk="READ", implementation_mode=implementation_mode, side_effect="READ_ONLY", timeout_ms=5000, allowed_error_codes=("AUTH_IDENTITY_MISMATCH", "ORDER_NOT_FOUND", "DATA_MISSING", "DATA_STALE", "DATA_CONFLICT", "INFRA_UNAVAILABLE", "INFRA_TIMEOUT"), callable=lambda **kwargs: maybe_fail("logistics/query@v1", logistics_call, **kwargs)))
        return M3Registry(canonical=Registry(specs))

    def _trusted_context(self, *, user_id: str, session_id: str, run_id: str, plan: PlanRevision, task: Task, attempt_id: str, agent_ref: str, capability_ref: str) -> InvocationContext:
        return InvocationContext(
            session_id=session_id,
            user_id=str(user_id),
            run_id=run_id,
            plan_revision_id=plan.plan_revision_id,
            task_id=task.task_id,
            attempt_id=attempt_id,
            agent_ref=agent_ref,
            auth_scope=capability_ref,
            idempotency_key=_safe_execution_id("r2_idemp_"),
            deadline=datetime.now(timezone.utc) + timedelta(seconds=10),
            config_version=self.config.agent_config.config_version,
            registry_version="r2.registry.v1",
            dataset_version="dev-order-logistics-dag-v1",
            trace_id=_safe_execution_id("r2_trace_"),
        )

    def _plan_from_candidate(
        self,
        *,
        run_id: str,
        goal: CustomerGoalV1,
        candidate: DagCandidateV1,
        repository: M2Repository,
        version: int = 1,
        supersedes: str | None = None,
    ) -> PlanRevision:
        """Map a structured DAG candidate through the canonical supervisor."""
        if len({node.node_key for node in candidate.nodes}) != len(candidate.nodes):
            raise R2RuntimeError("DAG_INVALID")
        registry = self._registry()
        candidate_capabilities = tuple(node.capability_ref for node in candidate.nodes)
        if set(candidate_capabilities) != set(goal.required_capabilities) or len(candidate_capabilities) != len(goal.required_capabilities):
            raise R2RuntimeError("DAG_CAPABILITY_MISMATCH")
        by_capability = {node.capability_ref: node for node in candidate.nodes}
        if goal.goal_type is BusinessIntent.ORDER_QUERY:
            order_node = by_capability["order/read@v1"]
            if order_node.depends_on or order_node.input_sources:
                raise R2RuntimeError("DAG_TOPOLOGY_INVALID")
        elif goal.goal_type is BusinessIntent.LOGISTICS_QUERY:
            logistics_node = by_capability["logistics/read@v1"]
            if logistics_node.depends_on or logistics_node.input_sources:
                raise R2RuntimeError("DAG_TOPOLOGY_INVALID")
        else:
            order_node = by_capability["order/read@v1"]
            logistics_node = by_capability["logistics/read@v1"]
            expected_sources = {
                "carrier_code": f"{order_node.node_key}.payload.carrier_code",
                "tracking_no": f"{order_node.node_key}.payload.tracking_no",
            }
            if order_node.depends_on or order_node.input_sources:
                raise R2RuntimeError("DAG_TOPOLOGY_INVALID")
            if tuple(logistics_node.depends_on) != (order_node.node_key,) or logistics_node.input_sources != expected_sources:
                raise R2RuntimeError("DAG_TOPOLOGY_INVALID")
        task_ids = {node.node_key: _safe_execution_id(f"r2_{node.node_key}_") for node in candidate.nodes}
        tasks: list[Task] = []
        allowed = {
            "order/read@v1": ("order-agent@v1", "order.result.v1"),
            "logistics/read@v1": ("logistics-agent@v1", "logistics.result.v1"),
        }
        for node in candidate.nodes:
            if node.capability_ref not in allowed:
                raise R2RuntimeError("DAG_CAPABILITY_NOT_ALLOWED")
            agent_ref, output_contract = allowed[node.capability_ref]
            depends = [task_ids[key] for key in node.depends_on if key in task_ids]
            if len(depends) != len(node.depends_on):
                raise R2RuntimeError("DAG_DEPENDENCY_MISSING")
            bindings = []
            for name, source in node.input_sources.items():
                parts = source.split(".")
                if len(parts) != 3 or parts[1] != "payload" or parts[0] not in node.depends_on:
                    raise R2RuntimeError("DAG_BINDING_INVALID")
                bindings.append(InputBinding(name=name, kind="result", source_task_id=task_ids[parts[0]], path=f"payload.{parts[2]}", required=True))
            tasks.append(Task(
                task_id=task_ids[node.node_key],
                plan_revision_id="pending",
                agent_ref=agent_ref,
                capability_refs=[node.capability_ref],
                depends_on=depends,
                input_bindings=bindings,
                output_contract=output_contract,
                failure_strategy="REPLAN_LOCAL" if node.capability_ref == "logistics/read@v1" and depends else "FAIL_RUN",
                side_effect="READ_ONLY",
                timeout_ms=5000,
            ))
        plan_id = f"r2_plan_{uuid4().hex}"
        tasks = [task.model_copy(update={"plan_revision_id": plan_id}) for task in tasks]
        draft = PlanDraft(
            draft_id=f"r2_draft_{uuid4().hex}",
            run_id=run_id,
            candidate_intent=CandidateIntent(intent=goal.goal_type.value, confidence=1.0, evidence_hash=sha256_json(goal.model_dump(mode="json")), source="r1.5_router"),
            tasks=tuple(tasks),
        )
        supervisor = M3Supervisor(registry=registry, manifests=self._manifests(), repository=repository, limits=PlanLimits(max_tasks=8, max_iterations=8, max_timeout_ms=30_000))
        return supervisor.activate_plan(
            draft,
            revision_reason="local_replan" if supersedes else "initial",
            supersedes_plan_revision_id=supersedes,
            version=version,
        )

    @staticmethod
    def _safe_response(code: str) -> str:
        return {
            "CLARIFICATION_REQUIRED": "请补充可验证的订单或物流信息。",
            "AUTH_IDENTITY_MISMATCH": "身份校验未通过，无法查询该物流事实。",
            "ORDER_NOT_FOUND": "未找到可验证的订单或物流记录。",
            "DATA_MISSING": "物流事实缺失，暂时无法形成完整回答。",
            "DATA_STALE": "物流事实已陈旧，需要更新后的数据。",
            "DATA_CONFLICT": "订单与物流事实存在冲突，需要进一步核验。",
            "INFRA_TIMEOUT": "物流查询超时，未将请求结果解释为成功。",
            "INFRA_UNAVAILABLE": "数据源暂时不可用。",
            "SAFE_STOP": "请求无法安全完成。",
        }.get(code, "请求无法安全完成。")

    @staticmethod
    def _result_artifact(result: Result) -> dict[str, Any]:
        """Project canonical Result into FreezeBundle-safe evidence fields."""
        row = result.model_dump(mode="json")
        usage = row.pop("usage", {}) or {}
        row["usage"] = {
            "input": int(usage.get("input_tokens", 0)),
            "output": int(usage.get("output_tokens", 0)),
            "available": bool(usage.get("input_tokens", 0) or usage.get("output_tokens", 0)),
        }
        return row

    @staticmethod
    def _topology_route(goal: CustomerGoalV1) -> str:
        return {
            BusinessIntent.ORDER_QUERY: "order_only",
            BusinessIntent.LOGISTICS_QUERY: "logistics_only",
            BusinessIntent.ORDER_AND_LOGISTICS: "order_to_logistics",
        }.get(goal.goal_type, "clarify")

    def run(self, *, user_id: str, goal: CustomerGoalV1, session_id: str | None = None, run_id: str | None = None) -> R2RunResult:
        if not str(user_id or "").strip():
            raise R2RuntimeError("AUTH_REQUIRED")
        sid = session_id or _safe_execution_id("r2_session_")
        rid = run_id or _safe_execution_id("r2_run_")
        root = self.artifact_root / rid
        root.mkdir(parents=True, exist_ok=True)
        runtime_repo = M2Repository(str(root / "runtime.db"))
        world = WorldSnapshot(world_fixture_ref="r2-order-logistics", world_template_version="r2.world.v1", seed=0, scene_clock=datetime(2026, 1, 1, tzinfo=timezone.utc), entities=[{"entity_type": "dataset", "entity_id": "dev-order-logistics-dag-v1"}])
        runtime_repo.create_session(sid, "r2-owner-" + hashlib.sha256(str(user_id).encode()).hexdigest()[:16])
        runtime_repo.create_run(Run(run_id=rid, session_id=sid, initial_world_hash=world.snapshot_hash, shared_state={"mode": self.mode.value}))
        versions = self._version_tuple()
        trace = TraceRecorder(run_id=rid, session_id=sid, repository=runtime_repo, scene_clock=world.scene_clock, version_tuple=versions, failure_script=self.failure_script_contract)
        trace.record("RUN_CREATED", payload={"scenario_id": "dev-order-logistics-dag-v1", "mode": self.mode.value}, actor="runtime")
        model_calls = 0
        tool_calls = 0
        plans: list[PlanRevision] = []
        results: list[Result] = []
        response = "请求无法安全完成。"
        terminal_code = "SAFE_STOP"
        terminal_status = "FAILED"
        replan_count = 0
        provider = self.provider
        if provider is None and self.mode is ExecutionMode.LIVE:
            provider = ChatOpenAIStructuredProvider(self.config)
        client = ModelClient(self.config, provider=provider)

        def finish(route: str) -> R2RunResult:
            trace.drain_jsonl(root / "trace.jsonl")
            bundle = trace.freeze_bundle(world_snapshot=world, final_response=response, final_fingerprint=sha256_json({"world": world.snapshot_hash, "status": terminal_status, "code": terminal_code, "result_hashes": [item.payload_hash for item in results]}), failure_script=self.failure_script_contract, run_context={"mode": self.mode.value, "route": route, "source_version": versions.tool_impl, "replan_count": replan_count}, version_tuple=versions, plan_revisions=[p.model_dump(mode="json") for p in plans], results=[self._result_artifact(r) for r in results])
            verification = verify_bundle(bundle, repository=runtime_repo)
            runtime_repo.close()
            return R2RunResult(rid, sid, route, terminal_status, terminal_code, response, tuple(plans), tuple(results), str(root / "trace.jsonl"), bundle, verification, model_calls, tool_calls, replan_count)

        try:
            nonlocal_model = {"count": 0}
            goal = normalize_customer_goal(goal)
            trace.record("STATE_DELTA", payload={"goal_hash": sha256_json(goal.model_dump(mode="json")), "source": "r1.5_router"}, actor="runtime")
            route = self._topology_route(goal)
            if goal.needs_clarification:
                terminal_code = "CLARIFICATION_REQUIRED"
                terminal_status = "NEEDS_CLARIFICATION"
                response = self._safe_response(terminal_code)
                trace.record("ERROR", payload={"code": terminal_code, "layer": "contract", "retryable": False}, actor="runtime")
                return finish(route)

            planner_prompt = (
                "r2.dag_candidate_schema=r2.dag-candidate.v1\n"
                "allowed_capabilities=[order/read@v1,logistics/read@v1]\n"
                "ORDER_QUERY requires exactly one order/read@v1 root node.\n"
                "LOGISTICS_QUERY requires exactly one logistics/read@v1 root node.\n"
                "ORDER_AND_LOGISTICS requires exactly order/read@v1 -> logistics/read@v1.\n"
                "Every root node must use depends_on=[] and input_sources={}; root arguments come from CustomerGoalV1 and literal entity values must never be copied into input_sources.\n"
                "dependent_logistics_result_bindings={carrier_code:order.payload.carrier_code,tracking_no:order.payload.tracking_no}\n"
                "planner_must_not_emit_outcomes_or_recovery_nodes\n"
                "goal_type=" + goal.goal_type.value + "\n"
                + json.dumps(goal.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
            )
            trace.record("MODEL_CALLED", payload={"model_ref": versions.model, "prompt_version": "r2.dag-candidate.v1", "input_hash": sha256_json(goal.model_dump(mode="json"))}, actor="runtime")
            nonlocal_model["count"] += 1
            candidate, plan_usage, plan_latency = client.complete(planner_prompt, DagCandidateV1)
            model_calls = nonlocal_model["count"]
            trace.record("MODEL_RETURNED", payload={"output_hash": sha256_json(candidate.model_dump(mode="json")), "usage": {"input": plan_usage.input_tokens, "output": plan_usage.output_tokens, "available": plan_usage.available}, "latency_ms": plan_latency}, actor="runtime")
            plan = self._plan_from_candidate(run_id=rid, goal=goal, candidate=candidate, repository=runtime_repo)
            plans.append(plan)
            trace.record("PLAN_CREATED", payload={"plan_revision_id": plan.plan_revision_id, "task_count": len(plan.tasks), "supersedes": None}, plan_revision_id=plan.plan_revision_id, actor="runtime")
            trace.record("PLAN_VALIDATED", payload={"plan_revision_id": plan.plan_revision_id, "acyclic": True, "binding_count": sum(len(task.input_bindings) for task in plan.tasks)}, plan_revision_id=plan.plan_revision_id, actor="runtime")
            registry = self._registry()
            reducer = M1Supervisor(runtime_repo)
            task_results: dict[str, Result] = {}
            for task in plan.tasks:
                for dep in task.depends_on:
                    if dep not in task_results or task_results[dep].status is not ResultStatus.SUCCEEDED:
                        attempt_id = f"r2_attempt_{uuid4().hex}"
                        runtime_repo.create_attempt(TaskAttempt(attempt_id=attempt_id, run_id=rid, plan_revision_id=plan.plan_revision_id, task_id=task.task_id, agent_ref=task.agent_ref, attempt_no=runtime_repo.next_attempt_no(task.task_id), input_hash=sha256_json({"blocked_by": dep})))
                        reducer.start_attempt(attempt_id)
                        blocked = Result(result_id=f"r2_result_{uuid4().hex}", run_id=rid, plan_revision_id=plan.plan_revision_id, task_id=task.task_id, attempt_id=attempt_id, status=ResultStatus.BLOCKED, output_contract=task.output_contract, payload=None, business_code="DEPENDENCY_FAILED")
                        blocked = blocked.model_copy(update={"evidence_refs": [blocked.payload_hash]})
                        runtime_repo.append_result(blocked, event_payload={"business_code": "DEPENDENCY_FAILED"})
                        results.append(blocked); task_results[task.task_id] = blocked
                        terminal_code = "DEPENDENCY_FAILED"; terminal_status = "FAILED"; response = self._safe_response("SAFE_STOP")
                        return finish(route)
                reducer.transition_task(task.task_id, TaskStatus.READY, plan_revision_id=plan.plan_revision_id)
                reducer.transition_task(task.task_id, TaskStatus.RUNNING, plan_revision_id=plan.plan_revision_id)
                task_map = {item.task_id: item for item in plan.tasks}
                binding_resolver = BindingResolver(task_map)
                bindings: dict[str, Any] = {}
                for input_binding in task.input_bindings:
                    if input_binding.kind != "result":
                        continue
                    source_task = task_map[str(input_binding.source_task_id)]
                    binding = ResultBinding(name=input_binding.name, source_task_id=str(input_binding.source_task_id), path=input_binding.path, expected_contract=source_task.output_contract, expected_type="Any")
                    bindings[input_binding.name] = binding_resolver.resolve_result(binding, target_task_id=task.task_id, run_id=rid, plan_revision_id=plan.plan_revision_id, results=task_results)
                if task.agent_ref == "order-agent@v1":
                    entity_map = goal.entity_map
                    if not entity_map.get("order_id") or not entity_map.get("phone_last4"):
                        raise R2RuntimeError("CLARIFICATION_REQUIRED")
                    args = {"order_id": entity_map["order_id"], "phone_last4": entity_map["phone_last4"]}
                    port = OrderAgent(registry=registry)
                else:
                    if task.depends_on and (not bindings.get("carrier_code") or not bindings.get("tracking_no")):
                        raise R2RuntimeError("DATA_MISSING")
                    entity_map = goal.entity_map
                    if not task.depends_on and not entity_map.get("phone_last4"):
                        raise R2RuntimeError("CLARIFICATION_REQUIRED")
                    carrier = bindings.get("carrier_code") if task.depends_on else entity_map.get("carrier_code")
                    tracking = bindings.get("tracking_no") if task.depends_on else entity_map.get("tracking_no")
                    if not carrier or not tracking:
                        raise R2RuntimeError("DATA_MISSING")
                    args = {"carrier_code": carrier, "tracking_no": tracking}
                    if entity_map.get("phone_last4"):
                        args["phone_last4"] = entity_map["phone_last4"]
                    port = LogisticsAgent(registry=registry)
                attempt_id = f"r2_attempt_{uuid4().hex}"
                runtime_repo.create_attempt(TaskAttempt(attempt_id=attempt_id, run_id=rid, plan_revision_id=plan.plan_revision_id, task_id=task.task_id, agent_ref=task.agent_ref, attempt_no=runtime_repo.next_attempt_no(task.task_id), input_hash=sha256_json({"task": task.task_id, "args_keys": sorted(args)})))
                reducer.start_attempt(attempt_id)
                trace.record("ATTEMPT_STARTED", payload={"logical_call_no": tool_calls + 1, "physical_attempt_no": 1}, plan_revision_id=plan.plan_revision_id, task_id=task.task_id, attempt_id=attempt_id, actor="runtime")
                trace.record("TOOL_CALLED", payload={"tool_ref": port.tool_ref, "capability_ref": task.capability_refs[0], "safe_args_hash": sha256_json(args), "implementation_mode": registry.get(port.tool_ref).implementation_mode}, plan_revision_id=plan.plan_revision_id, task_id=task.task_id, attempt_id=attempt_id, actor="tool")
                tool_calls += 1
                typed = port.invoke(args, context=self._trusted_context(user_id=user_id, session_id=sid, run_id=rid, plan=plan, task=task, attempt_id=attempt_id, agent_ref=port.agent_ref, capability_ref=registry.get(port.tool_ref).capability_ref))
                trace.record("TOOL_RETURNED", payload={"tool_ref": port.tool_ref, "ok": typed.ok, "error_code": typed.error_code, "output_hash": typed.payload_hash}, plan_revision_id=plan.plan_revision_id, task_id=task.task_id, attempt_id=attempt_id, actor="tool")
                result = Result(result_id=f"r2_result_{uuid4().hex}", run_id=rid, plan_revision_id=plan.plan_revision_id, task_id=task.task_id, attempt_id=attempt_id, status=ResultStatus.SUCCEEDED if typed.ok else ResultStatus.FAILED, output_contract=task.output_contract, payload=dict(typed.payload) if typed.ok and isinstance(typed.payload, Mapping) else None, business_code=None if typed.ok else typed.error_code, evidence_refs=[typed.payload_hash], claim_refs=[])
                runtime_repo.append_result_with_event(result, event_payload={"business_code": result.business_code} if result.business_code else None)
                results.append(result); task_results[task.task_id] = result
                terminal_code = "OK" if typed.ok else (typed.error_code or "TOOL_EXECUTION_FAILED")
                if not typed.ok:
                    if self.allow_replan and terminal_code == "INFRA_TIMEOUT" and task.failure_strategy == "REPLAN_LOCAL" and replan_count == 0:
                        replan_count = 1
                        old_plan = plan
                        # Plan revisions are immutable in the canonical M2
                        # store.  M3Supervisor.replan creates the successor
                        # and records its supersedes reference; do not mutate
                        # the persisted predecessor in place.
                        plan = M3Supervisor(registry=registry, manifests=self._manifests(), repository=runtime_repo).replan(old_plan, tasks=old_plan.tasks, reason="local_replan")
                        plans.append(plan)
                        trace.record("PLAN_CREATED", payload={"plan_revision_id": plan.plan_revision_id, "task_count": len(plan.tasks), "supersedes": old_plan.plan_revision_id}, plan_revision_id=plan.plan_revision_id, actor="runtime")
                        trace.record("PLAN_VALIDATED", payload={"plan_revision_id": plan.plan_revision_id, "acyclic": True, "binding_count": sum(len(task.input_bindings) for task in plan.tasks)}, plan_revision_id=plan.plan_revision_id, actor="runtime")
                        # Re-enter the same loop with the bounded failure consumed.
                        return self._run_replan_continuation(user_id=user_id, goal=goal, sid=sid, rid=rid, runtime_repo=runtime_repo, world=world, versions=versions, trace=trace, plans=plans, results=results, route=route, model_calls=model_calls, tool_calls=tool_calls, replan_count=replan_count, registry=registry, plan=plan, reducer=reducer, response=response)
                    terminal_status = "BLOCKED" if terminal_code == "DATA_CONFLICT" else "FAILED"; response = self._safe_response(terminal_code)
                    reducer.transition_task(task.task_id, TaskStatus.FAILED, plan_revision_id=plan.plan_revision_id)
                    return finish(route)
                reducer.transition_task(task.task_id, TaskStatus.SUCCEEDED, plan_revision_id=plan.plan_revision_id)
            terminal_status = "SUCCEEDED"
            terminal_code = "OK"
            response = self._render_success(route, results)
            return finish(route)
        except R2RuntimeError as exc:
            if runtime_repo.conn.execute("SELECT 1 FROM trace_manifests WHERE run_id=?", (rid,)).fetchone():
                runtime_repo.close()
                raise
            terminal_code = exc.code; terminal_status = "NEEDS_CLARIFICATION" if exc.code == "CLARIFICATION_REQUIRED" else "FAILED"; response = self._safe_response(exc.code)
            trace.record("ERROR", payload={"code": exc.code, "layer": "runtime", "retryable": False}, actor="runtime")
            return finish("clarify" if exc.code == "CLARIFICATION_REQUIRED" else "safe_stop")
        except Exception as exc:
            if runtime_repo.conn.execute("SELECT 1 FROM trace_manifests WHERE run_id=?", (rid,)).fetchone():
                runtime_repo.close()
                raise
            terminal_code = "R2_RUNTIME_ERROR"; terminal_status = "FAILED"; response = self._safe_response("SAFE_STOP")
            trace.record("ERROR", payload={"code": terminal_code, "error_class": type(exc).__name__}, actor="runtime")
            return finish("safe_stop")

    def _render_success(self, route: str, results: list[Result]) -> str:
        def safe(value: Any, allowed: set[str]) -> str:
            text = str(value or "").upper()
            return text if text in allowed else "UNKNOWN"
        order_statuses = {"PAID", "UNPAID", "CANCELLED", "SHIPPED", "DELIVERED", "COMPLETED", "REFUNDED", "UNKNOWN"}
        delivery_states = {"DELIVERED", "IN_TRANSIT", "OUT_FOR_DELIVERY", "DELAYED", "UNKNOWN"}
        order_payload = next((item.payload for item in results if item.output_contract == "order.result.v1" and isinstance(item.payload, Mapping)), None)
        logistics_payload = next((item.payload for item in results if item.output_contract == "logistics.result.v1" and isinstance(item.payload, Mapping)), None)
        if route == "order_only":
            return f"订单状态已核验：{safe(order_payload.get('order_status') if order_payload else None, order_statuses)}。"
        if route == "logistics_only":
            return f"物流状态已核验：{safe(logistics_payload.get('delivery_state') if logistics_payload else None, delivery_states)}。"
        if any(result.business_code in {"DATA_MISSING", "DATA_STALE", "DATA_CONFLICT"} for result in results):
            return "物流事实不足以形成安全结论。"
        return f"订单与物流状态已核验：订单 {safe(order_payload.get('order_status') if order_payload else None, order_statuses)}；物流 {safe(logistics_payload.get('delivery_state') if logistics_payload else None, delivery_states)}。"

    def _run_replan_continuation(self, **kwargs: Any) -> R2RunResult:
        # Replan continuation is intentionally bounded: it executes only the
        # new plan's tasks with the consumed one-shot fault and then freezes.
        runtime_repo: M2Repository = kwargs["runtime_repo"]
        trace: TraceRecorder = kwargs["trace"]
        plans: list[PlanRevision] = kwargs["plans"]
        results: list[Result] = kwargs["results"]
        plan: PlanRevision = kwargs["plan"]
        rid: str = kwargs["rid"]; sid: str = kwargs["sid"]; route: str = kwargs["route"]
        versions: VersionTuple = kwargs["versions"]; world: WorldSnapshot = kwargs["world"]
        registry: M3Registry = kwargs["registry"]
        reducer = M1Supervisor(runtime_repo)
        # The replan repeats the trusted order read, then derives logistics args
        # from its new ResultBinding.  The one-shot fault has already fired.
        goal: CustomerGoalV1 = kwargs["goal"]
        task_results: dict[str, Result] = {}
        terminal_code = "OK"; terminal_status = "SUCCEEDED"; response = "请求无法安全完成。"; tool_calls = int(kwargs.get("tool_calls", 0))
        task_map = {item.task_id: item for item in plan.tasks}
        binding_resolver = BindingResolver(task_map)
        for task in plan.tasks:
            reducer.transition_task(task.task_id, TaskStatus.READY, plan_revision_id=plan.plan_revision_id)
            reducer.transition_task(task.task_id, TaskStatus.RUNNING, plan_revision_id=plan.plan_revision_id)
            bindings: dict[str, Any] = {}
            for input_binding in task.input_bindings:
                if input_binding.kind != "result":
                    continue
                source_task = task_map[str(input_binding.source_task_id)]
                binding = ResultBinding(name=input_binding.name, source_task_id=str(input_binding.source_task_id), path=input_binding.path, expected_contract=source_task.output_contract, expected_type="Any")
                bindings[input_binding.name] = binding_resolver.resolve_result(binding, target_task_id=task.task_id, run_id=rid, plan_revision_id=plan.plan_revision_id, results=task_results)
            if task.agent_ref == "order-agent@v1":
                entity_map = goal.entity_map
                if not entity_map.get("order_id") or not entity_map.get("phone_last4"):
                    terminal_code = "CLARIFICATION_REQUIRED"; terminal_status = "NEEDS_CLARIFICATION"; response = self._safe_response(terminal_code); break
                args = {"order_id": entity_map["order_id"], "phone_last4": entity_map["phone_last4"]}; port = OrderAgent(registry=registry)
            else:
                if task.depends_on and (not bindings.get("carrier_code") or not bindings.get("tracking_no")):
                    terminal_code = "DATA_MISSING"; terminal_status = "FAILED"; response = self._safe_response(terminal_code); break
                entity_map = goal.entity_map
                if not task.depends_on and not entity_map.get("phone_last4"):
                    terminal_code = "CLARIFICATION_REQUIRED"; terminal_status = "NEEDS_CLARIFICATION"; response = self._safe_response(terminal_code); break
                args = {"carrier_code": bindings.get("carrier_code") if task.depends_on else entity_map.get("carrier_code"), "tracking_no": bindings.get("tracking_no") if task.depends_on else entity_map.get("tracking_no")}; port = LogisticsAgent(registry=registry)
                if entity_map.get("phone_last4"):
                    args["phone_last4"] = entity_map["phone_last4"]
            attempt_id = f"r2_attempt_{uuid4().hex}"
            runtime_repo.create_attempt(TaskAttempt(attempt_id=attempt_id, run_id=rid, plan_revision_id=plan.plan_revision_id, task_id=task.task_id, agent_ref=task.agent_ref, attempt_no=runtime_repo.next_attempt_no(task.task_id), input_hash=sha256_json({"task": task.task_id, "args_keys": sorted(args)})))
            reducer.start_attempt(attempt_id)
            trace.record("ATTEMPT_STARTED", payload={"logical_call_no": tool_calls + 1, "physical_attempt_no": 1}, plan_revision_id=plan.plan_revision_id, task_id=task.task_id, attempt_id=attempt_id, actor="runtime")
            trace.record("TOOL_CALLED", payload={"tool_ref": port.tool_ref, "capability_ref": task.capability_refs[0], "safe_args_hash": sha256_json(args), "implementation_mode": registry.get(port.tool_ref).implementation_mode}, plan_revision_id=plan.plan_revision_id, task_id=task.task_id, attempt_id=attempt_id, actor="tool")
            tool_calls += 1
            typed = port.invoke(args, context=self._trusted_context(user_id=kwargs["user_id"], session_id=sid, run_id=rid, plan=plan, task=task, attempt_id=attempt_id, agent_ref=port.agent_ref, capability_ref=registry.get(port.tool_ref).capability_ref))
            trace.record("TOOL_RETURNED", payload={"tool_ref": port.tool_ref, "ok": typed.ok, "error_code": typed.error_code, "output_hash": typed.payload_hash}, plan_revision_id=plan.plan_revision_id, task_id=task.task_id, attempt_id=attempt_id, actor="tool")
            result = Result(result_id=f"r2_result_{uuid4().hex}", run_id=rid, plan_revision_id=plan.plan_revision_id, task_id=task.task_id, attempt_id=attempt_id, status=ResultStatus.SUCCEEDED if typed.ok else ResultStatus.FAILED, output_contract=task.output_contract, payload=dict(typed.payload) if typed.ok and isinstance(typed.payload, Mapping) else None, business_code=None if typed.ok else typed.error_code, evidence_refs=[typed.payload_hash])
            runtime_repo.append_result(result)
            results.append(result); task_results[task.task_id] = result
            if not typed.ok:
                terminal_code = typed.error_code or "TOOL_EXECUTION_FAILED"; terminal_status = "BLOCKED" if terminal_code == "DATA_CONFLICT" else "FAILED"; response = self._safe_response(terminal_code); reducer.transition_task(task.task_id, TaskStatus.FAILED, plan_revision_id=plan.plan_revision_id); break
            reducer.transition_task(task.task_id, TaskStatus.SUCCEEDED, plan_revision_id=plan.plan_revision_id)
        if terminal_status == "SUCCEEDED":
            response = self._render_success(route, results)
        trace.record("RESULT_WRITTEN", payload={"status": terminal_status, "code": terminal_code, "result_count": len(results)}, actor="runtime")
        root = self.artifact_root / rid
        trace.drain_jsonl(root / "trace.jsonl")
        bundle = trace.freeze_bundle(world_snapshot=world, final_response=response, final_fingerprint=sha256_json({"world": world.snapshot_hash, "status": terminal_status, "code": terminal_code, "result_hashes": [item.payload_hash for item in results]}), failure_script=self.failure_script_contract, run_context={"mode": self.mode.value, "route": route, "source_version": versions.tool_impl, "replan_count": 1}, version_tuple=versions, plan_revisions=[p.model_dump(mode="json") for p in plans], results=[self._result_artifact(r) for r in results])
        verification = verify_bundle(bundle, repository=runtime_repo)
        runtime_repo.close()
        return R2RunResult(rid, sid, route, terminal_status, terminal_code, response, tuple(plans), tuple(results), str(root / "trace.jsonl"), bundle, verification, int(kwargs.get("model_calls", 0)), tool_calls, 1)


__all__ = ["DagCandidateV1", "DagNodeV1", "DeterministicR2Provider", "OrderLogisticsRepository", "R2OrderLogisticsRuntime", "R2RunResult", "R2RuntimeError"]
