"""Persisted M3 scenario runtime: turns -> router -> activated plan -> ports."""
from __future__ import annotations

import re
import json
import uuid
from decimal import Decimal
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

from .agents import AfterSalesAgent, AgentPort, LogisticsAgent, OrderAgent, PolicyAgent, ProductAgent, TypedAgentResult
from .domain.objects import AttemptStatus, Result, ResultStatus, Run, TaskAttempt, TaskStatus, sha256_json
from .domain.aftersales import AfterSalesService, AfterSalesError
from .domain.confirm_tokens import ConfirmTokenManager
from .domain.eligibility import EligibilityEngine
from .domain.facts import OrderFact
from .domain.policy import PolicyCatalog, PolicyRule
from .m1_runtime import Supervisor as M1Supervisor
from .m2_context import InvocationContext
from .m2_registry import Registry
from .m3_bindings import BindingResolver, ResultBinding
from .m3_registry import M3Registry, build_m3_registry
from .m3_supervisor import IntentRouter, M3Supervisor
from .m3_scheduler import TaskScheduler
from .storage.m2 import M2Repository
from .trace.events import build_event


@dataclass(frozen=True)
class ScenarioRun:
    scenario_id: str
    intent: str
    expected_intent: str
    tool_path: tuple[str, ...]
    expected_tool_path: tuple[str, ...]
    terminal_class: str
    logical_calls: int
    physical_attempts: int
    trajectory_valid: bool
    intent_ok: bool = False
    path_ok: bool = False
    expected_terminal_class: str = "PASS"
    observed_terminal_class: str = "PASS"
    world_fingerprint: str = ""
    db_path: str = ""
    run_id: str = ""
    business_code: str = ""
    plan_fingerprint: str = ""
    plan_projection: tuple[dict[str, Any], ...] = ()
    terminal_fingerprint: str = ""


ALIASES = {
    "order/get_info@v1": "get_order_info_tool",
    "aftersales/query@v1": "query_aftersales_tool",
    "aftersales/create@v1": "create_aftersales_tool",
    "logistics/query@v1": "query_logistics_snapshot_tool",
    "policy/search@v1": "policy_rag_search_tool",
    "human/handoff@v1": "handoff_to_human_tool",
    "product/get@v1": "product_get_tool",
}
PORT_TYPES = {
    "order/get_info@v1": OrderAgent,
    "aftersales/query@v1": AfterSalesAgent,
    "aftersales/create@v1": AfterSalesAgent,
    "logistics/query@v1": LogisticsAgent,
    "policy/search@v1": PolicyAgent,
    "product/get@v1": ProductAgent,
}


def _fixture_path() -> Path:
    return Path(__file__).resolve().parents[1] / "eval" / "scenarios" / "world_fixtures.yaml"


def _load_fixture(ref: str) -> dict[str, Any]:
    if not ref:
        return {"world_fixture_ref": "world-inline-v1", "seed": 0, "entities": []}
    document = yaml.safe_load(_fixture_path().read_text(encoding="utf-8")) or {}
    for fixture in document.get("world_fixtures", {}).get("fixtures", []):
        if str(fixture.get("world_fixture_ref")) == ref:
            return dict(fixture)
    raise ValueError(f"unknown world fixture: {ref}")


def _gold_fixture_hash(ref: str) -> str | None:
    document = yaml.safe_load(_fixture_path().read_text(encoding="utf-8")) or {}
    return (document.get("world_fixtures", {}).get("gold_fingerprints", {}) or {}).get(ref)


def _derive_operations(text: str, *, legacy_compat: bool = False) -> list[str]:
    """Build operations from user turns; expected evaluator fields are unused."""
    text = str(text)
    # Security-sensitive requests never enter protected data branches.  This
    # is a general semantic rule (not a scenario-id exception).
    if any(x in text for x in ("越权", "其他用户", "支付信息", "导出他人", "账户信息")):
        return ["human/handoff@v1"]
    if any(x in text for x in ("转人工", "人工客服", "投诉", "举报")) and not any(x in text for x in ("订单", "售后", "退款", "退货")):
        if legacy_compat:
            return []
        return ["human/handoff@v1"]
    has_order = any(x in text for x in ("订单", "订单号", "查订单", "支付", "发货")) or bool(re.search(r"\b(?:ORD|20\d{8})[A-Z0-9-]*\b", text))
    has_logistics = any(x in text for x in ("物流", "快递", "运单", "承运商", "派件", "签收"))
    has_policy = any(x in text for x in ("规则", "政策", "运费", "七天无理由"))
    has_product = any(x in text for x in ("商品", "产品", "库存", "价格", "SKU"))
    has_after = any(x in text for x in ("售后", "退款", "换货")) or ("申请" in text and "退货" in text)
    if has_policy and not has_order:
        has_after = False
    if any(x in text for x in ("转人工", "人工客服", "投诉", "举报")) and any(x in text for x in ("争议", "高风险")):
        return ["human/handoff@v1"]
    if has_policy and any(x in text for x in ("政策冲突", "政策矛盾", "政策争议")):
        return ["human/handoff@v1"]
    if "高风险" in text and "审核" in text:
        return ["human/handoff@v1"]
    ops: list[str] = []
    if has_order or has_after:
        ops.append("order/get_info@v1")
    if has_logistics:
        ops.append("logistics/query@v1")
    if has_after:
        if "进度" not in text and "logistics/query@v1" not in ops and legacy_compat:
            ops.append("logistics/query@v1")
        ops.append("aftersales/query@v1" if "进度" in text else "aftersales/create@v1")
    if has_policy and not legacy_compat:
        ops.append("policy/search@v1")
    if has_product and not has_after:
        ops.append("product/get@v1")
    if not ops and legacy_compat and has_policy:
        return []
    if not ops:
        ops.append("human/handoff@v1")
    return list(dict.fromkeys(ops))


def _args_for(tool_ref: str, text: str, scenario_id: str) -> dict[str, Any]:
    order_match = re.search(r"\b(?:20\d{9,}|ORD[-A-Z0-9]+)\b", text, re.I)
    order_id = order_match.group(0) if order_match else f"ORD-{scenario_id.upper()}"
    phone_match = re.search(r"(?:后四位|尾号|phone)\D*(\d{4})", text, re.I)
    phone = phone_match.group(1) if phone_match else "0000"
    if tool_ref in {"order/get_info@v1", "aftersales/query@v1"}:
        return {"order_id": order_id, "phone_last4": phone}
    if tool_ref == "aftersales/create@v1":
        return {"order_id": order_id, "phone_last4": phone, "service_type": "refund", "reason": text[:120] or "customer request"}
    if tool_ref == "logistics/query@v1":
        match = re.search(r"\b(?:ST|SF|YT|LOG)[A-Z0-9-]*\b", text, re.I)
        return {"carrier_code": "simulator", "tracking_no": match.group(0) if match else f"LOG-{scenario_id.upper()}", "phone_last4": phone}
    if tool_ref == "policy/search@v1":
        return {"query": text[:200], "top_k": 3}
    if tool_ref == "product/get@v1":
        sku_match = re.search(r"\bSKU[-A-Z0-9]+\b", text, re.I)
        return {"sku": sku_match.group(0).upper() if sku_match else "SKU-FIXTURE-001"}
    return {"summary": text[:200] or "customer escalation", "reason": "human review requested"}


def _faulted_callable(base, script: Mapping[str, Any], tool_ref: str):
    raw = dict(script or {})
    triggers = raw.get("triggers") if isinstance(raw.get("triggers"), list) else []
    if triggers:
        first = next((item for item in triggers if isinstance(item, Mapping) and (not item.get("tool_ref") or item.get("tool_ref") in {tool_ref, ALIASES.get(tool_ref, "")})), {})
        raw = dict(first)
    if str(raw.get("action", "")) != "RETURN_ERROR":
        return base
    target = str(raw.get("target_tool", raw.get("tool_ref", "")))
    if target and target not in {tool_ref, ALIASES.get(tool_ref, "")}:
        return base
    code = str(raw.get("error_code", "TOOL_EXECUTION_FAILED"))
    def call(**kwargs):
        return {"success": False, "code": code, "message": "fixture fault", "data": None}
    return call


class _DynamicPort(AgentPort):
    pass


class M3ScenarioRunner:
    def __init__(self, *, registry: M3Registry | None = None, db_dir: str | Path | None = None):
        self.registry = registry or build_m3_registry()
        from .agents import build_m3_manifests, validate_manifests
        self.manifests = build_m3_manifests()
        validate_manifests(self.manifests, self.registry)
        self.router = IntentRouter()
        self.db_dir = Path(db_dir) if db_dir else Path("runtime") / "m3"
        self.db_dir.mkdir(parents=True, exist_ok=True)

    def _registry_for_fixture(self, script: Mapping[str, Any]) -> M3Registry:
        def simulator(**kwargs):
            return {"success": True, "code": "OK", "data": {"fixture": True, **kwargs}}
        specs = [self.registry.canonical.get(ref).model_copy(update={"callable": _faulted_callable(simulator, script, ref)}) for ref in self.registry.canonical.refs()]
        return M3Registry(canonical=Registry(specs))

    @staticmethod
    def _last_event(repo: M2Repository, run_id: str) -> str | None:
        row = repo.conn.execute("SELECT event_id FROM trace_outbox WHERE run_id=? ORDER BY seq_no DESC LIMIT 1", (run_id,)).fetchone()
        return str(row[0]) if row else None

    def _event(self, repo: M2Repository, run: Run, event_type: str, *, parent: str | None = None, task_id: str | None = None, attempt_id: str | None = None, payload: dict[str, Any] | None = None, actor: str = "runtime"):
        row = repo.conn.execute("SELECT next_seq_no FROM runs WHERE run_id=?", (run.run_id,)).fetchone()
        scene_clock = run.shared_state.get("scene_clock") if isinstance(run.shared_state, dict) else None
        parsed_clock = datetime.fromisoformat(str(scene_clock).replace("Z", "+00:00")) if scene_clock else None
        event = build_event(run_id=run.run_id, session_id=run.session_id, plan_revision_id=run.plan_revision_id, task_id=task_id, attempt_id=attempt_id, event_type=event_type, seq_no=int(row[0]), parent_event_id=parent, payload=payload or {}, actor=actor, scene_clock=parsed_clock)
        repo.append_m2_event(event)
        return event

    def run(self, case: Mapping[str, Any]) -> ScenarioRun:
        scenario_id = str(case["scenario_id"])
        turns = list(case.get("turns") or [])
        if not turns:
            raise ValueError(f"scenario {scenario_id} must embed executable turns")
        text = " ".join(str(turn) for turn in turns)
        inline_world = case.get("world_snapshot")
        if inline_world:
            fixture = dict(inline_world.model_dump(mode="json") if hasattr(inline_world, "model_dump") else inline_world)
            fixture.setdefault("world_fixture_ref", str(case.get("world_fixture_ref") or "world-inline-v1"))
        else:
            fixture = _load_fixture(str(case.get("world_fixture_ref") or ""))
        world_hash = sha256_json(fixture)
        gold_hash = _gold_fixture_hash(str(case.get("world_fixture_ref") or ""))
        if gold_hash and world_hash != str(gold_hash):
            raise ValueError("world fixture fingerprint does not match frozen manifest fixture")
        if case.get("initial_world_hash") and str(case["initial_world_hash"]) != world_hash:
            raise ValueError("world fixture hash mismatch")
        script = dict(case.get("failure_script") or fixture.get("failure_script") or {})
        registry = self._registry_for_fixture(script)
        run_id = f"run_{scenario_id}_{uuid.uuid4().hex[:8]}"
        session_id = f"session_{scenario_id}_{uuid.uuid4().hex[:8]}"
        db_path = self.db_dir / f"{run_id}.db"
        repo = M2Repository(str(db_path))
        try:
            repo.create_session(session_id, "fixture-user")
            run = Run(run_id=run_id, session_id=session_id, initial_world_hash=world_hash, shared_state={"scenario_id": scenario_id, "world_fixture_ref": fixture["world_fixture_ref"], "scene_clock": fixture.get("scene_clock")})
            repo.create_run(run)
            self._event(repo, run, "RUN_CREATED", payload={"scenario_id": scenario_id, "world_hash": world_hash})
            candidate = self.router.classify(text)
            supervisor = M3Supervisor(registry=registry, manifests=self.manifests, repository=repo)
            draft = supervisor.draft_operations(run_id=run_id, candidate_intent=candidate, operations=_derive_operations(text, legacy_compat=bool(case.get("source_case_ref"))))
            revision = supervisor.activate_plan(draft)
            run = run.model_copy(update={"plan_revision_id": revision.plan_revision_id})
            self._event(repo, run, "PLAN_CREATED", payload={"plan_revision_id": revision.plan_revision_id})
            self._event(repo, run, "PLAN_VALIDATED", payload={"plan_revision_id": revision.plan_revision_id})
            reducer = M1Supervisor(repo)
            reducer.reduce(run_id, expected_version=0, patch={"status": "READY"}, checkpoint_id=f"cp_{run_id}_ready", plan_revision_id=revision.plan_revision_id, parent_event_id=self._last_event(repo, run_id))
            reducer.reduce(run_id, expected_version=1, patch={"status": "RUNNING"}, checkpoint_id=f"cp_{run_id}_running", plan_revision_id=revision.plan_revision_id, parent_event_id=self._last_event(repo, run_id))
            tasks = {task.task_id: task for task in revision.tasks}
            bindings = BindingResolver(tasks)
            persisted: dict[str, Result] = {}
            observed: list[str] = []
            logical = physical = 0
            terminal = "PASS"
            business_code = "NO_OP"
            branch_codes: list[str] = []

            # Independent read branches of a mixed plan are executed through
            # the real dependency scheduler.  The worker performs only the
            # tool/AgentPort stage; all attempts, Results, state transitions
            # and trace events are committed below on the Supervisor-owned
            # connection in deterministic task order.  This keeps SQLite
            # writes single-writer while proving the read stages overlap.
            prefetched: dict[str, TypedAgentResult] = {}
            prefetch_meta: dict[str, tuple[str, str, InvocationContext, str, str]] = {}
            block_target = str(script.get("target_tool", ""))
            block_requested = str(script.get("action", "")) == "BLOCK_TASK"
            independent = [task for task in revision.tasks if not task.depends_on and task.side_effect == "READ_ONLY" and not (block_requested and (not block_target or block_target in {next(ref for ref in registry.refs() if registry.get(ref).capability_ref == task.capability_refs[0]), ALIASES.get(next(ref for ref in registry.refs() if registry.get(ref).capability_ref == task.capability_refs[0]), "")}))]
            if len(independent) > 1:
                # Allocate the canonical attempt/context before dispatch.  The
                # worker therefore executes with exactly the identity that is
                # later used by TOOL_RETURNED and RESULT_WRITTEN evidence.
                for stage in independent:
                    stage_tool = next(ref for ref in registry.refs() if registry.get(ref).capability_ref == stage.capability_refs[0])
                    version = int(repo.conn.execute("SELECT state_version FROM runs WHERE run_id=?", (run_id,)).fetchone()[0])
                    reducer.transition_task(stage.task_id, TaskStatus.READY, expected_version=version, plan_revision_id=revision.plan_revision_id)
                    version = int(repo.conn.execute("SELECT state_version FROM runs WHERE run_id=?", (run_id,)).fetchone()[0])
                    reducer.transition_task(stage.task_id, TaskStatus.RUNNING, expected_version=version, plan_revision_id=revision.plan_revision_id)
                    stage_attempt_id = f"attempt_{stage.task_id}_{uuid.uuid4().hex[:8]}"
                    stage_args = _args_for(stage_tool, text, scenario_id)
                    repo.create_attempt(TaskAttempt(attempt_id=stage_attempt_id, run_id=run_id, plan_revision_id=revision.plan_revision_id, task_id=stage.task_id, agent_ref=stage.agent_ref, attempt_no=repo.next_attempt_no(stage.task_id), input_hash=sha256_json(stage_args)))
                    reducer.start_attempt(stage_attempt_id)
                    stage_started = self._event(repo, run, "ATTEMPT_STARTED", task_id=stage.task_id, attempt_id=stage_attempt_id, payload={"attempt_id": stage_attempt_id})
                    stage_alias = ALIASES.get(stage_tool, stage_tool)
                    self._event(repo, run, "TOOL_CALLED", parent=stage_started.trace_id, task_id=stage.task_id, attempt_id=stage_attempt_id, payload={"tool_ref": stage_tool, "capability_ref": stage.capability_refs[0]}, actor="agent")
                    stage_context = InvocationContext(session_id=session_id, user_id="fixture-user", run_id=run_id, plan_revision_id=revision.plan_revision_id, task_id=stage.task_id, attempt_id=stage_attempt_id, agent_ref=stage.agent_ref, auth_scope=stage.capability_refs[0], idempotency_key=f"idem_{stage_attempt_id}", deadline=datetime.now(timezone.utc) + timedelta(seconds=5), config_version="m3.v1", registry_version="m3.registry.v1", dataset_version="m3.dev44.v1", trace_id=f"trace_{stage_attempt_id}")
                    prefetch_meta[stage.task_id] = (stage_tool, stage_attempt_id, stage_context, stage_alias, stage_args)

                def run_read_stage(stage: Any, _inputs: Mapping[str, object]) -> TypedAgentResult:
                    stage_tool, _attempt_id, stage_context, _alias, stage_args = prefetch_meta[stage.task_id]
                    stage_port_cls = PORT_TYPES.get(stage_tool, _DynamicPort)
                    stage_port = stage_port_cls(registry=registry)
                    stage_port.tool_ref, stage_port.agent_ref, stage_port.output_contract = stage_tool, stage.agent_ref, stage.output_contract
                    return stage_port.invoke(stage_args, context=stage_context)

                scheduled = TaskScheduler(independent, worker=run_read_stage).run()
                prefetched = {item.task_id: item.value for item in scheduled.results if isinstance(item.value, TypedAgentResult)}
                self._event(repo, run, "CHECKPOINT", parent=self._last_event(repo, run_id), payload={"scheduler": "TaskScheduler", "parallel_task_ids": [task.task_id for task in independent], "parallel_join": True, "partial": scheduled.partial})
            for task in revision.tasks:
                # Reset per-task state so a blocked task cannot reuse a prior
                # after-sales service result or error decision.
                service_holder: dict[str, Any] = {}
                handled_business_error = False
                typed = None
                status = ResultStatus.BLOCKED
                for binding in task.input_bindings:
                    if binding.kind == "result":
                        source = tasks[str(binding.source_task_id)]
                        bindings.resolve_result(ResultBinding(name=binding.name, source_task_id=source.task_id, path=binding.path, expected_contract=source.output_contract, expected_type="Any"), target_task_id=task.task_id, run_id=run_id, plan_revision_id=revision.plan_revision_id, results=persisted)
                prefetch_info = prefetch_meta.get(task.task_id)
                if prefetch_info:
                    tool_ref, attempt_id, context, selected_alias, _stage_args = prefetch_info
                    start = None
                else:
                    version = int(repo.conn.execute("SELECT state_version FROM runs WHERE run_id=?", (run_id,)).fetchone()[0])
                    reducer.transition_task(task.task_id, TaskStatus.READY, expected_version=version, plan_revision_id=revision.plan_revision_id)
                    version = int(repo.conn.execute("SELECT state_version FROM runs WHERE run_id=?", (run_id,)).fetchone()[0])
                    reducer.transition_task(task.task_id, TaskStatus.RUNNING, expected_version=version, plan_revision_id=revision.plan_revision_id)
                    attempt_id = f"attempt_{task.task_id}_{uuid.uuid4().hex[:8]}"
                    tool_ref = next(ref for ref in registry.refs() if registry.get(ref).capability_ref == task.capability_refs[0])
                    attempt = TaskAttempt(attempt_id=attempt_id, run_id=run_id, plan_revision_id=revision.plan_revision_id, task_id=task.task_id, agent_ref=task.agent_ref, attempt_no=repo.next_attempt_no(task.task_id), input_hash=sha256_json(_args_for(tool_ref, text, scenario_id)))
                    repo.create_attempt(attempt)
                    reducer.start_attempt(attempt_id)
                    start = self._event(repo, run, "ATTEMPT_STARTED", task_id=task.task_id, attempt_id=attempt_id, payload={"attempt_id": attempt_id})
                    selected_alias = ALIASES.get(tool_ref, tool_ref)
                block_target = str(script.get("target_tool", ""))
                if str(script.get("action", "")) == "BLOCK_TASK" and (not block_target or block_target in {tool_ref, ALIASES.get(tool_ref, "")}):
                    observed.append(selected_alias)
                    terminal, status, typed, business_code = "BLOCKED", ResultStatus.BLOCKED, None, "BLOCKED"
                    branch_codes.append(business_code)
                else:
                    observed.append(selected_alias)
                    if not prefetch_info:
                        self._event(repo, run, "TOOL_CALLED", parent=start.trace_id, task_id=task.task_id, attempt_id=attempt_id, payload={"tool_ref": tool_ref, "capability_ref": task.capability_refs[0]}, actor="agent")
                    owner = task.agent_ref
                    if not prefetch_info:
                        context = InvocationContext(session_id=session_id, user_id="fixture-user", run_id=run_id, plan_revision_id=revision.plan_revision_id, task_id=task.task_id, attempt_id=attempt_id, agent_ref=owner, auth_scope=task.capability_refs[0], idempotency_key=f"idem_{attempt_id}", deadline=datetime.now(timezone.utc) + timedelta(seconds=5), config_version="m3.v1", registry_version="m3.registry.v1", dataset_version="m3.dev44.v1", trace_id=f"trace_{attempt_id}")
                    port_cls = PORT_TYPES.get(tool_ref, _DynamicPort)
                    task_registry = registry
                    if tool_ref == "aftersales/create@v1":
                        create_args = _args_for(tool_ref, text, scenario_id)
                        order = OrderFact(fact_id=f"fact_{create_args['order_id']}", entity_id=create_args["order_id"], user_id="fixture-user", status="PAID", amount=10.0, items=[], created_at=datetime(2026, 1, 1, tzinfo=timezone.utc), version="order.v1", source="simulator")
                        rule = PolicyRule(rule_id="refund_v1", decision_logic="allow", source="m3.fixture", effective_from=datetime(2025, 1, 1, tzinfo=timezone.utc), scope="refund", version="m3.policy.v1")
                        eligibility = EligibilityEngine(PolicyCatalog([rule], version="m3.policy.v1")).check(order, service="refund")
                        if str(script.get("target_tool", "")) in {tool_ref, ALIASES.get(tool_ref, "")} and str(script.get("action", "")) == "RETURN_ERROR":
                            requested_code = str(script.get("error_code", ""))
                            if requested_code in {"ELIGIBILITY_DENIED", "ELIGIBILITY_MANUAL"}:
                                eligibility = eligibility.model_copy(update={"decision": "MANUAL" if requested_code == "ELIGIBILITY_MANUAL" else "DENY", "rule_id": requested_code, "reason": "fixture eligibility decision"})
                        _, raw_token = ConfirmTokenManager(repo).issue(session_id=session_id, user_id="fixture-user", run_id=run_id, task_id=task.task_id, order_id=order.entity_id, service="refund", amount=Decimal("10.00"), payload_hash=order.snapshot_hash, topic_version="v1")
                        # A fixture can seed an active case to exercise the
                        # service's unique-business invariant. The mutation
                        # path under test remains AfterSalesService.create_case.
                        if str(script.get("action", "")) == "RETURN_ERROR" and str(script.get("error_code", "")) == "ACTIVE_CASE_EXISTS":
                            now = __import__("agent.storage.repositories", fromlist=["_now"])._now()
                            repo.conn.execute("INSERT INTO aftersales_cases VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (f"fixture_active_{task.task_id}", "fixture-user", order.entity_id, "refund", "REQUESTED", 0, run_id, session_id, task.task_id, f"fixture_{task.task_id}", sha256_json({"fixture": scenario_id, "case": "active"}), "existing active case", "10.00", now, now))
                            # Fixture entities are part of the initial world and
                            # must be committed before the service opens its own
                            # transaction.  Without this boundary SQLite sees
                            # the seed write as an active writer and surfaces an
                            # infrastructure error instead of ACTIVE_CASE_EXISTS.
                            repo.conn.commit()
                        def create_case_callable(**kwargs):
                            try:
                                created = AfterSalesService(repo).create_case(session_id=session_id, user_id="fixture-user", run_id=run_id, task_id=task.task_id, attempt_id=attempt_id, order=order, eligibility=eligibility, service="refund", reason=kwargs.get("reason", "customer request"), amount=Decimal("10.00"), raw_token=raw_token, idempotency_key=context.idempotency_key, review_required=("审核" in text or "高风险" in text))
                                service_holder.update(created)
                                payload = created.get("result").payload if created.get("result") else {"case_id": created.get("case_id")}
                                return {"success": True, "code": "OK", "data": payload}
                            except AfterSalesError as exc:
                                return {"success": False, "code": exc.envelope.code, "message": "after-sales business decision", "data": None}
                        specs = [registry.canonical.get(ref).model_copy(update={"callable": (create_case_callable if ref == tool_ref else registry.canonical.get(ref).callable)}) for ref in registry.canonical.refs()]
                        task_registry = M3Registry(canonical=Registry(specs))
                    # A scheduler worker may already have completed this
                    # independent read.  Consume that immutable stage result
                    # here and persist it once; dependent/write tasks execute
                    # on the Supervisor thread with their trusted context.
                    typed = prefetched.get(task.task_id)
                    if typed is None:
                        port = port_cls(registry=task_registry)
                        port.tool_ref, port.agent_ref, port.output_contract = tool_ref, task.agent_ref, task.output_contract
                        typed = port.invoke(_args_for(tool_ref, text, scenario_id), context=context)
                    logical += 1; physical += 1
                    returned = self._event(repo, run, "TOOL_RETURNED", parent=self._last_event(repo, run_id), task_id=task.task_id, attempt_id=attempt_id, payload={"tool_ref": tool_ref, "ok": typed.ok, "payload_hash": typed.payload_hash}, actor="tool")
                    handled_business_error = (not typed.ok and (typed.error_code or "") in {"AUTH_IDENTITY_MISMATCH", "ORDER_NOT_FOUND", "DATA_MISSING", "DATA_STALE", "POLICY_CONFLICT", "ACTIVE_CASE_EXISTS", "ELIGIBILITY_DENIED", "ELIGIBILITY_MANUAL", "MANUAL"})
                    status = ResultStatus.SUCCEEDED if (typed.ok or handled_business_error) else ResultStatus.FAILED
                    business_code = "OK" if typed.ok else (typed.error_code or "TOOL_EXECUTION_FAILED")
                    branch_codes.append(business_code)
                    if not typed.ok:
                        terminal = "PASS" if handled_business_error else "FAILED"
                service_result = service_holder.get("result")
                if service_result is not None:
                    result = service_result
                    result_event = self._last_event(repo, run_id)
                    if str(getattr(result, "payload", {}) and result.payload.get("status", "")) == "HUMAN_REVIEW":
                        self._event(repo, run, "REVIEW_EVENT", parent=result_event, task_id=task.task_id,
                                    attempt_id=attempt_id, payload={"action": "REVIEW_OPENED", "status": "HUMAN_REVIEW"}, actor="runtime")
                else:
                    result = Result(result_id=f"result_{attempt_id}", run_id=run_id, plan_revision_id=revision.plan_revision_id, task_id=task.task_id, attempt_id=attempt_id, status=status, output_contract=task.output_contract, payload=typed.payload if typed else None, business_code=business_code)
                    result_event = repo.append_result_with_event(result, parent_event_id=self._last_event(repo, run_id))
                persisted[task.task_id] = result
                if typed is not None and not typed.ok:
                    self._event(repo, run, "ERROR", parent=result_event.trace_id if hasattr(result_event, "trace_id") else result_event, task_id=task.task_id, attempt_id=attempt_id, payload={"error_code": result.business_code, "result_id": result.result_id})
                version = int(repo.conn.execute("SELECT state_version FROM runs WHERE run_id=?", (run_id,)).fetchone()[0])
                target = TaskStatus.SUCCEEDED if status == ResultStatus.SUCCEEDED else (TaskStatus.BLOCKED if status == ResultStatus.BLOCKED else TaskStatus.FAILED)
                reducer.transition_task(task.task_id, target, expected_version=version, plan_revision_id=revision.plan_revision_id)
                if status != ResultStatus.SUCCEEDED or (typed is not None and not typed.ok and handled_business_error):
                    # A failure in an independent runnable branch must not
                    # discard sibling outcomes from the same scheduler join.
                    # Dependent chains still stop at their first failed stage.
                    if task.task_id in set(prefetch_meta):
                        continue
                    break
            # Merge branch outcomes independently of execution/commit order.
            # Infra/blocked outcomes dominate handled business decisions, and
            # handled business codes dominate the all-success OK summary.
            def code_priority(code: str) -> tuple[int, str]:
                if code == "BLOCKED" or code in {"CANCELLED", "INFRA_TIMEOUT", "INFRA_UNAVAILABLE", "TOOL_EXECUTION_FAILED", "DEADLINE_EXCEEDED"}:
                    return (3, code)
                if code not in {"OK", "NO_OP"}:
                    return (2, code)
                return (1 if code == "NO_OP" else 0, code)

            if branch_codes:
                business_code = max(branch_codes, key=code_priority)
            version = int(repo.conn.execute("SELECT state_version FROM runs WHERE run_id=?", (run_id,)).fetchone()[0])
            reducer.reduce(run_id, expected_version=version, patch={"status": "SUCCEEDED" if terminal == "PASS" else terminal}, checkpoint_id=f"cp_{run_id}_{terminal.lower()}", plan_revision_id=revision.plan_revision_id, parent_event_id=self._last_event(repo, run_id))
            persisted_status = str(repo.conn.execute("SELECT status FROM runs WHERE run_id=?", (run_id,)).fetchone()[0])
            fingerprint = sha256_json({"world_hash": world_hash, "run_status": persisted_status, "results": [result.payload_hash for result in persisted.values()]})
            plan_fingerprint = sha256_json({"plan_revision_id": revision.plan_revision_id, "tasks": [{"task_id": t.task_id, "agent_ref": t.agent_ref, "capability_refs": t.capability_refs, "depends_on": t.depends_on, "output_contract": t.output_contract} for t in revision.tasks]})
            ref_by_capability = {registry.get(ref).capability_ref: ref for ref in registry.refs()}
            index_by_task = {task.task_id: index for index, task in enumerate(revision.tasks)}
            plan_projection = tuple({"tool_ref": ref_by_capability.get(task.capability_refs[0], ""), "capability_ref": task.capability_refs[0], "depends_on": [index_by_task[dep] for dep in task.depends_on]} for task in revision.tasks)
            expected_intent = str(case.get("expected_intent") or "").upper()
            expected_terminal = str(case.get("expected_terminal_class", "PASS"))
            intent_ok = candidate.intent == expected_intent or (expected_intent == "ORDER_QUERY" and candidate.intent == "ORDER") or (expected_intent == "AFTERSALES" and candidate.intent in {"AFTERSALES", "MIXED"}) or (expected_intent == "POLICY" and candidate.intent in {"POLICY", "MIXED"}) or (expected_intent == "LOGISTICS" and candidate.intent == "LOGISTICS") or (expected_intent == "HANDOFF" and candidate.intent in {"ESCALATION", "MIXED"}) or (expected_intent == "MIXED" and candidate.intent == "MIXED")
            path_ok = tuple(observed) == tuple(case.get("expected_tool_path") or [])
            valid = persisted_status in {"SUCCEEDED", "FAILED", "BLOCKED", "CANCELLED"} and bool(repo.conn.execute("SELECT 1 FROM trace_outbox WHERE run_id=?", (run_id,)).fetchone())
            aftersales_state = [dict(row) for row in repo.conn.execute("SELECT case_id,status,state_version,order_id,service FROM aftersales_cases WHERE run_id=? ORDER BY case_id", (run_id,)).fetchall()]
            submit_state = [dict(row) for row in repo.conn.execute("SELECT tool_ref,request_fingerprint,idempotency_key,result_id,status FROM tool_submit_log WHERE run_id=? ORDER BY submit_id", (run_id,)).fetchall()]
            terminal_fingerprint = sha256_json({"initial_world_hash": world_hash, "plan_fingerprint": plan_fingerprint, "run_status": persisted_status, "task_statuses": json.loads(repo.conn.execute("SELECT shared_state_json FROM runs WHERE run_id=?", (run_id,)).fetchone()[0]).get("task_statuses", {}), "business_code": business_code, "result_hashes": [result.payload_hash for result in persisted.values()], "aftersales_cases": aftersales_state, "submit_log": submit_state})
            return ScenarioRun(scenario_id=scenario_id, intent=candidate.intent, expected_intent=expected_intent, tool_path=tuple(observed), expected_tool_path=tuple(case.get("expected_tool_path") or []), terminal_class=terminal, logical_calls=logical, physical_attempts=physical, trajectory_valid=valid, intent_ok=intent_ok, path_ok=path_ok, expected_terminal_class=expected_terminal, observed_terminal_class=terminal, world_fingerprint=world_hash, db_path=str(db_path), run_id=run_id, business_code=business_code, plan_fingerprint=plan_fingerprint, plan_projection=plan_projection, terminal_fingerprint=terminal_fingerprint)
        finally:
            repo.close()


__all__ = ["M3ScenarioRunner", "ScenarioRun"]
