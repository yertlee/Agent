"""Small mode-aware runner for harness scenarios and public AgentPorts."""
from __future__ import annotations

import hashlib
import json
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

from agent.m2_context import InvocationContext
from agent.m3_registry import M3Registry, build_m3_registry
from agent.agents import TypedAgentResult
from agent.storage.m2 import M2Repository

from .contracts import ExecutionMode, FailureScript, Scenario, WorldSnapshot
from .simulator import HarnessToolResult, ToolSimulator
from .trace_recorder import TraceRecorder


@dataclass(frozen=True)
class HarnessRun:
    run_id: str
    session_id: str
    scenario_id: str
    mode: ExecutionMode
    world_snapshot: WorldSnapshot
    calls: tuple[dict[str, Any], ...]
    results: tuple[HarnessToolResult, ...]
    final_response: str
    final_fingerprint: str
    recorder: TraceRecorder
    runtime_result: Any = None
    plan_rows: tuple[dict[str, Any], ...] = ()
    result_rows: tuple[dict[str, Any], ...] = ()
    scenario_projection: dict[str, Any] | None = None

    def freeze(self):
        result_rows = list(self.result_rows) or [item.typed.model_dump(mode="json") for item in self.results]
        return self.recorder.freeze_bundle(
            world_snapshot=self.world_snapshot,
            final_response=self.final_response,
            final_fingerprint=self.final_fingerprint,
            failure_script=self.recorder.failure_script,
            run_context={"scenario_id": self.scenario_id, "mode": self.mode.value, "invocations": list(self.calls), **({"scenario": self.scenario_projection} if self.scenario_projection else {})},
            version_tuple=self.recorder.version_tuple,
            results=result_rows,
            plan_revisions=list(self.plan_rows),
        )


class AgentRunner:
    def __init__(self, *, world_snapshot: WorldSnapshot, registry: M3Registry | None = None,
                 mode: ExecutionMode | str = ExecutionMode.SIMULATED,
                 failure_script: FailureScript | Mapping[str, Any] | None = None,
                 recorder: TraceRecorder | None = None, run_id: str | None = None,
                 session_id: str | None = None, user_id: str = "harness-user",
                 version_tuple=None):
        self.registry = registry or build_m3_registry()
        self.mode = ExecutionMode(mode)
        self.world_snapshot = world_snapshot
        self.failure_script = FailureScript.from_mapping(failure_script)
        self.run_id = run_id or f"harness-run-{uuid.uuid4().hex}"
        self.session_id = session_id or f"harness-session-{uuid.uuid4().hex}"
        self.user_id = user_id
        self.recorder = recorder or TraceRecorder(run_id=self.run_id, session_id=self.session_id,
            failure_script=self.failure_script, scene_clock=world_snapshot.scene_clock, version_tuple=version_tuple)
        self.simulator = ToolSimulator(registry=self.registry, mode=self.mode, world_snapshot=world_snapshot,
                                       failure_script=self.failure_script)

    def _context(self, tool_ref: str, task_id: str, attempt_id: str) -> InvocationContext:
        spec = self.registry.get(tool_ref)
        return InvocationContext(session_id=self.session_id, user_id=self.user_id, run_id=self.run_id,
            plan_revision_id="harness-plan-v1", task_id=task_id, attempt_id=attempt_id,
            agent_ref=spec.owner, auth_scope=spec.capability_ref, idempotency_key=f"idem-{attempt_id}",
            deadline=self.world_snapshot.scene_clock + timedelta(seconds=max(spec.timeout_ms / 1000, 1)),
            config_version="m4.harness.v1", registry_version="m4.registry.v1",
            dataset_version="m4.dataset.v1", trace_id=f"trace-{attempt_id}")

    def run(self, scenario: Scenario | None = None, *, calls: list[Mapping[str, Any]] | None = None) -> HarnessRun:
        scenario_id = scenario.scenario_id if scenario else "inline"
        if scenario is not None and calls is None:
            if scenario.tool_calls:
                raise ValueError("scenario.tool_calls is evaluator/replay input; normal runs must use turns/world")
            return self.run_from_turns(scenario)
        if scenario is not None and calls is not None:
            raise ValueError("explicit calls are reserved for replay/re_execute, not normal scenarios")
        rows = calls or []
        if not rows:
            raise ValueError("runner requires scenario.tool_calls or calls")
        observed: list[HarnessToolResult] = []
        call_projection: list[dict[str, Any]] = []
        for index, row in enumerate(rows, start=1):
            item = dict(row)
            tool_ref = str(item["tool_ref"])
            args = dict(item.get("args") or {})
            task_id = str(item.get("task_id") or f"task-{index}")
            attempt_id = str(item.get("attempt_id") or f"attempt-{index}-{uuid.uuid4().hex[:8]}")
            context = self._context(tool_ref, task_id, attempt_id)
            self.recorder.record("ATTEMPT_STARTED", task_id=task_id, attempt_id=attempt_id,
                                 payload={"attempt_id": attempt_id}, plan_revision_id=context.plan_revision_id)
            self.recorder.record("TOOL_CALLED", task_id=task_id, attempt_id=attempt_id,
                                 payload={"tool_ref": tool_ref, "args_hash": hashlib.sha256(json.dumps(args, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()},
                                 plan_revision_id=context.plan_revision_id, actor="agent")
            outcome = self.simulator.invoke(tool_ref, args, context=context, logical_call_no=index,
                                            physical_attempt_no=int(item.get("physical_attempt_no", 1)), layer=item.get("layer"))
            observed.append(outcome)
            self.recorder.record("TOOL_RETURNED", task_id=task_id, attempt_id=attempt_id,
                                 payload={"tool_ref": tool_ref, "ok": outcome.ok, "error_code": outcome.typed.error_code,
                                          "payload_hash": outcome.typed.payload_hash}, plan_revision_id=context.plan_revision_id,
                                 actor="tool")
            safe_args = {}
            for key, value in args.items():
                if any(marker in key.lower() for marker in ("phone", "address", "payment", "token", "secret", "password")):
                    safe_args[f"{key}_hash"] = hashlib.sha256(str(value).encode("utf-8")).hexdigest()
                else:
                    safe_args[key] = value
            call_projection.append({"tool_ref": tool_ref, "args": safe_args, "task_id": task_id})
        final = {"ok": all(item.ok for item in observed), "results": [item.typed.model_dump(mode="json") for item in observed]}
        return HarnessRun(run_id=self.run_id, session_id=self.session_id, scenario_id=scenario_id, mode=self.mode,
                          world_snapshot=self.world_snapshot, calls=tuple(call_projection), results=tuple(observed),
                          final_response="", final_fingerprint=hashlib.sha256(str(final).encode()).hexdigest(), recorder=self.recorder)

    def run_from_turns(self, scenario: Scenario) -> HarnessRun:
        """Delegate normal turns/world execution to the public M3 runtime entrypoint.

        Harness code does not inspect or mutate M3's private state.  The
        returned runtime result is retained only as an opaque provenance value;
        replayable invocations are created later from a frozen run.
        """
        from agent.m3_runtime import M3ScenarioRunner
        script = scenario.failure_script.model_dump(mode="python")
        if scenario.failure_script.triggers:
            trigger = scenario.failure_script.triggers[0]
            script = {"version": scenario.failure_script.version, "action": trigger.action, "target_tool": trigger.tool_ref,
                      "error_code": trigger.error_code or "", "logical_call": trigger.logical_call,
                      "physical_attempt": trigger.physical_attempt}
        runtime = M3ScenarioRunner(db_dir=Path(tempfile.mkdtemp(prefix="m4-runtime-"))).run({
            "scenario_id": scenario.scenario_id,
            "turns": list(scenario.turns),
            "world_fixture_ref": scenario.world_fixture_ref,
            "failure_script": script,
            "world_snapshot": self.world_snapshot.canonical_projection(),
        })
        repo = M2Repository(runtime.db_path)
        runtime_session_id = str(repo.conn.execute("SELECT session_id FROM runs WHERE run_id=?", (runtime.run_id,)).fetchone()[0])
        recorder = TraceRecorder(run_id=runtime.run_id, session_id=runtime_session_id, repository=repo,
            scene_clock=self.world_snapshot.scene_clock, version_tuple=scenario.version_tuple, failure_script=scenario.failure_script)
        plan_rows = []
        revision_rows = repo.conn.execute("SELECT * FROM plan_revisions WHERE run_id=? ORDER BY version", (runtime.run_id,)).fetchall()
        for row in revision_rows:
            item = dict(row)
            item["tasks_json"] = json.loads(item["tasks_json"])
            plan_rows.append(item)
        result_rows = [dict(row) for row in repo.conn.execute("SELECT * FROM results WHERE run_id=? ORDER BY created_at", (runtime.run_id,)).fetchall()]
        fingerprint = getattr(runtime, "terminal_fingerprint", "") or scenario.version_tuple.fingerprint
        aliases = {"get_order_info_tool": "order/get_info@v1", "query_logistics_snapshot_tool": "logistics/query@v1",
                   "query_aftersales_tool": "aftersales/query@v1", "create_aftersales_tool": "aftersales/create@v1",
                   "policy_rag_search_tool": "policy/search@v1", "handoff_to_human_tool": "human/handoff@v1"}
        calls = tuple({"tool_ref": aliases.get(tool, tool), "args": {"order_id": f"ORD-{scenario.scenario_id.upper()}", "phone_last4_hash": hashlib.sha256(b"0000").hexdigest()}, "task_id": f"replay-task-{idx}"} for idx, tool in enumerate(runtime.tool_path, 1) if aliases.get(tool, tool) in self.registry.refs())
        return HarnessRun(run_id=runtime.run_id, session_id=runtime_session_id, scenario_id=scenario.scenario_id,
                          mode=self.mode, world_snapshot=self.world_snapshot, calls=calls, results=(),
                          final_response="", final_fingerprint=fingerprint, recorder=recorder,
                          runtime_result=runtime, plan_rows=tuple(plan_rows), result_rows=tuple(result_rows),
                          scenario_projection={"scenario_id": scenario.scenario_id, "turns": list(scenario.turns),
                                                "world_fixture_ref": scenario.world_fixture_ref})


__all__ = ["AgentRunner", "HarnessRun"]
