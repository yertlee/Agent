"""Read-only replay and isolated re-execution for frozen bundles."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from .contracts import ExecutionMode, FreezeBundle
from .runner import AgentRunner, HarnessRun


@dataclass(frozen=True)
class ReplayResult:
    run_id: str
    event_count: int
    final_fingerprint: str
    side_effects: bool = False
    tool_calls: int = 0


@dataclass(frozen=True)
class DivergenceReport:
    equivalent: bool
    differences: tuple[str, ...]
    compared_projection: Mapping[str, Any]


def _stable_result_projection(bundle: FreezeBundle | HarnessRun) -> dict[str, Any]:
    volatile_keys = {
        "run_id", "session_id", "task_id", "attempt_id", "plan_revision_id",
        "result_id", "trace_id", "parent_event_id", "event_id", "checkpoint_id",
        "audit_id", "idempotency_key", "created_at", "updated_at", "started_at",
        "ended_at", "model_hash", "snapshot_hash", "before_hash", "after_hash",
        # Argument hashes can depend on a redacted value restored from a
        # stable projection during re_execute; the invocation shape/tool path
        # remains compared separately.
        "args_hash",
    }

    def stable(value: Any, key: str | None = None) -> Any:
        if isinstance(value, dict):
            normalized: dict[str, Any] = {}
            for raw_key, child in value.items():
                field = str(raw_key)
                if field in volatile_keys:
                    continue
                # SQLite rows store nested objects as canonical JSON text.
                # Parse those values before normalization, otherwise dynamic
                # IDs hidden in ``*_json`` columns bypass the projection.
                if field.endswith("_json") and isinstance(child, str):
                    try:
                        child = json.loads(child)
                    except (TypeError, ValueError):
                        pass
                normalized[field] = stable(child, field)
            return normalized
        if isinstance(value, (list, tuple)):
            return [stable(item, key) for item in value]
        return value

    def trace_projection(events: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
        # RUN_FROZEN exists only in the source bundle; re_execute intentionally
        # produces a fresh, as-yet-unfrozen run.  The preceding event sequence
        # is the comparable runtime trace.
        return [stable({"event_type": row.get("event_type"), "payload": row.get("payload", {}), "scene_clock": row.get("scene_clock")})
                for row in events if row.get("event_type") != "RUN_FROZEN"]

    if isinstance(bundle, FreezeBundle):
        trace_rows: list[dict[str, Any]] = []
        with Path(bundle.trace.path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    trace_rows.append(json.loads(line))
        result_rows = [stable(json.loads(Path(item.path).read_text(encoding="utf-8"))) for item in bundle.results]
        plan_rows = [stable(json.loads(Path(item.path).read_text(encoding="utf-8"))) for item in bundle.plan_revisions]
        context = json.loads(Path(bundle.run_context.path).read_text(encoding="utf-8"))
        return {
            "trace": trace_projection(trace_rows),
            "plans": plan_rows,
            "results": result_rows,
            "world_snapshot": json.loads(Path(bundle.world_snapshot.path).read_text(encoding="utf-8")),
            "final_response": json.loads(Path(bundle.final_response.path).read_text(encoding="utf-8")).get("response", ""),
            "context": context,
        }
    events = [event.model_dump(mode="json") for event in bundle.recorder.events]
    return {
        "trace": trace_projection(events),
        "plans": stable(list(bundle.plan_rows)),
        "results": stable(bundle.result_rows or [item.typed.model_dump(mode="json") for item in bundle.results]),
        "world_snapshot": bundle.world_snapshot.canonical_projection(),
        "final_response": bundle.final_response,
    }


class ReplayRunner:
    def replay(self, bundle: FreezeBundle) -> ReplayResult:
        # Deliberately no repository, AgentPort, or tool simulator is touched.
        return ReplayResult(run_id=bundle.run_id, event_count=int(bundle.trace.event_count or 0),
                            final_fingerprint=bundle.final_world_fingerprint, side_effects=False,
                            tool_calls=sum(1 for line in Path(bundle.trace.path).read_text(encoding="utf-8").splitlines() if '"event_type":"TOOL_CALLED"' in line))

    def re_execute(self, bundle: FreezeBundle, *, runner_factory=None) -> tuple[HarnessRun, DivergenceReport]:
        context = json.loads(Path(bundle.run_context.path).read_text(encoding="utf-8"))
        calls = list(context.get("invocations") or [])
        for call in calls:
            args = dict(call.get("args") or {})
            for key in list(args):
                if key.endswith("_hash") and any(marker in key.lower() for marker in ("phone", "address", "payment", "token", "secret", "password")):
                    original_key = key[:-5]
                    args[original_key] = "0000" if "phone" in key.lower() else "redacted"
                    del args[key]
            call["args"] = args
        if not calls:
            raise ValueError("FreezeBundle does not contain replayable invocation projection")
        factory = runner_factory or AgentRunner
        world_value = json.loads(Path(bundle.world_snapshot.path).read_text(encoding="utf-8"))
        from .world import WorldStateBuilder
        world = WorldStateBuilder().build(world_value)
        failure_value = None
        if bundle.failure_script is not None:
            failure_value = json.loads(Path(bundle.failure_script.path).read_text(encoding="utf-8"))
        if context.get("scenario"):
            from .contracts import FailureScript, Scenario
            scenario = Scenario(scenario_id=context["scenario"]["scenario_id"], category="re_execute", turns=list(context["scenario"]["turns"]), world_fixture_ref=context["scenario"].get("world_fixture_ref", world.world_fixture_ref), split="re_execute", version_tuple=bundle.version_tuple, failure_script=FailureScript.from_mapping(failure_value))
            runner = factory(world_snapshot=world, mode=ExecutionMode.RE_EXECUTE, failure_script=failure_value, run_id=None, session_id=None, version_tuple=bundle.version_tuple)
            run = runner.run_from_turns(scenario)
            left = _stable_result_projection(bundle)
            right = _stable_result_projection(run)
            differences = []
            if left.get("world_snapshot") != right.get("world_snapshot"): differences.append("world_snapshot")
            if left.get("trace") != right.get("trace"): differences.append("trace")
            if left.get("plans") != right.get("plans"): differences.append("plans")
            if left.get("results") != right.get("results"): differences.append("results")
            return run, DivergenceReport(equivalent=not differences, differences=tuple(differences), compared_projection={"bundle": left, "re_execute": right})
        runner = factory(world_snapshot=world, mode=ExecutionMode.RE_EXECUTE,
                         failure_script=failure_value, run_id=None, session_id=None,
                         version_tuple=bundle.version_tuple)
        run = runner.run(calls=calls)
        left = _stable_result_projection(bundle)
        right = _stable_result_projection(run)
        differences: list[str] = []
        if left.get("world_snapshot") != right.get("world_snapshot"):
            differences.append("world_snapshot")
        if left.get("final_response") != right.get("final_response"):
            differences.append("final_response")
        if left.get("trace") != right.get("trace"):
            differences.append("trace")
        if left.get("plans") != right.get("plans"):
            differences.append("plans")
        # A bundle may intentionally omit result payloads; in that case compare
        # only the stable invocation/result contract, never random runtime IDs.
        if bundle.results and left.get("results") != right.get("results"):
            differences.append("results")
        return run, DivergenceReport(equivalent=not differences, differences=tuple(differences), compared_projection={"bundle": left, "re_execute": right})


__all__ = ["DivergenceReport", "ReplayResult", "ReplayRunner"]
