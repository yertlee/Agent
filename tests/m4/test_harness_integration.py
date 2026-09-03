from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from agent.domain.objects import Run, sha256_json
from agent.m2_registry import Registry, ToolSpec
from agent.m3_registry import M3Registry
from agent.storage.m2 import M2Repository
from eval.harness import (
    AgentRunner,
    EvaluationInput,
    FailureScript,
    FailureTrigger,
    ReplayRunner,
    ScenarioLoader,
    ToolSimulator,
    TraceRecorder,
    VersionTuple,
    WorldStateBuilder,
    verify_bundle,
)


def _registry() -> M3Registry:
    def order_info(*, order_id: str, phone_last4: str):
        return {"success": True, "data": {"order_id": order_id, "state": "PAID"}}

    return M3Registry(canonical=Registry([ToolSpec(
        tool_ref="order/get_info@v1", capability_ref="order/read@v1", owner="order-agent",
        args_schema="order.read.v1", result_schema="order.result.v1", risk="READ",
        implementation_mode="SIMULATED", side_effect="READ_ONLY", timeout_ms=5000,
        allowed_error_codes=("DATA_MISSING",), callable=order_info,
    )]))


def _world():
    return WorldStateBuilder().build({
        "world_fixture_ref": "m4-world-1", "world_template_version": "world-template.v2",
        "seed": 17, "scene_clock": "2026-01-01T00:00:00Z",
        "entities": [{"entity_type": "order", "entity_id": "o2"}, {"entity_type": "order", "entity_id": "o1"}],
    })


def _versions():
    return VersionTuple(schema="m4.schema.v1", model="m4.model.v1", prompt="m4.prompt.v1",
                        code="m4.code.v1", registry="m4.registry.v1", tool_impl="m4.tool.v1",
                        config="m4.config.v1", policy_catalog="m4.policy.v1", kb="m4.kb.v1",
                        dataset="dev72.v1", harness="m4.harness.v1", trace_schema="m4.trace.v1",
                        evaluator="m4.evaluator.v1", simulator="m4.sim.v1",
                        world_template="world-template.v2", seed=17)


def test_world_sort_and_fault_counts_do_not_change_logical_numbering():
    world = _world()
    assert [row["entity_id"] for row in world.canonical_entities] == ["o1", "o2"]
    script = FailureScript(triggers=[FailureTrigger(
        trigger_id="once", tool_ref="order/get_info@v1", action="RETURN_ERROR",
        error_code="DATA_MISSING", logical_call=1, count=1,
    )])
    simulator = ToolSimulator(registry=_registry(), mode="fault", world_snapshot=world, failure_script=script)
    from agent.m2_context import InvocationContext
    context = InvocationContext(session_id="s", user_id="u", run_id="r", plan_revision_id="p",
        task_id="t", attempt_id="a", agent_ref="order-agent", auth_scope="order/read@v1",
        idempotency_key="i", deadline=datetime(2030, 1, 1, tzinfo=timezone.utc),
        config_version="c", registry_version="r", dataset_version="d", trace_id="tr")
    first = simulator.invoke("order/get_info@v1", {"order_id": "o", "phone_last4": "1234"},
                             context=context, logical_call_no=1, physical_attempt_no=1)
    second = simulator.invoke("order/get_info@v1", {"order_id": "o", "phone_last4": "1234"},
                              context=context, logical_call_no=2, physical_attempt_no=1)
    assert first.logical_call_no == 1 and first.ok is False
    assert second.logical_call_no == 2 and second.ok is True


def test_multiple_failure_triggers_use_priority_then_script_order():
    world = _world()
    script = FailureScript(triggers=[
        FailureTrigger(trigger_id="first", tool_ref="order/get_info@v1", action="RETURN_ERROR",
                        error_code="DATA_MISSING", logical_call=1, priority=5),
        FailureTrigger(trigger_id="second", tool_ref="order/get_info@v1", action="RETURN_ERROR",
                        error_code="AUTH_IDENTITY_MISMATCH", logical_call=1, priority=5),
    ])
    simulator = ToolSimulator(registry=_registry(), mode="fault", world_snapshot=world, failure_script=script)
    from agent.m2_context import InvocationContext
    context = InvocationContext(session_id="s", user_id="u", run_id="r", plan_revision_id="p",
        task_id="t", attempt_id="a", agent_ref="order-agent", auth_scope="order/read@v1",
        idempotency_key="i", deadline=datetime(2030, 1, 1, tzinfo=timezone.utc),
        config_version="c", registry_version="r", dataset_version="d", trace_id="tr")
    first = simulator.invoke("order/get_info@v1", {"order_id": "o", "phone_last4": "1234"},
                             context=context, logical_call_no=1)
    second = simulator.invoke("order/get_info@v1", {"order_id": "o", "phone_last4": "1234"},
                              context=context, logical_call_no=1)
    assert first.injected and first.injected.trigger_id == "first"
    assert second.injected and second.injected.trigger_id == "second"


def test_trace_drain_freeze_and_late_audit_are_persisted(tmp_path):
    world = _world()
    repo = M2Repository(str(tmp_path / "run.db"))
    repo.create_session("s", "u")
    repo.create_run(Run(run_id="r", session_id="s", initial_world_hash=world.snapshot_hash))
    recorder = TraceRecorder(run_id="r", session_id="s", repository=repo,
        scene_clock=world.scene_clock, version_tuple=_versions())
    recorder.record("RUN_CREATED", payload={"scenario_id": "m4"})
    assert recorder.pending() == 1
    manifest = recorder.drain_jsonl(tmp_path / "trace.jsonl")
    assert manifest["event_count"] == 1
    assert recorder.pending() == 0
    assert repo.conn.execute("SELECT COUNT(*) FROM trace_manifests WHERE run_id='r'").fetchone()[0] == 0
    assert repo.conn.execute("SELECT delivered_at FROM trace_outbox WHERE event_id IS NOT NULL").fetchone()[0]
    bundle = recorder.freeze_bundle(world_snapshot=world, final_response="ok", final_fingerprint="f" * 64,
        failure_script=FailureScript(), run_context={"scenario_id": "m4"}, version_tuple=_versions())
    assert bundle.checksum
    assert verify_bundle(bundle, repository=repo)["ok"] is True
    assert repo.conn.execute("SELECT COUNT(*) FROM trace_manifests WHERE run_id='r'").fetchone()[0] == 1
    with pytest.raises(RuntimeError):
        recorder.record("CHECKPOINT", payload={})
    assert repo.conn.execute("SELECT COUNT(*) FROM late_event_audit WHERE run_id='r'").fetchone()[0] == 1
    fresh = TraceRecorder(run_id="r", session_id="s", repository=repo, scene_clock=world.scene_clock, version_tuple=_versions())
    with pytest.raises(RuntimeError):
        fresh.record("CHECKPOINT", payload={})
    assert repo.conn.execute("SELECT COUNT(*) FROM trace_outbox WHERE run_id='r'").fetchone()[0] == 2
    assert repo.conn.execute("SELECT COUNT(*) FROM late_event_audit WHERE run_id='r'").fetchone()[0] == 2


def test_verify_bundle_reports_corrupt_trace_without_raising(tmp_path):
    world = _world()
    recorder = TraceRecorder(run_id="r", session_id="s", scene_clock=world.scene_clock, version_tuple=_versions())
    recorder.record("RUN_CREATED", payload={})
    recorder.drain_jsonl(tmp_path / "trace.jsonl")
    bundle = recorder.freeze_bundle(world_snapshot=world, final_response="", final_fingerprint="f" * 64,
        failure_script=FailureScript(), run_context={"scenario_id": "m4"}, version_tuple=_versions())
    Path(bundle.trace.path).write_text("not-json\n", encoding="utf-8")
    result = verify_bundle(bundle)
    assert result["ok"] is False
    assert result["checksum_mismatches"]


def test_repository_cannot_bypass_trace_seal(tmp_path):
    world = _world()
    repo = M2Repository(str(tmp_path / "sealed.db"))
    repo.create_session("s", "u")
    repo.create_run(Run(run_id="r", session_id="s", initial_world_hash=world.snapshot_hash))
    recorder = TraceRecorder(run_id="r", session_id="s", repository=repo,
        scene_clock=world.scene_clock, version_tuple=_versions())
    recorder.record("RUN_CREATED", payload={})
    recorder.drain_jsonl(tmp_path / "sealed.jsonl")
    recorder.freeze_bundle(world_snapshot=world, final_response="ok", final_fingerprint="f" * 64,
        failure_script=FailureScript(), run_context={"scenario_id": "sealed"}, version_tuple=_versions())
    from agent.trace.events import build_event
    late = build_event(run_id="r", session_id="s", event_type="CHECKPOINT", seq_no=3,
        payload={}, scene_clock=world.scene_clock)
    with pytest.raises(RuntimeError, match="sealed"):
        repo.append_m2_event(late)
    assert repo.conn.execute("SELECT COUNT(*) FROM trace_outbox WHERE run_id='r'").fetchone()[0] == 2
    assert repo.conn.execute("SELECT COUNT(*) FROM late_event_audit WHERE event_id=?", (late.trace_id,)).fetchone()[0] == 1


def test_replay_is_read_only_and_reexecute_uses_new_identity(tmp_path):
    world = _world()
    runner = AgentRunner(registry=_registry(), world_snapshot=world, version_tuple=_versions())
    run = runner.run(calls=[{"tool_ref": "order/get_info@v1", "args": {"order_id": "o", "phone_last4": "1234"}}])
    run.recorder.drain_jsonl(tmp_path / "trace.jsonl")
    bundle = run.freeze()
    replay = ReplayRunner().replay(bundle)
    assert replay.side_effects is False and replay.tool_calls == 1
    replay_run, divergence = ReplayRunner().re_execute(
        bundle, runner_factory=lambda **kwargs: AgentRunner(registry=_registry(), **kwargs)
    )
    assert replay_run.run_id != run.run_id
    assert replay_run.results[0].typed.payload_hash == run.results[0].typed.payload_hash
    assert divergence.equivalent is True


def test_scenario_loader_discards_nested_evaluator_gold():
    scenario = ScenarioLoader().from_mapping({
        "scenario_id": "s1", "category": "order", "turns": ["查订单"],
        "world_fixture_ref": "m4-world-1", "split": "dev",
        "version_tuple": _versions().model_dump(),
        "tool_calls": [{"tool_ref": "order/get_info@v1", "args": {"order_id": "o", "phone_last4": "1234"}}],
        "gold": {"expected_terminal_class": "PASS", "nested": {"rubric": "secret"}},
    })
    assert "gold" not in scenario.model_dump(mode="python")


def test_freeze_bundle_rejects_recursive_gold_key(tmp_path):
    world = _world()
    recorder = TraceRecorder(run_id="r", session_id="s", scene_clock=world.scene_clock, version_tuple=_versions())
    recorder.record("RUN_CREATED", payload={})
    recorder.drain_jsonl(tmp_path / "trace.jsonl")
    with pytest.raises(ValueError, match="gold/evaluator"):
        recorder.freeze_bundle(world_snapshot=world, final_response="", final_fingerprint="f" * 64,
            failure_script=FailureScript(), run_context={"nested": {"expected": "x"}}, version_tuple=_versions())


def test_freeze_bundle_rejects_sensitive_value_hidden_in_turn(tmp_path):
    world = _world()
    recorder = TraceRecorder(run_id="r", session_id="s", scene_clock=world.scene_clock, version_tuple=_versions())
    recorder.record("RUN_CREATED", payload={})
    recorder.drain_jsonl(tmp_path / "trace.jsonl")
    with pytest.raises(ValueError, match="sensitive"):
        recorder.freeze_bundle(world_snapshot=world, final_response="", final_fingerprint="f" * 64,
            failure_script=FailureScript(), run_context={"turns": ["手机号 13800138000"]}, version_tuple=_versions())


def test_normal_scenario_cannot_bypass_public_runtime_with_tool_calls():
    world = _world()
    scenario = ScenarioLoader().from_mapping({
        "scenario_id": "s1", "category": "order", "turns": ["查订单"],
        "world_fixture_ref": "m4-world-1", "split": "dev",
        "version_tuple": _versions().model_dump(),
        "tool_calls": [{"tool_ref": "order/get_info@v1", "args": {"order_id": "o", "phone_last4": "1234"}}],
    })
    with pytest.raises(ValueError, match="evaluator/replay"):
        AgentRunner(registry=_registry(), world_snapshot=world, version_tuple=_versions()).run(scenario)


def test_version_tuple_is_complete():
    assert len(_versions().model_dump(by_alias=True)) == 16
    with pytest.raises(Exception):
        VersionTuple(dataset="d", registry="r", policy_catalog="p", simulator="s", world_template="w", seed=1)


def test_turns_bridge_freezes_real_m3_trace_plan_and_results(tmp_path):
    world = _world()
    scenario = ScenarioLoader().from_mapping({
        "scenario_id": "bridge-1", "category": "order", "turns": ["查订单"],
        "world_fixture_ref": "m4-world-1", "split": "dev", "version_tuple": _versions().model_dump(),
    })
    run = AgentRunner(world_snapshot=world, version_tuple=_versions()).run(scenario)
    run.recorder.drain_jsonl(tmp_path / "bridge-trace.jsonl")
    bundle = run.freeze()
    assert bundle.run_id == run.run_id
    assert bundle.plan_revisions and bundle.results
    assert bundle.trace.event_count and bundle.trace.path
    assert verify_bundle(bundle)["ok"] is True
    replay_run, divergence = ReplayRunner().re_execute(bundle)
    assert replay_run.run_id != run.run_id
    assert replay_run.runtime_result is not None
    assert divergence.equivalent is True
    assert divergence.compared_projection["bundle"]["world_snapshot"] == divergence.compared_projection["re_execute"]["world_snapshot"]
    rows = [line for line in open(bundle.trace.path, encoding="utf-8") if line.strip()]
    assert '"event_type":"RUN_FROZEN"' in rows[-1]
