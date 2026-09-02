from agent.m3_entrypoint import m3_enabled, run_m3_case
from agent.m3_runtime import M3ScenarioRunner


def test_isolated_m3_runtime_routes_and_tracks_logical_physical_counts(monkeypatch):
    case = {"scenario_id": "fixture", "category": "order_read_auth", "turns": ["查询订单 ORD-SYN"], "expected_intent": "order_query", "expected_tool_path": ["get_order_info_tool"], "expected_terminal_class": "PASS"}
    result = M3ScenarioRunner().run(case)
    assert result.trajectory_valid and result.tool_path == ("get_order_info_tool",)
    assert result.logical_calls == 1 and result.physical_attempts == 1
    monkeypatch.setenv("M3_EXECUTION_ENABLED", "1")
    assert m3_enabled() and run_m3_case(case).trajectory_valid
