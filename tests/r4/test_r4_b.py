from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from agent.r4_b_contracts import (
    R4_SUPERVISOR_PROMPT_CHECKSUM_SHA256,
    R4_SUPERVISOR_PROMPT_TEMPLATE,
    R4_SUPERVISOR_PROMPT_VERSION,
    R4_TOPOLOGY_CAPABILITIES,
    R4_TOPOLOGY_DEPENDENCIES,
    R4SupervisorDecisionV1,
    R4SupervisorProviderBoundary,
    StaticR4SupervisorProvider,
    build_supervisor_prompt,
)
from agent.interactive_runtime import ModelResult, ModelUsage
from eval.r4_b_eval import R4_B_MODES, evaluate_r4_b, evaluate_r4_b_supervisor, run_r4_b_case
from eval.r4_b_inputs import build_r4_b_dev_inputs, input_manifest, load_r4_b_inputs, write_r4_b_dev_inputs


def _contains_forbidden(value):
    forbidden = {"expected", "gold", "terminal", "error_code", "answer"}
    if isinstance(value, dict):
        return any(str(key).lower() in forbidden or _contains_forbidden(child) for key, child in value.items())
    if isinstance(value, list):
        return any(_contains_forbidden(child) for child in value)
    return False


def _execution_spec(cases):
    rows = {}
    for case in cases:
        if case.failure_family in {"dependency_pending", "recoverable_timeout", "recoverable_lost"}:
            topology = "order->logistics"
        elif case.failure_family == "partial_branch":
            topology = "order+policy"
        elif case.failure_family == "policy_only":
            topology = "policy_only"
        else:
            topology = "order_only"
        row = {"case_id": case.case_id, "topology": topology}
        if case.failure_family == "dependency_pending":
            row.update(action="exercise_dependency_ordering", dependencies=[("order", "logistics")])
        elif case.failure_family == "duplicate":
            row["action"] = "replay_request"
        elif case.failure_family == "late":
            row["action"] = "freeze_and_replay"
        if case.failure_family == "recoverable_timeout":
            row["failure_script"] = {"order": {"kind": "TIMEOUT", "count": 1}}
        if case.failure_family == "recoverable_lost":
            row["failure_script"] = {"order": {"kind": "LOST", "count": 1}}
        if case.failure_family == "non_retryable":
            row["failure_script"] = {"order": {"kind": "NON_RETRYABLE", "count": 1}}
        if case.failure_family in {"semantic_wrong", "partial_branch"}:
            row["failure_script"] = {"order": {"kind": "SEMANTIC_WRONG", "count": 1}}
        rows[case.case_id] = row
    return rows


def test_supervisor_contract_is_provider_only_and_extra_forbid():
    decision = R4SupervisorDecisionV1(
        requested_capabilities=("order/read@v1",),
        topology="order_only",
        entities={"order_id": "R4B-ORD-001", "phone_last4": "0001"},
    )
    assert decision.schema_version == "r4.supervisor.decision.v1"
    with pytest.raises(ValidationError):
        R4SupervisorDecisionV1.model_validate({**decision.model_dump(mode="python"), "run_id": "trusted-value"})
    boundary = R4SupervisorProviderBoundary(StaticR4SupervisorProvider(decision), model_ref="test-provider")
    result = boundary.decide("Please check the order.")
    assert result.decision == decision
    assert result.evidence.provider_called and result.evidence.schema_valid
    assert result.evidence.total_attempts == 1
    assert result.evidence.first_attempt_schema_valid is True
    assert result.evidence.retried is False


def test_supervisor_schema_invalid_retries_once_with_same_prompt_and_redacted_evidence():
    decision = {
        "topology": "order_only",
        "requested_capabilities": ("order/read@v1",),
    }
    invalid = {
        "case_id": "SECRET_CASE_ID",
        "gold": "SECRET_GOLD",
        "expected": "SECRET_EXPECTED",
        "topology": "order_only",
        "requested_capabilities": (),
    }
    prompts = []
    responses = [invalid, ModelResult(decision, ModelUsage(input_tokens=7, output_tokens=3), "model-retry")]

    class Provider:
        def __call__(self, prompt, _schema):
            prompts.append(prompt)
            return responses.pop(0)

    result = R4SupervisorProviderBoundary(Provider(), model_ref="fallback-model").decide("请检查订单")
    assert result.decision is not None
    assert len(prompts) == 2 and prompts[0] == prompts[1]
    assert result.evidence.total_attempts == 2
    assert result.evidence.retried is True
    assert result.evidence.first_attempt_schema_valid is False
    assert result.evidence.schema_valid is True
    assert result.evidence.attempt == 2
    assert [item.schema_valid for item in result.evidence.attempts] == [False, True]
    assert result.evidence.usage == {"input_tokens": 7, "output_tokens": 3}
    encoded = json.dumps(result.evidence.model_dump(mode="json"), ensure_ascii=False)
    assert all(secret not in encoded for secret in ("SECRET_CASE_ID", "SECRET_GOLD", "SECRET_EXPECTED"))
    assert "raw" not in encoded.lower()


def test_supervisor_schema_invalid_twice_fails_closed_after_two_attempts():
    prompts = []

    class Provider:
        def __call__(self, prompt, _schema):
            prompts.append(prompt)
            return {"topology": "order_only", "requested_capabilities": ()}

    result = R4SupervisorProviderBoundary(Provider()).decide("请检查订单")
    assert result.decision is None
    assert len(prompts) == 2 and prompts[0] == prompts[1]
    assert result.evidence.total_attempts == 2
    assert result.evidence.schema_valid is False
    assert result.evidence.error_code == "R4_SUPERVISOR_SCHEMA_INVALID"
    assert [item.error_code for item in result.evidence.attempts] == [
        "R4_SUPERVISOR_SCHEMA_INVALID",
        "R4_SUPERVISOR_SCHEMA_INVALID",
    ]


def test_supervisor_provider_exception_is_not_retried_or_redacted_with_exception_text():
    class Provider:
        calls = 0

        def __call__(self, _prompt, _schema):
            self.calls += 1
            raise RuntimeError("SECRET_EXCEPTION_DETAIL")

    provider = Provider()
    result = R4SupervisorProviderBoundary(provider).decide("请检查订单")
    assert result.decision is None
    assert provider.calls == 1
    assert result.evidence.total_attempts == 1
    assert result.evidence.error_code == "R4_SUPERVISOR_PROVIDER_ERROR"
    encoded = json.dumps(result.evidence.model_dump(mode="json"), ensure_ascii=False)
    assert "SECRET_EXCEPTION_DETAIL" not in encoded


def test_supervisor_evaluator_reports_first_eventual_retry_and_total_attempt_metrics():
    cases = build_r4_b_dev_inputs()[:2]
    decision = {"topology": "order_only", "requested_capabilities": ("order/read@v1",)}

    class Provider:
        calls = 0

        def __call__(self, _prompt, _schema):
            self.calls += 1
            if self.calls == 2:
                return {"topology": "order_only", "requested_capabilities": ()}
            return decision

    report = evaluate_r4_b_supervisor(cases, Provider())
    metrics = report["metrics"]
    assert metrics["first_attempt_schema_valid_rate"]["numerator"] == 1
    assert metrics["first_attempt_schema_valid_rate"]["denominator"] == 2
    assert metrics["schema_valid_rate"]["numerator"] == 2
    assert metrics["eventual_schema_valid_rate"]["numerator"] == 2
    assert metrics["retried_case_count"] == 1
    assert metrics["total_provider_attempts"] == 3


@pytest.mark.parametrize(
    ("topology", "capabilities", "dependencies"),
    [
        ("order_only", ("order/read@v1",), ()),
        ("logistics_only", ("logistics/read@v1",), ()),
        ("policy_only", ("policy/read@v1",), ()),
        ("order->logistics", ("order/read@v1", "logistics/read@v1"), (("order", "logistics"),)),
        ("order+policy", ("order/read@v1", "policy/read@v1"), ()),
        (
            "order->logistics+policy",
            ("order/read@v1", "logistics/read@v1", "policy/read@v1"),
            (("order", "logistics"),),
        ),
    ],
)
def test_supervisor_topology_capability_dependency_contract_is_closed(topology, capabilities, dependencies):
    decision = R4SupervisorDecisionV1(
        topology=topology,
        requested_capabilities=capabilities,
        dependencies=[
            {"upstream_task_id": upstream, "downstream_task_id": downstream}
            for upstream, downstream in dependencies
        ],
    )
    assert decision.topology == topology
    assert tuple(decision.requested_capabilities) == tuple(sorted(capabilities))


@pytest.mark.parametrize(
    "payload",
    [
        {
            "topology": "order_only",
            "requested_capabilities": ("logistics/read@v1",),
        },
        {
            "topology": "order_only",
            "requested_capabilities": ("order/read@v1",),
            "dependencies": [{"upstream_task_id": "order", "downstream_task_id": "logistics"}],
        },
        {
            "topology": "order->logistics",
            "requested_capabilities": ("order/read@v1",),
            "dependencies": [{"upstream_task_id": "order", "downstream_task_id": "logistics"}],
        },
        {
            "topology": "order+policy",
            "requested_capabilities": ("order/read@v1", "policy/read@v1"),
            "dependencies": [{"upstream_task_id": "order", "downstream_task_id": "policy"}],
        },
        {
            "topology": "policy_only",
            "requested_capabilities": ("policy/read@v1",),
            "dependencies": [{"upstream_task_id": "policy", "downstream_task_id": "order"}],
        },
    ],
)
def test_supervisor_topology_capability_dependency_mismatch_is_rejected(payload):
    with pytest.raises(ValidationError):
        R4SupervisorDecisionV1.model_validate(payload)


def test_supervisor_policy_entities_use_only_canonical_policy_query():
    valid = R4SupervisorDecisionV1(
        topology="policy_only",
        requested_capabilities=("policy/read@v1",),
        entities={"policy_query": "退货期限是多久？"},
    )
    assert valid.entities == {"policy_query": "退货期限是多久？"}
    with pytest.raises(ValidationError):
        R4SupervisorDecisionV1(
            topology="policy_only",
            requested_capabilities=("policy/read@v1",),
            entities={"query": "退货期限是多久？"},
        )


def test_builder_has_exactly_36_unique_gold_free_inputs():
    cases = build_r4_b_dev_inputs()
    assert len(cases) == 36
    assert len({case.case_id for case in cases}) == 36
    assert sum(case.failure_family == "normal" for case in cases) == 18
    assert sum(case.failure_family != "normal" for case in cases) == 18
    assert set(case.topology_family for case in cases if case.failure_family == "normal") == {
        "single_domain",
        "cross_domain_parallel",
        "cross_domain_dependency",
    }
    encoded = json.dumps([case.model_dump(mode="json") for case in cases])
    assert all(token not in encoded for token in ("order_only", "logistics_only", "policy_only", "order->logistics", "order+policy", "order->logistics+policy"))
    assert all(not _contains_forbidden(case.model_dump(mode="python")) for case in cases)


def test_normal_requests_cover_six_natural_semantic_combinations_three_each():
    normal = [case for case in build_r4_b_dev_inputs() if case.failure_family == "normal"]
    assert len(normal) == 18
    groups = [normal[index : index + 3] for index in range(0, 18, 3)]
    keywords = (
        ("订单", "付款"),
        ("包裹", "物流"),
        ("退货", "退款", "换货", "售后"),
        ("订单", "包裹", "物流"),
        ("订单", "退货", "退款", "换货"),
        ("订单", "包裹", "物流", "退货", "退款", "换货"),
    )
    for group, terms in zip(groups, keywords):
        assert len(group) == 3
        assert all(any(term in case.user_text for term in terms) for case in group)
    order_only, logistics_only, policy_only, order_logistics, order_policy, order_logistics_policy = groups
    order_forbidden = ("物流", "包裹", "运单", "配送", "承运商", "退货", "退款", "换货")
    assert all(not any(term in case.user_text for term in order_forbidden) for case in order_only)
    assert all(case.world.order_id in case.user_text for case in order_only)
    assert order_only[1].world.phone_last4 not in order_only[1].user_text
    assert all(case.world.phone_last4 in case.user_text for case in (order_only[0], order_only[2]))

    assert all(case.world.tracking_no in case.user_text for case in logistics_only)
    assert logistics_only[0].world.carrier_code not in logistics_only[0].user_text
    assert all(case.world.carrier_code in case.user_text for case in logistics_only[1:])
    assert all(not any(term in case.user_text for term in ("订单", "退货", "退款", "换货")) for case in logistics_only)

    assert all(case.world.policy_query in case.user_text for case in policy_only)
    assert all(not any(term in case.user_text for term in ("订单", "物流", "包裹", "运单", "配送")) for case in policy_only)

    assert all(case.world.order_id in case.user_text for case in order_logistics)
    assert all("订单" in case.user_text and "物流信息" in case.user_text for case in order_logistics)
    assert all(case.world.carrier_code not in case.user_text and case.world.tracking_no not in case.user_text for case in order_logistics)
    assert order_logistics[1].world.phone_last4 not in order_logistics[1].user_text
    assert all(case.world.phone_last4 in case.user_text for case in (order_logistics[0], order_logistics[2]))

    assert all(case.world.order_id in case.user_text and case.world.policy_query in case.user_text for case in order_policy)
    assert all(not any(term in case.user_text for term in ("物流", "包裹", "运单", "配送", "承运商")) for case in order_policy)
    assert order_policy[0].world.phone_last4 not in order_policy[0].user_text
    assert all(case.world.phone_last4 in case.user_text for case in order_policy[1:])

    assert all(case.world.order_id in case.user_text and case.world.policy_query in case.user_text for case in order_logistics_policy)
    assert all("订单" in case.user_text and "物流信息" in case.user_text for case in order_logistics_policy)
    assert all(case.world.carrier_code not in case.user_text and case.world.tracking_no not in case.user_text for case in order_logistics_policy)
    assert order_logistics_policy[1].world.phone_last4 not in order_logistics_policy[1].user_text
    assert all(case.world.phone_last4 in case.user_text for case in (order_logistics_policy[0], order_logistics_policy[2]))

    clarification_indexes = [index for index, case in enumerate(normal) if (
        (index == 1 and normal[index].world.phone_last4 not in case.user_text)
        or (index == 3 and normal[index].world.carrier_code not in case.user_text)
        or (index == 10 and normal[index].world.phone_last4 not in case.user_text)
        or (index == 12 and normal[index].world.phone_last4 not in case.user_text)
        or (index == 16 and normal[index].world.phone_last4 not in case.user_text)
    )]
    assert clarification_indexes == [1, 3, 10, 12, 16]
    structural = ("order_only", "logistics_only", "policy_only", "capability", "agent", "terminal", "error", "gold")
    normal_encoded = json.dumps([case.model_dump(mode="json") for case in normal], ensure_ascii=False).lower()
    assert all(token not in normal_encoded for token in structural)


def test_supervisor_prompt_snapshot_is_generic_versioned_and_stable():
    forbidden = ("case", "family", "gold", "expected")
    assert not any(token in R4_SUPERVISOR_PROMPT_TEMPLATE.lower() for token in forbidden)
    prompt = build_supervisor_prompt("我想确认订单 R4B-ORD-001 的状态。")
    assert R4_SUPERVISOR_PROMPT_VERSION in prompt
    assert R4_SUPERVISOR_PROMPT_CHECKSUM_SHA256
    assert "R4B-ORD-001" in prompt
    assert build_supervisor_prompt("同一请求") == build_supervisor_prompt("同一请求")


def test_supervisor_prompt_v3_has_generic_semantic_and_canonical_rules():
    template = R4_SUPERVISOR_PROMPT_TEMPLATE.lower()
    assert "r4.supervisor.prompt.v3" in template
    assert "policy_query" in template
    assert "entities.query" in template
    assert "return deadlines" in template
    assert "refund arrival" in template
    assert "exchange conditions" in template
    assert "tracking" in template
    assert "prefix" in template and "hyphen" in template
    assert "order_id" in template and "phone_last4" in template
    assert "order -> logistics" in template
    assert "policy questions are independent" in template
    assert "only allowed dependency edge is order -> logistics" in template
    assert "r4b-ord-001" not in template


def test_supervisor_prompt_topology_mapping_matches_contract_constants_without_leakage():
    lines = [line for line in R4_SUPERVISOR_PROMPT_TEMPLATE.splitlines() if line.startswith("- ")]
    expected_lines = []
    for topology, capabilities in R4_TOPOLOGY_CAPABILITIES.items():
        ordered_capabilities = [
            capability for capability in ("order/read@v1", "logistics/read@v1", "policy/read@v1")
            if capability in capabilities
        ]
        dependency_text = "none" if not R4_TOPOLOGY_DEPENDENCIES[topology] else "order -> logistics"
        expected_lines.append(
            f"- {topology} -> capabilities: {' + '.join(ordered_capabilities)}; dependencies: {dependency_text}."
        )
    assert lines == expected_lines
    template = R4_SUPERVISOR_PROMPT_TEMPLATE.lower()
    assert all(token not in template for token in ("r4b-ord-", "r4b-log-", "gold", "failure family"))


def test_provider_receives_changed_generic_prompt_and_joint_handoff_metric():
    case = build_r4_b_dev_inputs()[0]
    seen_prompts = []
    decision = R4SupervisorDecisionV1(
        requested_capabilities=("order/read@v1",),
        topology="order_only",
        entities={"order_id": case.world.order_id, "phone_last4": case.world.phone_last4},
    )

    class Provider:
        def __call__(self, prompt, _schema):
            seen_prompts.append(prompt)
            return decision

    changed_text = "请帮我确认订单 R4B-ORD-001 的付款状态。"
    boundary_result = R4SupervisorProviderBoundary(Provider()).decide(changed_text)
    assert changed_text in seen_prompts[0]
    assert boundary_result.evidence.prompt_version == R4_SUPERVISOR_PROMPT_VERSION
    report = evaluate_r4_b_supervisor(
        (case,),
        StaticR4SupervisorProvider(decision),
        gold={
            "cases": [{
                "case_id": case.case_id,
                "topology": "order_only",
                "requested_capabilities": ["order/read@v1"],
                "agents": ["order-agent@v1"],
                "dependencies": [],
                "entity_binding": dict(decision.entities),
                "needs_clarification": False,
            }]
        },
    )
    assert report["metrics"]["exact_handoff_joint"]["numerator"] == 1
    assert report["metrics"]["exact_handoff_joint"]["denominator"] == 1


def test_builder_round_trip_and_manifest_evidence_are_deterministic(tmp_path: Path):
    cases = build_r4_b_dev_inputs()
    target = tmp_path / "r4-b-inputs.jsonl"
    manifest = write_r4_b_dev_inputs(target, cases)
    loaded = load_r4_b_inputs(target)
    assert [item.case_id for item in loaded] == [item.case_id for item in cases]
    assert manifest == input_manifest(cases)
    assert manifest["unique_case_N"] == 36
    assert manifest["overlap"]["intersection_N"] == 0
    assert not (tmp_path / "r4-b-gold.jsonl").exists()


def test_modes_share_world_but_have_distinct_call_paths(tmp_path: Path):
    case = build_r4_b_dev_inputs()[0]
    spec = _execution_spec([case])
    observations = {
        mode: run_r4_b_case(case, mode, work_root=tmp_path, execution_spec=spec)
        for mode in R4_B_MODES
    }
    assert len({item["world_fingerprint"] for item in observations.values()}) == 1
    assert observations["single_agent"]["call_counts"]["physical_attempts"] == 0
    assert observations["fixed_order"]["call_counts"]["physical_attempts"] == 3
    assert observations["a2a"]["trace_complete"] is True


def test_duplicate_and_retry_ablations_change_physical_behavior(tmp_path: Path):
    cases = build_r4_b_dev_inputs()
    duplicate = next(case for case in cases if case.failure_family == "duplicate")
    spec = _execution_spec([duplicate])
    a2a_duplicate = run_r4_b_case(duplicate, "a2a", work_root=tmp_path, execution_spec=spec)
    direct_duplicate = run_r4_b_case(duplicate, "direct_call", work_root=tmp_path, execution_spec=spec)
    no_dedup = run_r4_b_case(duplicate, "no_dedup", work_root=tmp_path, execution_spec=spec)
    assert a2a_duplicate["call_counts"]["physical_attempts"] == 1
    assert direct_duplicate["call_counts"]["physical_attempts"] == 2
    assert no_dedup["call_counts"]["physical_attempts"] == 2
    timeout = next(case for case in cases if case.failure_family == "recoverable_timeout")
    timeout_spec = _execution_spec([timeout])
    a2a_retry = run_r4_b_case(timeout, "a2a", work_root=tmp_path, execution_spec=timeout_spec)
    no_retry = run_r4_b_case(timeout, "no_retry", work_root=tmp_path, execution_spec=timeout_spec)
    assert a2a_retry["dispatches"]["order"]["attempt_count"] == 2
    assert no_retry["dispatches"]["order"]["attempt_count"] == 1


def test_disabled_specialist_has_zero_calls_for_named_capability(tmp_path: Path):
    case = build_r4_b_dev_inputs()[0]
    spec = {case.case_id: {"case_id": case.case_id, "topology": "policy_only", "disabled_specialist": "policy"}}
    observation = run_r4_b_case(case, "disabled_specialist", work_root=tmp_path, execution_spec=spec, disabled_specialist="policy")
    assert observation["dispatches"]["policy"]["error_code"] == "SPECIALIST_DISABLED"
    assert observation["call_counts"]["physical_by_capability"].get("policy/read@v1", 0) == 0
    assert observation["execution"]["ledger_used"] is True
    assert observation["execution"]["verifier_enabled"] is True
    assert "policy/read@v1" not in observation["execution"]["capability_set"]


def test_evaluator_accepts_only_explicit_separate_gold_and_reports_metrics(tmp_path: Path):
    cases = build_r4_b_dev_inputs()[:2]
    gold_path = tmp_path / "external-gold.json"
    gold_path.write_text(
        json.dumps({"cases": [{"case_id": item.case_id, "status": "SUCCEEDED"} for item in cases]}),
        encoding="utf-8",
    )
    report = evaluate_r4_b(cases, work_root=tmp_path / "runs", modes=("a2a",), gold=gold_path, input_path=tmp_path / "inputs.jsonl", execution_spec=_execution_spec(cases))
    assert report["gold"]["used"] is True
    metric = report["modes"]["a2a"]["metrics"]["status"]
    assert {"numerator", "denominator", "unique_case_N", "wilson95"}.issubset(metric)
    with pytest.raises(ValueError):
        evaluate_r4_b(cases, work_root=tmp_path / "same", modes=("a2a",), gold=gold_path, input_path=gold_path)


def test_no_external_execution_spec_does_not_run_coordination(tmp_path: Path):
    case = build_r4_b_dev_inputs()[0]
    report = evaluate_r4_b((case,), work_root=tmp_path, modes=("a2a", "single_agent"))
    assert report["status"] == "NOT_RUN_NO_EXECUTION_SPEC"
    assert report["modes"]["a2a"]["observations"] == []
    assert report["modes"]["single_agent"]["observations"][0]["status"] == "NOT_RUN_NO_PROVIDER"


def test_single_agent_without_provider_has_no_quality_denominator(tmp_path: Path):
    case = build_r4_b_dev_inputs()[0]
    spec = _execution_spec([case])
    gold = {"cases": [{"case_id": case.case_id, "normal_e2e_success": True, "status": "SUCCEEDED"}]}
    report = evaluate_r4_b((case,), modes=("single_agent",), execution_spec=spec, gold=gold, work_root=tmp_path)
    assert report["modes"]["single_agent"]["status"] == "NOT_RUN_NO_PROVIDER"
    assert "normal_e2e_success" not in report["modes"]["single_agent"]["metrics"]


def test_user_text_mutation_cannot_change_execution_plan(tmp_path: Path):
    case = build_r4_b_dev_inputs()[0]
    mutated = case.model_copy(update={"user_text": "Please do something completely different."})
    spec = _execution_spec([case])
    original = run_r4_b_case(case, "a2a", work_root=tmp_path / "original", execution_spec=spec)
    changed = run_r4_b_case(mutated, "a2a", work_root=tmp_path / "changed", execution_spec={mutated.case_id: spec[case.case_id]})
    assert original["status"] == changed["status"] == "SUCCEEDED"
    assert original["execution"]["capability_set"] == changed["execution"]["capability_set"]


def test_gold_ids_are_exact_and_duplicate_ids_are_rejected(tmp_path: Path):
    cases = build_r4_b_dev_inputs()[:2]
    spec = _execution_spec(cases)
    for rows in (
        [{"case_id": cases[0].case_id}],
        [{"case_id": cases[0].case_id}, {"case_id": cases[1].case_id}, {"case_id": "extra"}],
        [{"case_id": cases[0].case_id}, {"case_id": cases[0].case_id}],
    ):
        with pytest.raises(ValueError):
            evaluate_r4_b(cases, modes=("a2a",), execution_spec=spec, gold={"cases": rows})


def test_dependency_release_is_not_duplicate_and_protocol_components_are_scored(tmp_path: Path):
    case = next(item for item in build_r4_b_dev_inputs() if item.failure_family == "dependency_pending")
    spec = _execution_spec([case])
    observation = run_r4_b_case(case, "a2a", work_root=tmp_path, execution_spec=spec)
    assert observation["dispatches"]["logistics_replay"]["replay_reason"] == "dependency_release"
    assert observation["dispatches"]["logistics_replay"]["duplicate"] is False
    gold = {
        "cases": [{
            "case_id": case.case_id,
            "status": observation["status"],
            "trace_complete": True,
            "physical_attempts": observation["call_counts"]["physical_attempts"],
            "protocol_safe_success": True,
        }]
    }
    report = evaluate_r4_b((case,), modes=("a2a",), execution_spec=spec, gold=gold, work_root=tmp_path / "score")
    assert report["modes"]["a2a"]["metrics"]["protocol_safe_success"]["unique_case_N"] == 1

    wrong_gold = {
        "cases": [{
            "case_id": case.case_id,
            "status": observation["status"],
            "trace_complete": True,
            "physical_attempts": observation["call_counts"]["physical_attempts"] + 1,
            "protocol_safe_success": True,
        }]
    }
    wrong_report = evaluate_r4_b((case,), modes=("a2a",), execution_spec=spec, gold=wrong_gold, work_root=tmp_path / "wrong-score")
    assert wrong_report["modes"]["a2a"]["metrics"]["protocol_safe_success"]["numerator"] == 0


def test_metric_denominator_and_unique_n_use_only_declared_metric_cases(tmp_path: Path):
    cases = build_r4_b_dev_inputs()[:2]
    spec = _execution_spec(cases)
    gold = {
        "cases": [
            {"case_id": cases[0].case_id, "trace_complete": True},
            {"case_id": cases[1].case_id},
        ]
    }
    report = evaluate_r4_b(cases, modes=("a2a",), execution_spec=spec, gold=gold, work_root=tmp_path)
    metric = report["modes"]["a2a"]["metrics"]["trace_complete"]
    assert metric["denominator"] == 1
    assert metric["unique_case_N"] == 1


def test_non_retryable_and_semantic_wrong_are_distinct_error_families(tmp_path: Path):
    cases = build_r4_b_dev_inputs()
    non_retryable = next(item for item in cases if item.failure_family == "non_retryable")
    semantic_wrong = next(item for item in cases if item.failure_family == "semantic_wrong")
    one = run_r4_b_case(non_retryable, "a2a", work_root=tmp_path, execution_spec=_execution_spec([non_retryable]))
    two = run_r4_b_case(semantic_wrong, "a2a", work_root=tmp_path, execution_spec=_execution_spec([semantic_wrong]))
    assert one["dispatches"]["order"]["error_code"] == "NON_RETRYABLE_FAULT"
    assert two["dispatches"]["order"]["error_code"] == "A2A_SEMANTIC_WRONG"
    assert one["dispatches"]["order"]["error_code"] != two["dispatches"]["order"]["error_code"]


def test_model_result_metadata_crosses_provider_boundary_without_static_oracle():
    decision = {"requested_capabilities": ("order/read@v1",), "topology": "order_only"}

    def provider(_text, _schema):
        return ModelResult(decision, ModelUsage(input_tokens=7, output_tokens=3, available=True), "model-from-result")

    evidence = R4SupervisorProviderBoundary(provider).decide("changed user text").evidence
    assert evidence.model_ref == "model-from-result"
    assert evidence.usage == {"input_tokens": 7, "output_tokens": 3}
    assert evidence.prompt_version == R4_SUPERVISOR_PROMPT_VERSION
    assert evidence.prompt_checksum_sha256 == R4_SUPERVISOR_PROMPT_CHECKSUM_SHA256
