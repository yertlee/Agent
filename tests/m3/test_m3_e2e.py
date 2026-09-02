from eval.m3_e2e import run_manifest


def test_dev44_contract_e2e_and_trajectory_report():
    report = run_manifest("eval/manifests/dev44.yaml")
    assert report["denominator"]["N_applicable"] == 44
    assert report["trajectory"]["valid_runs"] == 44
    # Assert the architecture gate, not a fixture-specific pass count.  The
    # evaluator keeps business-error PASS runs eligible while true infra and
    # blocked runs remain failures in the denominator.
    assert report["metrics"]["task_completion"] >= 0.90
    assert report["metrics"]["intent_accuracy"] >= 0.90
    assert report["metrics"]["tool_path"] == 1.0
    assert report["denominator"]["N_failed"] == 2
    assert report["denominator"]["PASS"] + report["denominator"]["FAILED"] + report["denominator"]["BLOCKED"] + report["denominator"]["CANCELLED"] == 44
    assert report["denominator"]["N_assertions_pass"] == 44
