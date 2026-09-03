from eval.compare_m4 import _sequences
from eval.comparator import BaselineComparator, ComparableRun
from eval.datasets import make_manifest


def test_pair_contract_uses_all_72_ids_and_only_declared_variations():
    manifest = make_manifest(split="dev", dataset_version="m4.dev72.v1")
    outputs = [{"scenario_id": case.scenario_id, "status": "PASS", "task_completion": True,
                "intent": case.expected_intent, "tool_path": list(case.expected_tool_path),
                "case_pass": True, "world_fingerprint": case.expected_world_fingerprint} for case in manifest.cases]
    values = _sequences(manifest, outputs)
    assert all(len(sequence) == 72 for sequence in values.values())
    left_version = dict(manifest.cases[0].version_tuple); left_version.update(code="baseline", harness="adapter")
    right_version = dict(manifest.cases[0].version_tuple); right_version.update(code="candidate", harness="harness")
    left = ComparableRun("left", left_version, manifest.dataset_version, tuple(c.scenario_id for c in manifest.cases), "gold", "m4.evaluator-registry.v1", values)
    right = ComparableRun("right", right_version, manifest.dataset_version, tuple(c.scenario_id for c in manifest.cases), "gold", "m4.evaluator-registry.v1", values)
    assert BaselineComparator().compare(left, right, allowed_variations={"code", "harness"}).comparable
