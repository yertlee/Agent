import unittest
from datetime import datetime, timedelta, timezone

from agent.domain.objects import InputBinding, PlanRevision, PlanStatus, Run, Task, sha256_json
from agent.trace.events import TraceEvent
from agent.domain.plan_validator import PlanValidationError, PlanValidator


def task(tid, revision="p1", deps=None, bindings=None):
    return Task(task_id=tid, plan_revision_id=revision, agent_ref="order@v1", capability_refs=["order/read@v1"], depends_on=deps or [], input_bindings=bindings or [], output_contract="order.read.v1", failure_strategy="FAIL_RUN", side_effect="READ_ONLY", timeout_ms=1000)


class M1ObjectContractTests(unittest.TestCase):
    def revision(self, tasks):
        return PlanRevision(plan_revision_id="p1", run_id="r1", created_by="supervisor", revision_reason="initial", version=1, status=PlanStatus.DRAFT, tasks=tasks)

    def test_valid_dag_and_result_binding_closure(self):
        binding = InputBinding(name="order", kind="result", source_task_id="a", path="payload.order_id")
        revision = self.revision([task("a"), task("b", deps=["a"], bindings=[binding])])
        self.assertIs(PlanValidator(agents={"order@v1"}, capabilities={"order/read@v1"}).validate(revision), revision)

    def test_duplicate_missing_cycle_unknown_and_deadline_are_rejected(self):
        validator = PlanValidator(agents={"order@v1"}, capabilities={"order/read@v1"})
        cases = [
            [task("a"), task("a")],
            [task("a", deps=["missing"])],
            [task("a", deps=["b"]), task("b", deps=["a"])],
            [Task(**{**task("a").model_dump(), "agent_ref": "bad@v1"})],
            [Task(**{**task("a").model_dump(), "deadline": datetime.now(timezone.utc) - timedelta(seconds=1)})],
        ]
        for tasks in cases:
            with self.assertRaises(PlanValidationError): validator.validate(self.revision(tasks))

    def test_binding_model_rejects_incomplete_result_and_confirm_binding(self):
        with self.assertRaises(ValueError): InputBinding(name="x", kind="result")
        with self.assertRaises(ValueError): InputBinding(name="x", kind="confirm_token")

    def test_unknown_fields_and_bad_hash_fail_closed(self):
        with self.assertRaises(ValueError): InputBinding(name="x", kind="invocation_context", extra="x")
        with self.assertRaises(ValueError): InputBinding(name="x", kind="invocation_context")
        with self.assertRaises(ValueError): Task(task_id="t", plan_revision_id="p", agent_ref="a", capability_refs=[], output_contract="x", failure_strategy="FAIL_RUN", side_effect="READ_ONLY", timeout_ms=1)
        with self.assertRaises(ValueError): PlanRevision(plan_revision_id="p", run_id="r", created_by="model", revision_reason="initial", version=1)
        with self.assertRaises(ValueError): Run(run_id="r", session_id="s", initial_world_hash="short")
        with self.assertRaises(ValueError): TraceEvent(trace_id="e", run_id="r", session_id="s", seq_no=1, event_type="RUN_CREATED", occurred_at="2026-01-01T00:00:00Z", actor="unknown", payload={})
