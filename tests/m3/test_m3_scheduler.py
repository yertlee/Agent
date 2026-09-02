from agent.domain.objects import Task, TaskStatus
from agent.m3_scheduler import TaskScheduler


def task(tid, deps=None, strategy="FAIL_RUN"):
    return Task(task_id=tid, plan_revision_id="p", agent_ref="order-agent@v1", capability_refs=["order/read@v1"], depends_on=deps or [], output_contract="order.result.v1", failure_strategy=strategy, side_effect="READ_ONLY", timeout_ms=1000)


def test_parallel_join_retains_both_branch_results():
    seen = []
    def worker(t, inputs):
        seen.append(t.task_id)
        return t.task_id
    report = TaskScheduler([task("a"), task("b"), task("join", ["a", "b"])], worker=worker).run()
    assert report.by_task["join"].status == "SUCCEEDED"
    assert report.by_task["join"].value == "join"
    assert set(seen) == {"a", "b", "join"}


def test_failed_dependency_blocks_by_default_and_partial_can_continue():
    def worker(t, inputs):
        if t.task_id == "a":
            raise RuntimeError("fixture fault")
        return sorted(inputs)
    blocked = TaskScheduler([task("a"), task("b", ["a"])], worker=worker).run()
    assert blocked.by_task["b"].status == TaskStatus.BLOCKED.value
    partial = TaskScheduler([task("a"), task("b", ["a"], "CONTINUE_PARTIAL")], worker=worker).run()
    assert partial.by_task["b"].status == TaskStatus.SUCCEEDED.value
    assert partial.partial is True
