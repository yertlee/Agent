from datetime import datetime, timezone
from pathlib import Path

from agent.domain.objects import AttemptStatus, PlanRevision, PlanStatus, Run, Task, TaskAttempt, sha256_json
from agent.storage.m2 import M2Repository


def setup_repo(tmp_path: Path):
    repo = M2Repository(str(tmp_path / "m2.db"))
    repo.create_session("s1", "u1")
    repo.create_run(Run(run_id="r1", session_id="s1", initial_world_hash=sha256_json({"world": "m2"})))
    task = Task(task_id="t1", plan_revision_id="p1", agent_ref="aftersales@v1", capability_refs=["aftersales/write@v1"], output_contract="aftersales.result.v1", failure_strategy="FAIL_RUN", side_effect="WRITE", timeout_ms=5000)
    plan = PlanRevision(plan_revision_id="p1", run_id="r1", created_by="supervisor", revision_reason="initial", version=1, status=PlanStatus.ACTIVE, tasks=[task])
    repo.create_plan_revision(plan)
    repo.create_attempt(TaskAttempt(attempt_id="a1", run_id="r1", plan_revision_id="p1", task_id="t1", agent_ref=task.agent_ref, attempt_no=1, status=AttemptStatus.CREATED, input_hash=sha256_json({"input": 1})))
    return repo, task


def order_fact(user_id="u1", order_id="o1", *, status="PAID", quality="FRESH"):
    from agent.domain.facts import DataQuality, OrderFact
    return OrderFact(fact_id=f"fact_{order_id}", entity_id=order_id, user_id=user_id, status=status, amount=10.0, items=[], created_at=datetime(2026, 1, 1, tzinfo=timezone.utc), version="order.v1", source="system", data_quality=quality)
