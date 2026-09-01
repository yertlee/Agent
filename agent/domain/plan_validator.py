"""Fail-closed DAG and binding validator for canonical PlanRevision."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, Mapping, Optional

from .objects import PlanRevision, Task


@dataclass
class PlanValidationError(ValueError):
    errors: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        super().__init__("invalid plan: " + "; ".join(self.errors))


@dataclass(frozen=True)
class PlanLimits:
    max_tasks: int = 32
    max_iterations: int = 128
    max_timeout_ms: int = 300_000


class PlanValidator:
    def __init__(
        self,
        *,
        agents: Optional[Iterable[str]] = None,
        capabilities: Optional[Iterable[str]] = None,
        limits: PlanLimits = PlanLimits(),
    ) -> None:
        self.agents = set(agents or ())
        self.capabilities = set(capabilities or ())
        self.limits = limits

    def errors(self, revision: PlanRevision, *, now: Optional[datetime] = None) -> list[str]:
        problems: list[str] = []
        tasks = revision.tasks
        ids = [task.task_id for task in tasks]
        if len(ids) != len(set(ids)):
            problems.append("task IDs must be unique")
        if len(tasks) > self.limits.max_tasks:
            problems.append("max_tasks exceeded")
        known = set(ids)
        graph = {t.task_id: set(t.depends_on) for t in tasks}
        def dependency_closure(task_id: str) -> set[str]:
            seen: set[str] = set()
            stack = list(graph.get(task_id, ()))
            while stack:
                dep = stack.pop()
                if dep in seen: continue
                seen.add(dep); stack.extend(graph.get(dep, ()))
            return seen
        for task in tasks:
            if task.plan_revision_id != revision.plan_revision_id:
                problems.append(f"task {task.task_id} has wrong plan parent")
            if self.agents and task.agent_ref not in self.agents:
                problems.append(f"unknown agent_ref: {task.agent_ref}")
            if len(task.capability_refs) != len(set(task.capability_refs)):
                problems.append(f"duplicate capability_refs: {task.task_id}")
            missing_caps = sorted(set(task.capability_refs) - self.capabilities) if self.capabilities else []
            if missing_caps:
                problems.append(f"unknown capability_refs: {','.join(missing_caps)}")
            for dep in task.depends_on:
                if dep not in known:
                    problems.append(f"missing dependency: {task.task_id}->{dep}")
            for binding in task.input_bindings:
                if binding.kind == "result" and binding.source_task_id not in dependency_closure(task.task_id):
                    problems.append(f"binding source outside dependency closure: {task.task_id}->{binding.source_task_id}")
            if task.timeout_ms > self.limits.max_timeout_ms:
                problems.append(f"task timeout exceeds limit: {task.task_id}")
            if task.deadline is not None:
                check_now = now or datetime.now(timezone.utc)
                deadline = task.deadline
                if deadline.tzinfo is None:
                    deadline = deadline.replace(tzinfo=timezone.utc)
                if deadline <= check_now:
                    problems.append(f"deadline expired: {task.task_id}")
        problems.extend(self._cycle_errors(tasks))
        return problems

    @staticmethod
    def _cycle_errors(tasks: list[Task]) -> list[str]:
        graph = {t.task_id: list(t.depends_on) for t in tasks}
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node: str) -> bool:
            if node in visiting:
                return True
            if node in visited:
                return False
            visiting.add(node)
            if any(dep in graph and visit(dep) for dep in graph.get(node, ())):
                return True
            visiting.remove(node)
            visited.add(node)
            return False

        return ["dependency graph contains a cycle"] if any(visit(n) for n in graph) else []

    def validate(self, revision: PlanRevision, *, now: Optional[datetime] = None) -> PlanRevision:
        problems = self.errors(revision, now=now)
        if problems:
            raise PlanValidationError(problems)
        return revision


def validate_plan(revision: PlanRevision, **kwargs) -> PlanRevision:
    return PlanValidator(**kwargs).validate(revision)
