"""Deterministic dependency scheduler with parallel join and partial policy."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Mapping

from .domain.objects import Task, TaskStatus


@dataclass(frozen=True)
class ScheduledResult:
    task_id: str
    status: str
    value: object = None
    error: str | None = None


@dataclass(frozen=True)
class ScheduleReport:
    results: tuple[ScheduledResult, ...]
    partial: bool

    @property
    def by_task(self) -> dict[str, ScheduledResult]:
        return {item.task_id: item for item in self.results}


class TaskScheduler:
    def __init__(self, tasks: list[Task], *, worker: Callable[[Task, Mapping[str, object]], object]):
        self.tasks = {task.task_id: task for task in tasks}
        self.worker = worker
        if len(self.tasks) != len(tasks):
            raise ValueError("task IDs must be unique")
        for task in tasks:
            if any(dep not in self.tasks for dep in task.depends_on):
                raise ValueError(f"missing dependency for {task.task_id}")

    def run(self) -> ScheduleReport:
        results: dict[str, ScheduledResult] = {}
        pending = set(self.tasks)
        while pending:
            ready = [self.tasks[task_id] for task_id in sorted(pending) if all(dep in results for dep in self.tasks[task_id].depends_on)]
            if not ready:
                raise ValueError("dependency graph cannot make progress")
            runnable: list[Task] = []
            for task in ready:
                deps = [results[dep] for dep in task.depends_on]
                failed = [item for item in deps if item.status not in {TaskStatus.SUCCEEDED.value, "SUCCEEDED"}]
                if failed and task.failure_strategy != "CONTINUE_PARTIAL":
                    results[task.task_id] = ScheduledResult(task.task_id, TaskStatus.BLOCKED.value, error="dependency_failed")
                else:
                    runnable.append(task)
                pending.remove(task.task_id)
            if runnable:
                def execute(task: Task) -> ScheduledResult:
                    inputs = {dep: results[dep].value for dep in task.depends_on if results[dep].status == TaskStatus.SUCCEEDED.value}
                    try:
                        value = self.worker(task, inputs)
                        # Typed tool adapters return an explicit failed result
                        # rather than raising; preserve that outcome so a
                        # CONTINUE_PARTIAL join can retain the failure branch.
                        failed = getattr(value, "ok", True) is False
                        return ScheduledResult(task.task_id, TaskStatus.FAILED.value if failed else TaskStatus.SUCCEEDED.value, value=value, error="tool_failed" if failed else None)
                    except Exception as exc:
                        return ScheduledResult(task.task_id, TaskStatus.FAILED.value, error=type(exc).__name__)
                with ThreadPoolExecutor(max_workers=len(runnable)) as pool:
                    for outcome in pool.map(execute, runnable):
                        results[outcome.task_id] = outcome
        return ScheduleReport(tuple(results[key] for key in sorted(results)), any(item.status != TaskStatus.SUCCEEDED.value for item in results.values()))


__all__ = ["ScheduleReport", "ScheduledResult", "TaskScheduler"]
