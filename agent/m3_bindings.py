"""Typed binding resolution with dependency, run and hash closure checks."""
from __future__ import annotations

from typing import Any, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field

from .domain.objects import InputBinding, Result, Task, canonical_json, sha256_json
from .m2_context import InvocationContext


CONTEXT_KEYS = {"session_id", "user_id", "run_id", "plan_revision_id", "task_id", "attempt_id", "agent_ref", "auth_scope", "idempotency_key", "deadline", "cancellation", "config_version", "registry_version", "dataset_version", "trace_id"}


class ResultBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    kind: str = "result"
    source_task_id: str = Field(min_length=1)
    path: str = Field(min_length=1)
    expected_contract: str = Field(min_length=1)
    expected_type: str = Field(min_length=1)


class ContextBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    kind: str = "invocation_context"
    path: str = Field(min_length=1)


class ConfirmTokenBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    kind: str = "confirm_token"
    token_purpose: str = Field(min_length=1)


class BindingError(ValueError):
    pass


def dependency_closure(tasks: Mapping[str, Task], task_id: str) -> set[str]:
    if task_id not in tasks:
        raise BindingError(f"unknown task: {task_id}")
    seen: set[str] = set()
    stack = list(tasks[task_id].depends_on)
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        if current not in tasks:
            raise BindingError(f"missing dependency: {current}")
        seen.add(current)
        stack.extend(tasks[current].depends_on)
    return seen


class BindingResolver:
    def __init__(self, tasks: Mapping[str, Task]):
        self.tasks = dict(tasks)

    def validate(self, binding: InputBinding | ResultBinding | ContextBinding | ConfirmTokenBinding, *, target_task_id: str) -> None:
        if isinstance(binding, ResultBinding):
            target = self.tasks.get(target_task_id)
            if target is None:
                raise BindingError(f"unknown target task: {target_task_id}")
            if binding.source_task_id not in dependency_closure(self.tasks, target_task_id):
                raise BindingError("result binding source is outside dependency closure")
            source = self.tasks[binding.source_task_id]
            if source.output_contract != binding.expected_contract:
                raise BindingError("result binding contract mismatch")
            return
        if isinstance(binding, InputBinding):
            if binding.kind == "result":
                self.validate(ResultBinding(name=binding.name, source_task_id=str(binding.source_task_id), path=binding.path, expected_contract=self.tasks[str(binding.source_task_id)].output_contract, expected_type="Any"), target_task_id=target_task_id)
            elif binding.kind == "invocation_context":
                self.validate(ContextBinding(name=binding.name, path=binding.path), target_task_id=target_task_id)
            else:
                self.validate(ConfirmTokenBinding(name=binding.name, token_purpose=str(binding.token_purpose)), target_task_id=target_task_id)
            return
        if isinstance(binding, ContextBinding):
            if binding.path not in CONTEXT_KEYS:
                raise BindingError(f"context path not allowed: {binding.path}")
            return
        if isinstance(binding, ConfirmTokenBinding):
            return
        raise BindingError("unknown binding type")

    @staticmethod
    def _path(payload: Any, path: str) -> Any:
        if path in {"", "payload"}:
            return payload
        current = payload
        for part in path.split("."):
            if part == "":
                raise BindingError("binding path is empty")
            if isinstance(current, BaseModel):
                current = getattr(current, part)
            elif isinstance(current, Mapping) and part in current:
                current = current[part]
            else:
                raise BindingError(f"binding path does not resolve: {path}")
        return current

    def resolve_result(self, binding: ResultBinding, *, target_task_id: str, run_id: str, plan_revision_id: str, results: Mapping[str, Result | Mapping[str, Any]]) -> Any:
        self.validate(binding, target_task_id=target_task_id)
        if binding.source_task_id not in results:
            raise BindingError("source Result is not persisted")
        source = results[binding.source_task_id]
        if isinstance(source, Result):
            if source.run_id != run_id or source.plan_revision_id != plan_revision_id:
                raise BindingError("cross-run or cross-plan result reference")
            payload = source.payload
            if source.payload_hash != sha256_json(payload):
                raise BindingError("source Result payload hash mismatch")
            if source.output_contract != binding.expected_contract:
                raise BindingError("source Result contract mismatch")
        else:
            if str(source.get("run_id")) != run_id or str(source.get("plan_revision_id")) != plan_revision_id:
                raise BindingError("cross-run or cross-plan result reference")
            payload = source.get("payload")
            if source.get("payload_hash") != sha256_json(payload):
                raise BindingError("source Result payload hash mismatch")
            if source.get("output_contract") != binding.expected_contract:
                raise BindingError("source Result contract mismatch")
        return self._path(payload, binding.path.removeprefix("payload."))

    @staticmethod
    def resolve_context(binding: ContextBinding, context: InvocationContext) -> Any:
        if binding.path not in CONTEXT_KEYS:
            raise BindingError(f"context path not allowed: {binding.path}")
        return getattr(context, binding.path)


__all__ = ["BindingError", "BindingResolver", "ConfirmTokenBinding", "ContextBinding", "ResultBinding", "dependency_closure"]
