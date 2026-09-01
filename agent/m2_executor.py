"""M2 sync/async executor with trusted context and cancellation barriers."""
from __future__ import annotations

import inspect
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

from .m2_context import InvocationContext
from .m2_errors import ErrorCatalog, ErrorEnvelope
from .m2_registry import Registry, ToolSpec


@dataclass(frozen=True)
class ExecutionResult:
    ok: bool
    data: Any
    error: Optional[ErrorEnvelope]
    logical_call_no: int
    physical_attempt_no: int


class ExecutionCancelled(RuntimeError):
    pass


class M2Executor:
    def __init__(self, registry: Registry):
        self.registry = registry
        self._lock = threading.Lock()
        self._logical_calls: dict[tuple[str, str], int] = {}

    def _next_logical_call(self, context: InvocationContext) -> int:
        key = (context.run_id, context.task_id)
        with self._lock:
            value = self._logical_calls.get(key, 0) + 1
            self._logical_calls[key] = value
            return value

    @staticmethod
    def _barrier(context: InvocationContext, cancelled: Callable[[], bool]) -> None:
        if context.cancellation or cancelled():
            raise ExecutionCancelled("cancellation barrier is active")
        if context.is_expired():
            raise TimeoutError("invocation deadline exceeded")

    @staticmethod
    def _authorize(spec: ToolSpec, context: InvocationContext, capability_ref: Optional[str]) -> None:
        # Authorization is explicit and bound to both the agent owner and scope.
        if not capability_ref:
            raise PermissionError("explicit capability authorization is required")
        owner = context.agent_ref.split("@", 1)[0]
        expected_owner = spec.owner.split("@", 1)[0]
        if owner != expected_owner:
            raise PermissionError("invocation agent is not the registered owner")
        if context.auth_scope != spec.capability_ref:
            raise PermissionError("invocation scope is not the requested capability")

    @staticmethod
    def _call_sync(spec: ToolSpec, args: Mapping[str, Any]) -> Any:
        # A read-only callable can be bounded. Writes use a completion barrier:
        # once started, wait for the callable to finish before reporting status.
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(spec.callable, **dict(args))
            if spec.side_effect == "WRITE":
                return future.result()
            try:
                return future.result(timeout=spec.timeout_ms / 1000)
            except FutureTimeoutError as exc:
                future.cancel()
                raise TimeoutError("tool timeout exceeded") from exc

    def invoke(
        self,
        tool_ref: str,
        context: InvocationContext,
        args: Mapping[str, Any],
        *,
        capability_ref: Optional[str] = None,
        cancelled: Optional[Callable[[], bool]] = None,
    ) -> ExecutionResult:
        cancelled = cancelled or (lambda: False)
        logical = self._next_logical_call(context)
        physical = 1
        try:
            spec = self.registry.require_capability(tool_ref, capability_ref or "")
            self._authorize(spec, context, capability_ref)
            spec.validate_candidate_args(args)
            self._barrier(context, cancelled)
            value = self._call_sync(spec, args)
            if spec.side_effect != "WRITE":
                self._barrier(context, cancelled)
            if inspect.isawaitable(value):
                raise TypeError("async callable requires invoke_async")
            if isinstance(value, Mapping) and value.get("success") is False:
                error_code = ErrorCatalog.canonicalize(str(value.get("code") or "TOOL_EXECUTION_FAILED"))
                if error_code not in spec.allowed_error_codes:
                    return ExecutionResult(False, None, ErrorCatalog.envelope("TOOL_CONTRACT_VIOLATION", details={"reason": "error_code_not_allowed"}, trace_id=context.trace_id), logical, physical)
                error = ErrorCatalog.envelope(error_code, details={"source": "tool_result"}, trace_id=context.trace_id)
                return ExecutionResult(False, None, error, logical, physical)
            return ExecutionResult(True, value, None, logical, physical)
        except ExecutionCancelled:
            return ExecutionResult(False, None, ErrorCatalog.envelope("CANCELLED", trace_id=context.trace_id), logical, physical)
        except TimeoutError:
            return ExecutionResult(False, None, ErrorCatalog.envelope("DEADLINE_EXCEEDED", trace_id=context.trace_id), logical, physical)
        except PermissionError as exc:
            return ExecutionResult(False, None, ErrorCatalog.envelope("AUTH_CAPABILITY_DENIED", details={"reason": type(exc).__name__}, trace_id=context.trace_id), logical, physical)
        except (KeyError, ValueError) as exc:
            return ExecutionResult(False, None, ErrorCatalog.envelope("TOOL_CONTRACT_VIOLATION", details={"reason": type(exc).__name__}, trace_id=context.trace_id), logical, physical)
        except Exception as exc:
            return ExecutionResult(False, None, ErrorCatalog.envelope("TOOL_EXECUTION_FAILED", details={"reason": type(exc).__name__}, trace_id=context.trace_id), logical, physical)

    async def invoke_async(
        self,
        tool_ref: str,
        context: InvocationContext,
        args: Mapping[str, Any],
        *,
        capability_ref: Optional[str] = None,
        cancelled: Optional[Callable[[], bool]] = None,
    ) -> ExecutionResult:
        cancelled = cancelled or (lambda: False)
        logical = self._next_logical_call(context)
        physical = 1
        try:
            spec: ToolSpec = self.registry.require_capability(tool_ref, capability_ref or "")
            self._authorize(spec, context, capability_ref)
            spec.validate_candidate_args(args)
            self._barrier(context, cancelled)
            value = spec.callable(**dict(args))
            if inspect.isawaitable(value):
                if spec.side_effect == "WRITE":
                    value = await value
                else:
                    value = await asyncio.wait_for(value, timeout=spec.timeout_ms / 1000)
            if spec.side_effect != "WRITE":
                self._barrier(context, cancelled)
            if isinstance(value, Mapping) and value.get("success") is False:
                error_code = ErrorCatalog.canonicalize(str(value.get("code") or "TOOL_EXECUTION_FAILED"))
                if error_code not in spec.allowed_error_codes:
                    return ExecutionResult(False, None, ErrorCatalog.envelope("TOOL_CONTRACT_VIOLATION", details={"reason": "error_code_not_allowed"}, trace_id=context.trace_id), logical, physical)
                error = ErrorCatalog.envelope(error_code, details={"source": "tool_result"}, trace_id=context.trace_id)
                return ExecutionResult(False, None, error, logical, physical)
            return ExecutionResult(True, value, None, logical, physical)
        except ExecutionCancelled:
            return ExecutionResult(False, None, ErrorCatalog.envelope("CANCELLED", trace_id=context.trace_id), logical, physical)
        except (TimeoutError, FutureTimeoutError, asyncio.TimeoutError):
            return ExecutionResult(False, None, ErrorCatalog.envelope("DEADLINE_EXCEEDED", trace_id=context.trace_id), logical, physical)
        except PermissionError as exc:
            return ExecutionResult(False, None, ErrorCatalog.envelope("AUTH_CAPABILITY_DENIED", details={"reason": type(exc).__name__}, trace_id=context.trace_id), logical, physical)
        except (KeyError, ValueError) as exc:
            return ExecutionResult(False, None, ErrorCatalog.envelope("TOOL_CONTRACT_VIOLATION", details={"reason": type(exc).__name__}, trace_id=context.trace_id), logical, physical)
        except Exception as exc:
            return ExecutionResult(False, None, ErrorCatalog.envelope("TOOL_EXECUTION_FAILED", details={"reason": type(exc).__name__}, trace_id=context.trace_id), logical, physical)


__all__ = ["ExecutionCancelled", "ExecutionResult", "M2Executor"]
