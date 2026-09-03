"""Mode-bounded ToolSimulator using the public AgentPort boundary."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from agent.agents import AfterSalesAgent, AgentPort, LogisticsAgent, OrderAgent, PolicyAgent, ProductAgent, TypedAgentResult
from agent.m2_context import InvocationContext
from agent.m3_registry import M3Registry, build_m3_registry
from agent.domain.objects import sha256_json

from .contracts import ExecutionMode, FailureScript, FailureTrigger, WorldSnapshot


_PORT_TYPES = {
    "order/get_info@v1": OrderAgent,
    "aftersales/query@v1": AfterSalesAgent,
    "aftersales/create@v1": AfterSalesAgent,
    "logistics/query@v1": LogisticsAgent,
    "policy/search@v1": PolicyAgent,
    "product/get@v1": ProductAgent,
}


@dataclass(frozen=True)
class HarnessToolResult:
    typed: TypedAgentResult
    logical_call_no: int
    physical_attempt_no: int
    injected: FailureTrigger | None = None

    @property
    def ok(self) -> bool:
        return self.typed.ok


class _FailureController:
    def __init__(self, script: FailureScript):
        self.script = script
        self.used: dict[str, int] = {}
        # Script ordinal is the deterministic tie-break required by the
        # failure_script contract.  ``trigger_id`` is an identifier and must
        # not alter which rule wins when priorities are equal.
        self._ordinals = {trigger.trigger_id: ordinal for ordinal, trigger in enumerate(script.triggers)}

    def match(self, tool_ref: str, logical: int, physical: int, layer: str | None) -> FailureTrigger | None:
        candidates = []
        for trigger in self.script.triggers:
            if trigger.tool_ref not in {tool_ref, "*"}:
                continue
            if trigger.logical_call is not None and trigger.logical_call != logical:
                continue
            if trigger.physical_attempt is not None and trigger.physical_attempt != physical:
                continue
            if trigger.layer is not None and trigger.layer != layer:
                continue
            used = self.used.get(trigger.trigger_id, 0)
            if used >= trigger.count and trigger.exhaustion == "ignore":
                continue
            if used >= trigger.count and trigger.exhaustion == "error":
                return trigger.model_copy(update={"action": "RETURN_ERROR", "error_code": "FAILURE_SCRIPT_EXHAUSTED", "trigger_id": f"{trigger.trigger_id}:exhausted"})
            candidates.append(trigger)
        if not candidates:
            return None
        chosen = min(candidates, key=lambda item: (-item.priority, self._ordinals[item.trigger_id]))
        if chosen.exhaustion != "repeat":
            self.used[chosen.trigger_id] = self.used.get(chosen.trigger_id, 0) + 1
        return chosen


class ToolSimulator:
    """Deterministic faults wrap, but never bypass, the public runtime port."""
    def __init__(self, *, registry: M3Registry | None = None, mode: ExecutionMode | str = ExecutionMode.SIMULATED,
                 world_snapshot: WorldSnapshot, failure_script: FailureScript | Mapping[str, Any] | None = None):
        self.registry = registry or build_m3_registry()
        self.mode = ExecutionMode(mode)
        if self.mode == ExecutionMode.REPLAY:
            raise ValueError("ToolSimulator cannot execute in replay mode")
        self.world_snapshot = world_snapshot
        self.failure_script = FailureScript.from_mapping(failure_script)
        self._controller = _FailureController(self.failure_script)

    def _port(self, tool_ref: str) -> AgentPort:
        port_type = _PORT_TYPES.get(tool_ref, AgentPort)
        port = port_type(registry=self.registry)
        # Dynamic/custom tools still use AgentPort.invoke; only the public
        # class attributes are filled to select the admitted registry entry.
        port.tool_ref = tool_ref
        spec = self.registry.get(tool_ref)
        port.agent_ref = spec.owner
        port.output_contract = spec.result_schema
        return port

    def invoke(self, tool_ref: str, args: Mapping[str, Any], *, context: InvocationContext,
               logical_call_no: int, physical_attempt_no: int = 1, layer: str | None = None) -> HarnessToolResult:
        if self.mode == ExecutionMode.REPLAY:
            raise RuntimeError("replay is read-only and cannot invoke tools")
        if context.plan_revision_id == "":
            raise ValueError("invalid invocation context")
        spec = self.registry.get(tool_ref)
        trigger = None
        if self.mode in {ExecutionMode.FAULT, ExecutionMode.SIMULATED}:
            trigger = self._controller.match(tool_ref, logical_call_no, physical_attempt_no, layer)
        if trigger is not None:
            if trigger.action == "BLOCK_TASK":
                typed = TypedAgentResult(contract=spec.result_schema, ok=False, payload=None,
                    payload_hash=sha256_json({"error_code": "BLOCKED"}), error_code="BLOCKED",
                    agent_ref=spec.owner, tool_ref=tool_ref)
            else:
                typed = TypedAgentResult(contract=spec.result_schema, ok=False, payload=None,
                    payload_hash=sha256_json({"error_code": trigger.error_code}), error_code=trigger.error_code,
                    agent_ref=spec.owner, tool_ref=tool_ref)
            return HarnessToolResult(typed, logical_call_no, physical_attempt_no, trigger)
        typed = self._port(tool_ref).invoke(args, context=context)
        return HarnessToolResult(typed, logical_call_no, physical_attempt_no)


__all__ = ["HarnessToolResult", "ToolSimulator"]
