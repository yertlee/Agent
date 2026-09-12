"""R5 plan contract: one authoritative structure, no triple duplication.

R4 required the model to fill ``topology``, ``requested_capabilities`` and
``dependencies`` as three mutually-consistent copies of the same relationship
and only admitted six fixed topologies.  R5 replaces that with a single
authoritative plan: nodes (capability + typed args/bindings) and edges
(dependencies).  Capability sets and topology names are *derived* from that
structure, never re-entered by the model.  The schema admits any registered
capability combination, so it cannot encode the answer for a particular case.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


R5_PLAN_SCHEMA_VERSION = "r5.plan.v1"

# Capabilities a plan may reference.  This limits the *vocabulary*, not the
# answer: any acyclic combination that satisfies the binding rules is legal.
R5_READ_CAPABILITIES = frozenset(
    {"product/read@v1", "order/read@v1", "logistics/read@v1", "policy/read@v1", "aftersales/read@v1", "aftersales/eligibility@v1"}
)
R5_WRITE_CAPABILITIES = frozenset({"aftersales/write@v1"})
# Escalation to a human reviewer is an action, but it is not a business write:
# it needs no eligibility predecessor and no user confirmation (module guide 01
# §6, R5 guide §3: complaint/handoff is a human escalation path, not a sixth
# domain agent).
R5_ESCALATION_CAPABILITIES = frozenset({"human/handoff@v1"})
R5_ALL_CAPABILITIES = R5_READ_CAPABILITIES | R5_WRITE_CAPABILITIES | R5_ESCALATION_CAPABILITIES

# The write capability may only be planned after eligibility was established.
R5_WRITE_PRECONDITIONS = {"aftersales/write@v1": "aftersales/eligibility@v1"}

R5_FAILURE_STRATEGIES = frozenset({"RETRY", "REPLAN_LOCAL", "WAIT_USER", "WAIT_HUMAN", "BLOCK", "CONTINUE_PARTIAL", "FAIL_RUN"})

R5_CONTEXT_KEYS = frozenset(
    {"order_id", "phone_last4", "carrier_code", "tracking_no", "sku", "case_id", "service", "reason", "amount", "user_id", "policy_query"}
)


class R5BindingV1(BaseModel):
    """A typed input binding for a node.

    ``result`` bindings must name an upstream node in the dependency closure;
    ``context`` bindings read a trusted runtime context key.  The model may
    never bind a value it invented: a missing trusted binding is a
    clarification/block, not a fallback to a model-supplied field.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["result", "context"]
    source_node_id: str | None = None
    path: str | None = None
    context_key: str | None = None
    required: bool = True

    @model_validator(mode="after")
    def valid_binding(self) -> "R5BindingV1":
        if self.kind == "result":
            if not self.source_node_id or not self.path:
                raise ValueError("result binding requires source_node_id and path")
        else:
            if self.context_key not in R5_CONTEXT_KEYS:
                raise ValueError("context binding requires a registered trusted context key")
        return self


class R5PlanNodeV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    node_id: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    capability_ref: str = Field(min_length=1)
    args: dict[str, Any] = Field(default_factory=dict)
    bindings: dict[str, R5BindingV1] = Field(default_factory=dict)
    failure_strategy: str = "FAIL_RUN"

    @model_validator(mode="after")
    def valid_node(self) -> "R5PlanNodeV1":
        if self.capability_ref not in R5_ALL_CAPABILITIES:
            raise ValueError("capability is not registered")
        if self.failure_strategy not in R5_FAILURE_STRATEGIES:
            raise ValueError("unknown failure strategy")
        # Literal args may not shadow a binding of the same name.
        overlap = set(self.args) & set(self.bindings)
        if overlap:
            raise ValueError(f"arg and binding names overlap: {sorted(overlap)}")
        return self


class R5PlanEdgeV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    upstream_node_id: str = Field(min_length=1)
    downstream_node_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def no_self_edge(self) -> "R5PlanEdgeV1":
        if self.upstream_node_id == self.downstream_node_id:
            raise ValueError("edge cannot point to itself")
        return self


class R5PlanV1(BaseModel):
    """The single authoritative plan.  Derived views are computed, not stored."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = R5_PLAN_SCHEMA_VERSION
    nodes: tuple[R5PlanNodeV1, ...] = ()
    edges: tuple[R5PlanEdgeV1, ...] = ()
    needs_clarification: bool = False
    clarification_reason: str | None = None
    business_goal: str = Field(min_length=1)

    @model_validator(mode="after")
    def valid_plan(self) -> "R5PlanV1":
        if self.schema_version != R5_PLAN_SCHEMA_VERSION:
            raise ValueError("plan schema version mismatch")
        ids = [node.node_id for node in self.nodes]
        if len(ids) != len(set(ids)):
            raise ValueError("node ids must be unique")
        id_set = set(ids)
        for edge in self.edges:
            if edge.upstream_node_id not in id_set or edge.downstream_node_id not in id_set:
                raise ValueError("edge references an unknown node")
        if len({(e.upstream_node_id, e.downstream_node_id) for e in self.edges}) != len(self.edges):
            raise ValueError("duplicate edge")
        closure = self.ancestors()
        if self.nodes and not self.is_acyclic():
            raise ValueError("plan must be acyclic")
        for node in self.nodes:
            for binding in node.bindings.values():
                if binding.kind == "result" and binding.source_node_id not in closure.get(node.node_id, set()):
                    raise ValueError(f"result binding source is not an upstream dependency: {node.node_id}")
            pre = R5_WRITE_PRECONDITIONS.get(node.capability_ref)
            if pre is not None:
                ancestors = closure.get(node.node_id, set())
                if not any(self._capability(a) == pre for a in ancestors):
                    raise ValueError("write capability requires an eligibility ancestor")
        if self.needs_clarification and not (self.clarification_reason or "").strip():
            raise ValueError("clarification reason is required")
        if self.needs_clarification and any(n.capability_ref in (R5_WRITE_CAPABILITIES | R5_ESCALATION_CAPABILITIES) for n in self.nodes):
            raise ValueError("a clarification plan cannot contain write or escalation nodes")
        return self

    def _capability(self, node_id: str) -> str | None:
        for node in self.nodes:
            if node.node_id == node_id:
                return node.capability_ref
        return None

    def ancestors(self) -> dict[str, set[str]]:
        """Compute the transitive upstream closure for every node (Kahn)."""
        incoming: dict[str, set[str]] = {n.node_id: set() for n in self.nodes}
        edges: dict[str, set[str]] = {n.node_id: set() for n in self.nodes}
        for edge in self.edges:
            incoming[edge.downstream_node_id].add(edge.upstream_node_id)
            edges[edge.upstream_node_id].add(edge.downstream_node_id)
        closure: dict[str, set[str]] = {n: set() for n in incoming}
        # Relax until stable; acyclic input converges in <= N rounds.
        for _ in range(len(incoming) + 1):
            changed = False
            for node, parents in incoming.items():
                expanded = set(parents)
                for parent in parents:
                    expanded |= closure[parent]
                    expanded |= {parent}
                if expanded != closure[node]:
                    closure[node] = expanded
                    changed = True
            if not changed:
                break
        return closure

    def is_acyclic(self) -> bool:
        order = self.derived_dependency_order()
        return len(order) == len(self.nodes)

    def derived_capabilities(self) -> frozenset[str]:
        return frozenset(node.capability_ref for node in self.nodes)

    def derived_dependency_order(self) -> tuple[str, ...]:
        """Topological order; empty when the graph is cyclic."""
        incoming = {n.node_id: 0 for n in self.nodes}
        outgoing: dict[str, list[str]] = {n.node_id: [] for n in self.nodes}
        for edge in self.edges:
            incoming[edge.downstream_node_id] += 1
            outgoing[edge.upstream_node_id].append(edge.downstream_node_id)
        ready = sorted(node for node, count in incoming.items() if count == 0)
        order: list[str] = []
        while ready:
            node = ready.pop(0)
            order.append(node)
            for child in sorted(outgoing[node]):
                incoming[child] -= 1
                if incoming[child] == 0:
                    ready.append(child)
                    ready.sort()
        return tuple(order)

    def dependency_edges(self) -> tuple[tuple[str, str], ...]:
        return tuple((e.upstream_node_id, e.downstream_node_id) for e in self.edges)


def plan_has_capability(plan: R5PlanV1, capability_ref: str) -> bool:
    return capability_ref in plan.derived_capabilities()


__all__ = [
    "R5BindingV1",
    "R5PlanEdgeV1",
    "R5PlanNodeV1",
    "R5PlanV1",
    "R5_ALL_CAPABILITIES",
    "R5_CONTEXT_KEYS",
    "R5_ESCALATION_CAPABILITIES",
    "R5_FAILURE_STRATEGIES",
    "R5_PLAN_SCHEMA_VERSION",
    "R5_READ_CAPABILITIES",
    "R5_WRITE_CAPABILITIES",
    "R5_WRITE_PRECONDITIONS",
    "plan_has_capability",
]
