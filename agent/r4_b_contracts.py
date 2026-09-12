"""R4-B supervisor/provider boundary contracts.

The supervisor is allowed to propose a small, typed handoff decision.  The
runtime remains the owner of run identity, message envelopes, deadlines,
idempotency and tool authorization.  A provider therefore receives a frozen
generic contract prompt containing the user request and returns only
:class:`R4SupervisorDecisionV1`; it cannot provide
an A2A envelope or trusted execution fields.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Mapping, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


R4_SUPERVISOR_SCHEMA_VERSION = "r4.supervisor.decision.v1"
R4_TOPOLOGIES = frozenset(
    {
        "order_only",
        "logistics_only",
        "policy_only",
        "order->logistics",
        "order+policy",
        "order->logistics+policy",
    }
)
R4_CAPABILITY_REFS = frozenset({"order/read@v1", "logistics/read@v1", "policy/read@v1"})
R4_ENTITY_KEYS = frozenset(
    {"order_id", "phone_last4", "carrier_code", "tracking_no", "policy_query"}
)

R4_CAPABILITY_TASKS = {
    "order/read@v1": "order",
    "logistics/read@v1": "logistics",
    "policy/read@v1": "policy",
}
R4_TOPOLOGY_CAPABILITIES = {
    "order_only": frozenset({"order/read@v1"}),
    "logistics_only": frozenset({"logistics/read@v1"}),
    "policy_only": frozenset({"policy/read@v1"}),
    "order->logistics": frozenset({"order/read@v1", "logistics/read@v1"}),
    "order+policy": frozenset({"order/read@v1", "policy/read@v1"}),
    "order->logistics+policy": frozenset({"order/read@v1", "logistics/read@v1", "policy/read@v1"}),
}
R4_TOPOLOGY_DEPENDENCIES = {
    "order_only": frozenset(),
    "logistics_only": frozenset(),
    "policy_only": frozenset(),
    "order->logistics": frozenset({("order", "logistics")}),
    "order+policy": frozenset(),
    "order->logistics+policy": frozenset({("order", "logistics")}),
}

def _topology_mapping_prompt_lines() -> str:
    """Render the closed topology mapping from the trusted contract constants."""

    lines: list[str] = []
    for topology, capabilities in R4_TOPOLOGY_CAPABILITIES.items():
        ordered_capabilities = [
            capability for capability in R4_CAPABILITY_TASKS if capability in capabilities
        ]
        capability_text = " + ".join(ordered_capabilities)
        dependencies = R4_TOPOLOGY_DEPENDENCIES[topology]
        dependency_text = "none" if not dependencies else ", ".join(
            f"{upstream} -> {downstream}" for upstream, downstream in sorted(dependencies)
        )
        lines.append(
            f"- {topology} -> capabilities: {capability_text}; dependencies: {dependency_text}."
        )
    return "\n".join(lines)


R4_SUPERVISOR_PROMPT_VERSION = "r4.supervisor.prompt.v3"
R4_SUPERVISOR_PROMPT_TEMPLATE = f"""R4 supervisor contract v3 (r4.supervisor.prompt.v3)
You are a read-only planning component. Return only one JSON object that conforms to the supplied schema.
Permitted topology values: order_only, logistics_only, policy_only, order->logistics, order+policy, order->logistics+policy.
Permitted task ids: order, logistics, policy.
Permitted capability refs: order/read@v1, logistics/read@v1, policy/read@v1.
Topology/capability/dependency mapping (copy exactly; no other combination is valid):
{_topology_mapping_prompt_lines()}
Dependency rules: use only listed task ids; an edge means the upstream task must finish before the downstream task; use order -> logistics only when the request explicitly requires that sequence.
Lookup rules: an order lookup normally requires an explicitly stated order_id and phone_last4; a direct logistics lookup requires carrier_code and tracking_no. If the user asks to continue from an identified order to its logistics information, an order -> logistics dependency is allowed and carrier_code/tracking_no may come from the order result rather than the user request.
Semantic rules: requests about return deadlines, refund arrival, or exchange conditions are Policy tasks. Copy a policy question verbatim into entities.policy_query; never use entities.query. Copy a tracking identifier as one whole value; do not infer a carrier from a prefix or hyphen. A direct logistics request without an explicitly stated carrier requires clarification.
Clarification rules: copy only entities and questions explicitly stated by the user; order and order-derived logistics lookups require both an explicitly stated order_id and phone_last4, and a missing one requires needs_clarification=true; if a required identity or policy question is missing, preserve the recognizable capability/topology and give a short reason; never invent values from hidden records.
Policy questions are independent of order and logistics. Do not create any dependency involving policy; the only allowed dependency edge is order -> logistics.
The proposal is read-only and contains no runtime, message, tool, result, terminal, or answer fields.
User request:
{{user_text}}
"""
R4_SUPERVISOR_PROMPT_CHECKSUM_SHA256 = hashlib.sha256(R4_SUPERVISOR_PROMPT_TEMPLATE.encode("utf-8")).hexdigest()


def build_supervisor_prompt(user_text: str) -> str:
    """Build the frozen generic prompt around one user request."""

    return R4_SUPERVISOR_PROMPT_TEMPLATE.format(user_text=str(user_text))


class R4SupervisorContractError(ValueError):
    """Provider-boundary failure with a stable code and no raw provider text."""

    def __init__(self, code: str, message: str = "") -> None:
        self.code = str(code)
        super().__init__(message or self.code)


class R4SupervisorDependencyV1(BaseModel):
    """One proposed task dependency; task identities are proposal-local."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    upstream_task_id: str = Field(min_length=1)
    downstream_task_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def no_self_edge(self) -> "R4SupervisorDependencyV1":
        if self.upstream_task_id == self.downstream_task_id:
            raise ValueError("dependency cannot point to itself")
        return self


class R4SupervisorDecisionV1(BaseModel):
    """The only structure a supervisor provider may return.

    The model deliberately has no run, plan, message, sender, receiver,
    deadline, idempotency, tool, result, terminal or answer fields.  Those
    values are injected and verified by the trusted runtime after this
    boundary.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, validate_assignment=True)

    schema_version: str = R4_SUPERVISOR_SCHEMA_VERSION
    requested_capabilities: tuple[str, ...] = ()
    topology: str = Field(min_length=1)
    dependencies: tuple[R4SupervisorDependencyV1, ...] = ()
    entities: dict[str, Any] = Field(default_factory=dict)
    needs_clarification: bool = False
    clarification_reason: str | None = None

    @field_validator("requested_capabilities")
    @classmethod
    def canonical_capabilities(cls, value: Sequence[str]) -> tuple[str, ...]:
        normalized = tuple(str(item) for item in value)
        if len(normalized) != len(set(normalized)):
            raise ValueError("requested_capabilities must be unique")
        if any(item not in R4_CAPABILITY_REFS for item in normalized):
            raise ValueError("requested_capabilities contains an unknown capability")
        return tuple(sorted(normalized))

    @field_validator("entities")
    @classmethod
    def minimal_entities(cls, value: Mapping[str, Any]) -> dict[str, Any]:
        unknown = set(map(str, value)) - R4_ENTITY_KEYS
        if unknown:
            raise ValueError(f"unsupported business entity fields: {sorted(unknown)}")
        return {str(key): item for key, item in value.items()}

    @field_validator("topology")
    @classmethod
    def known_topology(cls, value: str) -> str:
        normalized = str(value).strip().lower().replace(" ", "")
        if normalized not in R4_TOPOLOGIES:
            raise ValueError("unsupported R4 topology")
        return normalized

    @model_validator(mode="after")
    def validate_decision(self) -> "R4SupervisorDecisionV1":
        if self.schema_version != R4_SUPERVISOR_SCHEMA_VERSION:
            raise ValueError("supervisor schema version mismatch")
        edges = {(edge.upstream_task_id, edge.downstream_task_id) for edge in self.dependencies}
        if len(edges) != len(self.dependencies):
            raise ValueError("duplicate dependency edge")
        expected_capabilities = R4_TOPOLOGY_CAPABILITIES[self.topology]
        actual_capabilities = frozenset(self.requested_capabilities)
        if actual_capabilities != expected_capabilities:
            raise ValueError("topology/capability mismatch")
        actual_dependencies = frozenset(edges)
        expected_dependencies = R4_TOPOLOGY_DEPENDENCIES[self.topology]
        if actual_dependencies != expected_dependencies:
            raise ValueError("topology/dependency mismatch")
        allowed_tasks = {R4_CAPABILITY_TASKS[capability] for capability in expected_capabilities}
        if any(task not in allowed_tasks for edge in self.dependencies for task in (edge.upstream_task_id, edge.downstream_task_id)):
            raise ValueError("dependency references a task outside the topology")
        if self.needs_clarification and not (self.clarification_reason or "").strip():
            raise ValueError("clarification reason is required")
        if not self.needs_clarification and self.clarification_reason is not None:
            raise ValueError("clarification reason requires needs_clarification")
        return self


class R4SupervisorProvider(Protocol):
    """Provider callable used by the boundary; it cannot receive trusted context."""

    def __call__(self, prompt: str, schema: type[R4SupervisorDecisionV1]) -> Any: ...


class R4SupervisorAttemptEvidenceV1(BaseModel):
    """Redacted evidence for one provider attempt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider_called: bool = False
    provider_returned: bool = False
    schema_valid: bool = False
    latency_ms: float = Field(default=0.0, ge=0.0)
    usage: dict[str, int] = Field(default_factory=dict)
    attempt: int = Field(default=1, ge=1)
    model_ref: str | None = None
    prompt_version: str = R4_SUPERVISOR_PROMPT_VERSION
    prompt_checksum_sha256: str = R4_SUPERVISOR_PROMPT_CHECKSUM_SHA256
    error_code: str | None = None
    observed_at: datetime

    @field_validator("observed_at")
    @classmethod
    def utc_observed_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("observed_at must be timezone-aware")
        return value.astimezone(timezone.utc)


class R4SupervisorCallEvidenceV1(BaseModel):
    """Final and aggregate provider evidence without raw output or errors."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # These fields preserve the original public evidence shape.  They describe
    # the final attempt, except provider_called/provider_returned which retain
    # the historical any-attempt meaning used by the evaluator.
    provider_called: bool = False
    provider_returned: bool = False
    schema_valid: bool = False
    latency_ms: float = Field(default=0.0, ge=0.0)
    usage: dict[str, int] = Field(default_factory=dict)
    attempt: int = Field(default=1, ge=1, le=2)
    model_ref: str | None = None
    prompt_version: str = R4_SUPERVISOR_PROMPT_VERSION
    prompt_checksum_sha256: str = R4_SUPERVISOR_PROMPT_CHECKSUM_SHA256
    error_code: str | None = None
    observed_at: datetime

    # Correction-06 aggregate and per-attempt evidence.
    attempts: tuple[R4SupervisorAttemptEvidenceV1, ...] = ()
    total_attempts: int = Field(default=1, ge=1, le=2)
    first_attempt_schema_valid: bool = False
    retried: bool = False

    @field_validator("observed_at")
    @classmethod
    def utc_observed_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("observed_at must be timezone-aware")
        return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class R4SupervisorBoundaryResult:
    decision: R4SupervisorDecisionV1 | None
    evidence: R4SupervisorCallEvidenceV1


class R4SupervisorProviderBoundary:
    """Validate provider output before trusted runtime construction.

    This class intentionally accepts no run context and never constructs an
    A2A message.  A provider failure is represented by bounded, redacted
    evidence so evaluator artifacts do not contain raw prompts or exceptions.
    """

    def __init__(self, provider: R4SupervisorProvider, *, model_ref: str | None = None, clock: Any = None):
        self.provider = provider
        self.model_ref = model_ref
        self.clock = clock

    def _now(self) -> datetime:
        value = self.clock() if callable(self.clock) else self.clock
        if value is None:
            value = datetime.now(timezone.utc)
        if value.tzinfo is None or value.utcoffset() is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    @staticmethod
    def _usage_payload(raw: Any) -> dict[str, int]:
        actual_usage = getattr(raw, "usage", None)
        if actual_usage is None:
            return {}
        if isinstance(actual_usage, Mapping):
            source_usage = actual_usage
        else:
            source_usage = {
                name: getattr(actual_usage, name)
                for name in ("input_tokens", "output_tokens", "total_tokens")
                if getattr(actual_usage, name, None) is not None
            }
        return {
            str(key): int(value)
            for key, value in source_usage.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }

    def _aggregate_evidence(
        self,
        attempts: Sequence[R4SupervisorAttemptEvidenceV1],
        *,
        started: float,
    ) -> R4SupervisorCallEvidenceV1:
        if not attempts:
            raise RuntimeError("supervisor evidence requires an attempt")
        final = attempts[-1]
        aggregate_usage: dict[str, int] = {}
        for item in attempts:
            for key, value in item.usage.items():
                aggregate_usage[key] = aggregate_usage.get(key, 0) + int(value)
        return R4SupervisorCallEvidenceV1(
            provider_called=any(item.provider_called for item in attempts),
            provider_returned=any(item.provider_returned for item in attempts),
            schema_valid=final.schema_valid,
            latency_ms=max(0.0, (perf_counter() - started) * 1000.0),
            usage=aggregate_usage,
            attempt=final.attempt,
            model_ref=final.model_ref,
            prompt_version=R4_SUPERVISOR_PROMPT_VERSION,
            prompt_checksum_sha256=R4_SUPERVISOR_PROMPT_CHECKSUM_SHA256,
            error_code=final.error_code,
            observed_at=final.observed_at,
            attempts=tuple(attempts),
            total_attempts=len(attempts),
            first_attempt_schema_valid=attempts[0].schema_valid,
            retried=len(attempts) > 1,
        )

    def decide(self, user_text: str) -> R4SupervisorBoundaryResult:
        started = perf_counter()
        prompt = build_supervisor_prompt(user_text)
        attempts: list[R4SupervisorAttemptEvidenceV1] = []
        for attempt_no in (1, 2):
            attempt_started = perf_counter()
            observed_at = self._now()
            provider_called = False
            provider_returned = False
            actual_model: Any = None
            usage_payload: dict[str, int] = {}
            try:
                provider_called = True
                raw = self.provider(prompt, R4SupervisorDecisionV1)
                provider_returned = True
                actual_model = getattr(raw, "model", None)
                usage_payload = self._usage_payload(raw)
                if hasattr(raw, "output"):
                    raw = raw.output
                elif hasattr(raw, "value"):
                    raw = raw.value
                decision = R4SupervisorDecisionV1.model_validate(raw)
            except Exception:
                attempt = R4SupervisorAttemptEvidenceV1(
                    provider_called=provider_called,
                    provider_returned=provider_returned,
                    schema_valid=False,
                    latency_ms=max(0.0, (perf_counter() - attempt_started) * 1000.0),
                    usage=usage_payload,
                    attempt=attempt_no,
                    model_ref=str(actual_model) if actual_model is not None else self.model_ref,
                    prompt_version=R4_SUPERVISOR_PROMPT_VERSION,
                    prompt_checksum_sha256=R4_SUPERVISOR_PROMPT_CHECKSUM_SHA256,
                    error_code=(
                        "R4_SUPERVISOR_SCHEMA_INVALID"
                        if provider_returned
                        else "R4_SUPERVISOR_PROVIDER_ERROR"
                    ),
                    observed_at=observed_at,
                )
                attempts.append(attempt)
                # Only a returned but schema-invalid result is eligible for
                # one bounded retry.  Provider exceptions stop immediately.
                if not provider_returned or attempt_no == 2:
                    return R4SupervisorBoundaryResult(
                        decision=None,
                        evidence=self._aggregate_evidence(attempts, started=started),
                    )
                continue

            attempt = R4SupervisorAttemptEvidenceV1(
                provider_called=provider_called,
                provider_returned=provider_returned,
                schema_valid=True,
                latency_ms=max(0.0, (perf_counter() - attempt_started) * 1000.0),
                usage=usage_payload,
                attempt=attempt_no,
                model_ref=str(actual_model) if actual_model is not None else self.model_ref,
                prompt_version=R4_SUPERVISOR_PROMPT_VERSION,
                prompt_checksum_sha256=R4_SUPERVISOR_PROMPT_CHECKSUM_SHA256,
                observed_at=observed_at,
            )
            attempts.append(attempt)
            return R4SupervisorBoundaryResult(
                decision=decision,
                evidence=self._aggregate_evidence(attempts, started=started),
            )

        raise RuntimeError("supervisor retry loop exhausted")


class StaticR4SupervisorProvider:
    """Deterministic provider for contract/evaluator shell tests only."""

    test_only = True

    def __init__(self, decision: R4SupervisorDecisionV1 | Mapping[str, Any]):
        self.decision = R4SupervisorDecisionV1.model_validate(decision)
        self.calls = 0

    def __call__(self, _user_text: str, schema: type[R4SupervisorDecisionV1]) -> R4SupervisorDecisionV1:
        if schema is not R4SupervisorDecisionV1:
            raise R4SupervisorContractError("R4_SUPERVISOR_SCHEMA_UNEXPECTED")
        self.calls += 1
        return self.decision


__all__ = [
    "R4_CAPABILITY_REFS",
    "R4_ENTITY_KEYS",
    "R4_SUPERVISOR_SCHEMA_VERSION",
    "R4_SUPERVISOR_PROMPT_VERSION",
    "R4_SUPERVISOR_PROMPT_TEMPLATE",
    "R4_SUPERVISOR_PROMPT_CHECKSUM_SHA256",
    "R4_TOPOLOGIES",
    "R4_CAPABILITY_TASKS",
    "R4_TOPOLOGY_CAPABILITIES",
    "R4_TOPOLOGY_DEPENDENCIES",
    "R4SupervisorAttemptEvidenceV1",
    "R4SupervisorBoundaryResult",
    "R4SupervisorCallEvidenceV1",
    "R4SupervisorContractError",
    "R4SupervisorDependencyV1",
    "R4SupervisorDecisionV1",
    "R4SupervisorProvider",
    "R4SupervisorProviderBoundary",
    "build_supervisor_prompt",
    "StaticR4SupervisorProvider",
]
