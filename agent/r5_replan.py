"""R5 bounded replan policy.

Replanning is only allowed onto a *declared*, argument-contract-compatible
alternative capability, only for retryable failures, only for read nodes and
only within the pre-registered budget (default 1).  When no legitimate
alternative exists the correct outcome is to stop/clarify/escalate — never to
invent an always-succeeding fallback.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .r5_capability_args import CAPABILITY_ARGS
from .r5_plan_contracts import R5_WRITE_CAPABILITIES

RETRYABLE_FAILURE_CODES = frozenset({"INFRA_TIMEOUT", "INFRA_UNAVAILABLE", "INFRA_RATE_LIMITED", "TOOL_EXECUTION_FAILED"})


@dataclass(frozen=True)
class ReplanDecision:
    replan: bool
    alternative_capability: str | None
    reason: str


def _arg_compatible(left: str, right: str) -> bool:
    return set(CAPABILITY_ARGS.get(left, {})) == set(CAPABILITY_ARGS.get(right, {}))


class BoundedReplanPolicy:
    """Declared alternative capabilities with a hard budget."""

    def __init__(self, alternatives: Mapping[str, tuple[str, ...]] | None = None, *, budget: int = 1):
        if budget < 0:
            raise ValueError("replan budget must be >= 0")
        self.alternatives = {str(k): tuple(str(v) for v in vs) for k, vs in (alternatives or {}).items()}
        self.budget = int(budget)
        self.used = 0

    def decide(self, *, capability_ref: str, error_code: str | None, side_effect: str = "READ_ONLY") -> ReplanDecision:
        if capability_ref in R5_WRITE_CAPABILITIES:
            return ReplanDecision(False, None, "write_capability_not_replanned")
        if error_code not in RETRYABLE_FAILURE_CODES:
            return ReplanDecision(False, None, "not_retryable")
        if self.used >= self.budget:
            return ReplanDecision(False, None, "budget_exhausted")
        for alternative in self.alternatives.get(capability_ref, ()):
            if _arg_compatible(capability_ref, alternative):
                self.used += 1
                return ReplanDecision(True, alternative, "declared_arg_compatible_alternative")
        return ReplanDecision(False, None, "no_legitimate_alternative")


__all__ = ["BoundedReplanPolicy", "RETRYABLE_FAILURE_CODES", "ReplanDecision"]
