"""Read-only M4 evaluators and reproducible statistics.

Evaluators consume a frozen run projection and a separate evaluator input.  No
gold data is imported by the runtime harness.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Iterable, Mapping, Sequence

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 10000
WILSON_Z = 1.959963984540054


class CaseStatus(str, Enum):
    PASS = "PASS"
    MISSING = "MISSING"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    CANCELLED = "CANCELLED"
    NA = "N/A"


def wilson_interval(successes: int | float, trials: int | float, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    n = float(trials)
    if n <= 0: return (0.0, 0.0)
    p = min(max(float(successes) / n, 0.0), 1.0)
    z = WILSON_Z if confidence == 0.95 else 1.959963984540054
    d = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / d
    half = z * math.sqrt((p * (1 - p) / n) + z * z / (4.0 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def paired_bootstrap(left: Sequence[float], right: Sequence[float], *, seed: int = BOOTSTRAP_SEED,
                     resamples: int = BOOTSTRAP_RESAMPLES, confidence: float = 0.95) -> dict[str, Any]:
    """Fixed-seed case-level paired bootstrap for mean(right-left)."""
    if len(left) != len(right): raise ValueError("paired bootstrap requires equal-length samples")
    if not left: return {"delta": 0.0, "ci95": [0.0, 0.0], "resamples": resamples, "seed": seed, "n": 0}
    import random
    differences = [float(r) - float(l) for l, r in zip(left, right)]
    observed = sum(differences) / len(differences)
    rng = random.Random(seed)
    values = []
    for _ in range(int(resamples)):
        values.append(sum(differences[rng.randrange(len(differences))] for _ in differences) / len(differences))
    values.sort()
    alpha = (1.0 - confidence) / 2.0
    lo = values[min(len(values) - 1, max(0, int(math.floor(alpha * len(values)))))]
    hi = values[min(len(values) - 1, max(0, int(math.floor((1.0 - alpha) * len(values))) - 1))]
    return {"delta": observed, "ci95": [lo, hi], "resamples": int(resamples), "seed": int(seed), "n": len(differences)}


def _status(value: Any) -> CaseStatus:
    if isinstance(value, CaseStatus): return value
    text = str(value or "PASS").upper().replace("_", " ")
    return {"N/A": CaseStatus.NA, "NA": CaseStatus.NA, "MISSING": CaseStatus.MISSING,
            "FAILED": CaseStatus.FAILED, "BLOCKED": CaseStatus.BLOCKED,
            "CANCELLED": CaseStatus.CANCELLED}.get(text, CaseStatus.PASS)


def _same(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return False
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        return {k: v for k, v in a.items() if k not in {"run_id", "task_id", "attempt_id", "result_id"}} == {k: v for k, v in b.items() if k not in {"run_id", "task_id", "attempt_id", "result_id"}}
    return a == b


@dataclass(frozen=True)
class MetricResult:
    name: str
    numerator: int
    denominator: int
    n: int
    n_applicable: int
    n_missing: int
    n_failed: int
    n_blocked: int
    n_cancelled: int
    n_na: int
    value: float
    ci95: tuple[float, float]
    coverage: float

    def as_dict(self) -> dict[str, Any]:
        return {"metric": self.name, "value": self.value, "numerator": self.numerator,
                "denominator": self.denominator, "N": self.n, "N_applicable": self.n_applicable,
                "N_missing": self.n_missing, "N_failed": self.n_failed, "N_blocked": self.n_blocked,
                "N_cancelled": self.n_cancelled, "N/A": self.n_na, "coverage": self.coverage,
                "wilson95": list(self.ci95)}


def _metric(name: str, values: Sequence[bool | float | int], statuses: Sequence[CaseStatus], applicable: Sequence[bool]) -> MetricResult:
    considered = [bool(v) for v, ok in zip(values, applicable) if ok]
    n = len(values); numerator = sum(considered); denominator = len(considered)
    counts = {s: sum(1 for x in statuses if x == s) for s in CaseStatus}
    value = numerator / denominator if denominator else 0.0
    return MetricResult(name, numerator, denominator, n, denominator, counts[CaseStatus.MISSING],
                        counts[CaseStatus.FAILED], counts[CaseStatus.BLOCKED], counts[CaseStatus.CANCELLED],
                        counts[CaseStatus.NA], value, wilson_interval(numerator, denominator), denominator / n if n else 0.0)


@dataclass(frozen=True)
class EvaluatorSpec:
    name: str
    version: str
    fn: Callable[[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]], bool | float]


class EvaluatorRegistry:
    def __init__(self, *, version: str = "m4.evaluator-registry.v1"):
        self.version = version
        self._items: dict[str, EvaluatorSpec] = {}

    def register(self, name: str, fn: Callable[[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]], bool | float], *, version: str | None = None) -> None:
        if name in self._items: raise ValueError(f"evaluator already registered: {name}")
        self._items[name] = EvaluatorSpec(name, version or f"{self.version}:{name}:v1", fn)

    def get(self, name: str) -> EvaluatorSpec:
        try: return self._items[name]
        except KeyError as exc: raise KeyError(f"unknown evaluator: {name}") from exc

    def names(self) -> tuple[str, ...]: return tuple(sorted(self._items))

    def versions(self) -> dict[str, str]: return {name: item.version for name, item in sorted(self._items.items())}

    def evaluate(self, cases: Sequence[Mapping[str, Any]], outputs: Sequence[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]],
                 gold: Sequence[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]] | None = None,
                 *, evaluator_names: Iterable[str] | None = None) -> dict[str, Any]:
        return evaluate_cases(cases, outputs, gold, registry=self, evaluator_names=evaluator_names)


def default_registry() -> EvaluatorRegistry:
    reg = EvaluatorRegistry()
    reg.register("task_completion", lambda c, r, g: bool(r.get("task_completion", r.get("completed", False))))
    reg.register("intent_accuracy", lambda c, r, g: _same(r.get("intent"), g.get("intent")))
    reg.register("plan_accuracy", lambda c, r, g: _same(r.get("plan"), g.get("plan")))
    reg.register("business_code", lambda c, r, g: _same(r.get("business_code", r.get("last_business_code")), g.get("business_code", g.get("expected_business_code"))))
    reg.register("tool_path", lambda c, r, g: _same(r.get("tool_path", r.get("tool_names", [])), g.get("tool_path", g.get("expected_tool_path", []))))
    reg.register("case_pass", lambda c, r, g: bool(r.get("case_pass", False)))
    reg.register("invariant_rate", lambda c, r, g: float(r.get("invariant_passed", r.get("invariants_passed", 0))) / max(1, float(r.get("invariant_total", r.get("invariants_total", 1)))))
    reg.register("world_fingerprint", lambda c, r, g: _same(r.get("world_fingerprint"), g.get("world_fingerprint")))
    reg.register("safety", lambda c, r, g: bool(r.get("safety_pass", r.get("safe", False))))
    reg.register("rag_recall_at_5", lambda c, r, g: bool(set(r.get("retrieved_chunk_ids", [])) & set(g.get("gold_chunk_ids", g.get("gold_source_ids", [])))))
    reg.register("claim_evidence", lambda c, r, g: bool(r.get("claim_evidence_correct", r.get("evidence_correct", False))))
    return reg


def evaluate_cases(cases: Sequence[Mapping[str, Any]], outputs: Sequence[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]],
                   gold: Sequence[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]] | None = None,
                   *, registry: EvaluatorRegistry | None = None, evaluator_names: Iterable[str] | None = None) -> dict[str, Any]:
    """Evaluate frozen outcomes; gold is accepted only as a separate argument."""
    reg = registry or default_registry()
    names = tuple(evaluator_names or reg.names())
    def lookup(values: Any, index: int, sid: str) -> Mapping[str, Any]:
        if values is None: return {}
        if isinstance(values, Mapping): return values.get(sid, {})
        return values[index] if index < len(values) else {}
    result: dict[str, Any] = {"evaluator_version": reg.version, "metrics": {}, "N": len(cases)}
    statuses: list[CaseStatus] = []
    for i, case in enumerate(cases):
        sid = str(case.get("scenario_id", case.get("case_id", i)))
        run = lookup(outputs, i, sid)
        # An absent frozen outcome is a counted MISSING case, never an implicit
        # success or N/A.
        if not run:
            statuses.append(CaseStatus.MISSING)
        else:
            statuses.append(_status(run.get("status", "PASS")))
    for name in names:
        spec = reg.get(name); values: list[bool | float] = []; applicable: list[bool] = []
        for i, case in enumerate(cases):
            sid = str(case.get("scenario_id", case.get("case_id", i))); run = lookup(outputs, i, sid); expected = lookup(gold, i, sid)
            explicit_na = bool(expected.get("not_applicable", False) or name in (expected.get("not_applicable_metrics") or []))
            applicable.append(not explicit_na and statuses[i] != CaseStatus.NA)
            if statuses[i] in {CaseStatus.MISSING, CaseStatus.FAILED, CaseStatus.BLOCKED, CaseStatus.CANCELLED}:
                values.append(False)
            else:
                try: values.append(spec.fn(case, run, expected))
                except (KeyError, TypeError, ValueError, ZeroDivisionError): values.append(False)
        metric = _metric(name, values, statuses, applicable)
        result["metrics"][name] = metric.as_dict()
    result["status_counts"] = {s.value: sum(1 for x in statuses if x == s) for s in CaseStatus}
    return result


__all__ = ["BOOTSTRAP_RESAMPLES", "BOOTSTRAP_SEED", "CaseStatus", "EvaluatorRegistry", "MetricResult", "default_registry", "evaluate_cases", "paired_bootstrap", "wilson_interval"]
