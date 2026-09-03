"""Version-safe paired comparison for M4 evaluation runs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .evaluator import BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, paired_bootstrap, wilson_interval

VERSION_FIELDS = ("schema", "model", "prompt", "code", "registry", "tool_impl", "config", "policy_catalog", "kb", "dataset", "harness", "trace_schema", "evaluator", "simulator", "world_template", "seed")


def _version(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"): return dict(value.model_dump(mode="json", by_alias=True))
    if isinstance(value, Mapping):
        result = dict(value)
        if "schema_version" in result and "schema" not in result: result["schema"] = result.pop("schema_version")
        return result
    return {}


@dataclass(frozen=True)
class ComparableRun:
    run_id: str
    version_tuple: Mapping[str, Any]
    dataset_version: str
    scenario_ids: tuple[str, ...]
    gold_version: str
    evaluator_version: str
    metrics: Mapping[str, Sequence[float]]
    statuses: Mapping[str, str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.statuses is None: object.__setattr__(self, "statuses", {})


@dataclass(frozen=True)
class ComparisonResult:
    comparable: bool
    reason: str
    paired_coverage: float
    metrics: Mapping[str, Any]
    version_diff: Mapping[str, tuple[Any, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {"comparable": self.comparable, "reason": self.reason, "paired_coverage": self.paired_coverage,
                "metrics": dict(self.metrics), "version_diff": {k: list(v) for k, v in self.version_diff.items()}}


def _coerce_run(value: Any, side: str) -> ComparableRun:
    if isinstance(value, ComparableRun): return value
    if not isinstance(value, Mapping): raise TypeError("comparison input must be ComparableRun or mapping")
    outputs = value.get("outputs", value.get("cases", {}))
    if isinstance(outputs, Mapping):
        ids = tuple(sorted(str(x) for x in outputs))
        rows = [outputs[x] for x in ids]
    else:
        rows = list(outputs or []); ids = tuple(str(x.get("scenario_id", i)) for i, x in enumerate(rows))
    metrics: dict[str, list[float]] = {}
    for row in rows:
        for key, item in (row.get("metrics", row) if isinstance(row, Mapping) else {}).items():
            if isinstance(item, (int, float, bool)): metrics.setdefault(key, []).append(float(item))
    return ComparableRun(str(value.get("run_id", side)), _version(value.get("version_tuple", value.get("versions", {}))),
        str(value.get("dataset_version", value.get("dataset", ""))), ids, str(value.get("gold_version", "")),
        str(value.get("evaluator_version", "")), metrics, {str(k): str(v) for k, v in (value.get("statuses") or {}).items()})


class BaselineComparator:
    def __init__(self, *, seed: int = BOOTSTRAP_SEED, resamples: int = BOOTSTRAP_RESAMPLES):
        self.seed, self.resamples = seed, resamples

    def compare(self, left: ComparableRun | Mapping[str, Any], right: ComparableRun | Mapping[str, Any], *, allowed_variations: Iterable[str] = ()) -> ComparisonResult:
        a, b = _coerce_run(left, "left"), _coerce_run(right, "right")
        version_diff = {field: (a.version_tuple.get(field), b.version_tuple.get(field)) for field in VERSION_FIELDS if a.version_tuple.get(field) != b.version_tuple.get(field)}
        allowed = set(allowed_variations)
        required_missing = [name for name in VERSION_FIELDS if not a.version_tuple.get(name) or not b.version_tuple.get(name)]
        if required_missing:
            return ComparisonResult(False, "incomplete version tuple: " + ",".join(required_missing), 0.0, {}, version_diff)
        if a.dataset_version != b.dataset_version or a.dataset_version != a.version_tuple.get("dataset") or b.dataset_version != b.version_tuple.get("dataset"):
            return ComparisonResult(False, "dataset versions differ", 0.0, {}, version_diff)
        if a.gold_version != b.gold_version:
            return ComparisonResult(False, "gold versions differ", 0.0, {}, version_diff)
        if a.evaluator_version != b.evaluator_version or a.evaluator_version not in {a.version_tuple.get("evaluator"), b.version_tuple.get("evaluator")}:
            return ComparisonResult(False, "evaluator versions differ", 0.0, {}, version_diff)
        if set(a.scenario_ids) != set(b.scenario_ids):
            return ComparisonResult(False, "scenario manifests differ", 0.0, {}, version_diff)
        unexpected = set(version_diff) - allowed
        if unexpected:
            return ComparisonResult(False, "version tuple differs: " + ",".join(sorted(unexpected)), 0.0, {}, version_diff)
        n = len(a.scenario_ids)
        if not n: return ComparisonResult(False, "no paired cases", 0.0, {}, version_diff)
        metrics: dict[str, Any] = {}
        for name in sorted(set(a.metrics) & set(b.metrics)):
            av, bv = list(a.metrics[name]), list(b.metrics[name])
            if len(av) != n or len(bv) != n or len(av) != len(bv):
                continue
            metrics[name] = paired_bootstrap(av, bv, seed=self.seed, resamples=self.resamples)
            metrics[name]["left"] = sum(av) / len(av); metrics[name]["right"] = sum(bv) / len(bv)
            metrics[name]["left_wilson95"] = list(wilson_interval(sum(av), len(av)))
            metrics[name]["right_wilson95"] = list(wilson_interval(sum(bv), len(bv)))
        return ComparisonResult(True, "comparable", 1.0, metrics, version_diff)


def compare_runs(left: ComparableRun | Mapping[str, Any], right: ComparableRun | Mapping[str, Any], **kwargs: Any) -> ComparisonResult:
    return BaselineComparator(**{k: kwargs.pop(k) for k in ("seed", "resamples") if k in kwargs}).compare(left, right, **kwargs)


__all__ = ["BaselineComparator", "ComparableRun", "ComparisonResult", "VERSION_FIELDS", "compare_runs"]
