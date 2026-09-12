"""R5 Router/Planner evaluator with real execution and separate metrics.

Metric definitions (R5 review feedback):
1. ``plan_required_goal_coverage_rate``: required goals are present in the plan.
2. ``plan_conformance_rate``: required + allowed-optional present, forbidden and
   over-reach absent, clarification decision correct, schema valid.
3. ``task_completion_rate``: the model's actual plan, executed via the real R5
   A2A runtime, reaches the expected terminal and business state.  Guarded
   writes stop at pending confirmation; the evaluator never auto-approves.

The legacy combined number is retained under the explicit name
``plan_capability_coverage_and_clarification_match_rate`` and is flagged as
NOT representing task completion.  Real execution completion is reported
separately and is measured, not inferred from plan text.

The evaluator never uses gold to rebuild a plan.  An invalid plan is refused
by the executor and cannot earn credit for matching a clarification flag.

Run:  python -m eval.r5_router_planner_eval --split dev --candidate real
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

from agent.r5_capability_args import normalize_plan
from agent.r5_plan_contracts import R5PlanV1, R5_READ_CAPABILITIES, R5_WRITE_CAPABILITIES
from agent.r5_router_planner import (
    DeterministicRouterPlannerAdapter,
    R5RouterPlannerBoundary,
)
from eval.r5_plan_executor import R5PlanExecutor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "eval" / "datasets" / "r5"
DEMO = {
    "orders_db": PROJECT_ROOT / "ecommerce.db",
    "product_db": PROJECT_ROOT / "data" / "r5_demo_v1.db",
    "aftersales_db": PROJECT_ROOT / "data" / "r5_aftersales_demo_v1.db",
    "logistics_db": PROJECT_ROOT / "data" / "r5_logistics_demo_v1.db",
}


def _demo_paths(data_dir: str | Path | None = None) -> dict[str, Path]:
    """Resolve the four evaluator databases, optionally from a seeded fixture."""
    if data_dir is None:
        return dict(DEMO)
    root = Path(data_dir).expanduser().resolve()
    return {
        "orders_db": root / "ecommerce.db",
        "product_db": root / "r5_demo_v1.db",
        "aftersales_db": root / "r5_aftersales_demo_v1.db",
        "logistics_db": root / "r5_logistics_demo_v1.db",
    }


FIXTURE_SOURCE_VERSIONS = {
    "orders_db": "ecommerce.sqlite.read.v1",
    "product_db": "self_built_product_catalog.v1",
    "aftersales_db": "self_built_aftersales_demo.v1",
    "logistics_db": "versioned_order_derived_logistics_snapshot.v1",
}
# Session identity is a runtime fact, not gold: the demo user who owns the
# orders used by each split.
SESSION = {"dev": {"phone_last4": "1234", "user_id": "demo_user_1234"}, "validation": {"phone_last4": "9156", "user_id": "demo_user_9156"}}

# Terminal states the current plan vocabulary can actually reach.
EXECUTABLE_TERMINALS = {"CLARIFY", "ANSWER", "PENDING_CONFIRMATION", "REJECT", "HUMAN"}
BUSINESS_REJECTION_CODES = {"AUTH_RESOURCE_FORBIDDEN", "AUTH_IDENTITY_MISMATCH", "STATE_TRANSITION_REJECTED", "ELIGIBILITY_DENIED", "DATA_MISSING", "ORDER_NOT_FOUND"}

# Keep the intent metric label space independent of the labels emitted by a
# candidate.  ``MULTI_INTENT`` remains part of the frozen twelve-class macro
# for continuity, while the eleven-class core macro is reported separately as
# the primary semantic view.
# Keep this evaluator contract explicit and independent of the labels emitted
# by any provider or of a mutable runtime registry.
R5_INTENT_LABELS = (
    "AFTERSALES_CANCEL",
    "AFTERSALES_CREATE",
    "AFTERSALES_MODIFY",
    "AFTERSALES_STATUS",
    "CHITCHAT",
    "COMPLAINT",
    "LOGISTICS_QUERY",
    "MULTI_INTENT",
    "ORDER_QUERY",
    "POLICY_QA",
    "PRODUCT_QUERY",
    "UNKNOWN",
)
R5_CORE_INTENT_LABELS = tuple(label for label in R5_INTENT_LABELS if label != "MULTI_INTENT")
MULTI_INTENT_LABEL = "MULTI_INTENT"

# Per-case evidence is intended for audit and failure classification.  Keep the
# structure and stable digests of identifiers while avoiding order, tracking,
# phone, SKU, free-text query, and runtime payload values in saved reports.
_SENSITIVE_KEY_PARTS = (
    "order_id",
    "tracking_no",
    "phone",
    "sku",
    "ticket",
    "user_id",
    "query",
    "summary",
    "reason",
    "business_goal",
    "clarification_question",
)
_ORDER_RE = re.compile(r"(?<!\d)20\d{9,11}(?!\d)")
_SKU_RE = re.compile(r"\bSKU-[A-Za-z0-9]{2,}\b", re.IGNORECASE)
_LONG_ID_RE = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]{0,4}\d{10,20}(?![A-Za-z0-9])")


def _digest(value: Any) -> str:
    raw = str(value).encode("utf-8", errors="replace")
    return f"<redacted:{hashlib.sha256(raw).hexdigest()[:12]}>"


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _redact(value: Any, *, key: str | None = None) -> Any:
    """Recursively redact user/runtime values while retaining evidence shape."""
    if isinstance(value, Mapping):
        return {str(k): _redact(v, key=str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact(v, key=key) for v in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    text = str(value)
    lowered = (key or "").lower()
    if any(part in lowered for part in _SENSITIVE_KEY_PARTS):
        return _digest(text)
    text = _ORDER_RE.sub("<redacted:order>", text)
    text = _SKU_RE.sub("<redacted:sku>", text)
    text = _LONG_ID_RE.sub("<redacted:id>", text)
    return text


def _model_dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value


def _safe_execution_outcome(outcome: Any) -> dict[str, Any]:
    """Keep execution evidence without persisting domain payload values."""
    raw = outcome.to_dict()
    nodes: list[dict[str, Any]] = []
    for node in raw.get("node_results", []):
        safe = {k: v for k, v in node.items() if k != "payload"}
        payload = node.get("payload")
        safe["payload_present"] = payload is not None
        safe["payload_keys"] = sorted(str(k) for k in payload.keys()) if isinstance(payload, Mapping) else []
        nodes.append(_redact(safe))
    raw["node_results"] = nodes
    return _redact(raw)


def load_split(split: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (DATASET_DIR / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


def _oracle_plan(capabilities: list[str], entities: Mapping[str, Any], text: str, *, clarify: bool) -> dict[str, Any]:
    """Contract-valid plan used only to validate that metrics can reach their upper bound."""
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    present: set[str] = set()

    def node_id(cap: str) -> str:
        namespace, rest = cap.split("/", 1)
        return f"{namespace}_{rest.split('@', 1)[0]}"

    def add(cap: str, args: dict, bindings: dict) -> str:
        nid = node_id(cap)
        nodes.append({"node_id": nid, "capability_ref": cap, "args": args, "bindings": bindings, "failure_strategy": "FAIL_RUN"})
        present.add(cap)
        return nid

    phone = "phone_last4"
    stated_phone = str(entities.get("phone_last4")) if entities.get("phone_last4") is not None else None
    phone_args = {"phone_last4": stated_phone} if stated_phone is not None else {}
    phone_binding = {} if stated_phone is not None else {"phone_last4": {"kind": "context", "context_key": phone}}
    if "product/read@v1" in capabilities:
        add("product/read@v1", {"sku": str(entities.get("sku", "SKU-1001"))}, {})
    if "order/read@v1" in capabilities:
        add("order/read@v1", {"order_id": str(entities.get("order_id", "")), **phone_args}, dict(phone_binding))
    if "aftersales/read@v1" in capabilities:
        add("aftersales/read@v1", {"order_id": str(entities.get("order_id", "")), **phone_args}, dict(phone_binding))
    if "aftersales/eligibility@v1" in capabilities:
        add("aftersales/eligibility@v1", {"order_id": str(entities.get("order_id", "")), "service": str(entities.get("service", "refund")), **phone_args}, dict(phone_binding))
    if "policy/read@v1" in capabilities:
        add("policy/read@v1", {"query": text}, {})
    if "logistics/read@v1" in capabilities:
        if "order/read@v1" in present:
            order_nid = node_id("order/read@v1")
            add("logistics/read@v1", {}, {
                "carrier_code": {"kind": "result", "source_node_id": order_nid, "path": "carrier_code"},
                "tracking_no": {"kind": "result", "source_node_id": order_nid, "path": "tracking_no"},
            })
            edges.append({"upstream_node_id": order_nid, "downstream_node_id": node_id("logistics/read@v1")})
        else:
            add("logistics/read@v1", {"carrier_code": str(entities.get("carrier_code", "yuantong")), "tracking_no": str(entities.get("tracking_no", ""))}, {})
    if "aftersales/write@v1" in capabilities:
        # A write node is only legal when it is not a clarification plan and an
        # eligibility predecessor exists (plan contract rule).
        if (not clarify) and "aftersales/eligibility@v1" in present:
            nid = add("aftersales/write@v1", {}, {})
            edges.append({"upstream_node_id": node_id("aftersales/eligibility@v1"), "downstream_node_id": nid})
    if "human/handoff@v1" in capabilities and not clarify:
        add("human/handoff@v1", {}, {})
    return {"schema_version": "r5.plan.v1", "nodes": tuple(nodes), "edges": tuple(edges), "needs_clarification": clarify, "clarification_reason": "gold" if clarify else None, "business_goal": text[:80]}


class OracleProvider:
    name = "oracle"

    def __init__(self, gold_by_text: Mapping[str, dict]):
        self.gold_by_text = gold_by_text

    def __call__(self, prompt, router_schema, plan_schema):
        text = prompt.rsplit("用户消息：", 1)[-1].split("\n")[0]
        gold = self.gold_by_text.get(text)
        if gold is None:
            return {"router": {"intents": ("UNKNOWN",), "business_goal": "unknown", "needs_clarification": True, "clarification_question": "请补充"}, "plan": {"schema_version": "r5.plan.v1", "nodes": (), "edges": (), "needs_clarification": True, "clarification_reason": "no_gold", "business_goal": "unknown"}}
        expected = gold["expected"]
        clarify = bool(expected["needs_clarification"])
        capabilities = list(expected["required_capabilities"]) + list(expected.get("optional_capabilities", []))
        return {
            "router": {"intents": tuple(expected["intents"]), "entities": dict(expected["entities"]), "missing_slots": (), "needs_clarification": clarify, "clarification_question": "请补充信息" if clarify else None, "business_goal": "|".join(expected["intents"])},
            "plan": _oracle_plan(capabilities, expected["entities"], text, clarify=clarify),
        }


class EmptyProvider:
    name = "empty_output"

    def __call__(self, prompt, router_schema, plan_schema):
        return {}


class AlwaysClarifyProvider:
    name = "always_clarify"

    def __call__(self, prompt, router_schema, plan_schema):
        return {"router": {"intents": ("UNKNOWN",), "business_goal": "clarify", "needs_clarification": True, "clarification_question": "请补充"}, "plan": {"schema_version": "r5.plan.v1", "nodes": (), "edges": (), "needs_clarification": True, "clarification_reason": "always", "business_goal": "clarify"}}


class KeywordProvider(DeterministicRouterPlannerAdapter):
    name = "keyword_router"


class WrongEntityProvider(DeterministicRouterPlannerAdapter):
    name = "wrong_entity"

    def __call__(self, prompt, router_schema, plan_schema):
        out = dict(super().__call__(prompt, router_schema, plan_schema))
        router = dict(out["router"])
        entities = dict(router.get("entities") or {})
        for key in list(entities):
            entities[key] = "WRONG-" + entities[key] if key != "order_id" else "00000000000"
        router["entities"] = entities
        plan = dict(out["plan"])
        plan["nodes"] = tuple({**dict(n), "args": {**dict(n.get("args", {})), **entities}} for n in plan.get("nodes", ()))
        out["router"], out["plan"] = router, plan
        return out


class MissingGoalProvider(DeterministicRouterPlannerAdapter):
    name = "missing_goal"

    def __call__(self, prompt, router_schema, plan_schema):
        return {"router": {"intents": ("PRODUCT_QUERY",), "entities": {"sku": "SKU-1001"}, "business_goal": "product"}, "plan": {"schema_version": "r5.plan.v1", "nodes": ({"node_id": "product_read", "capability_ref": "product/read@v1", "args": {"sku": "SKU-1001"}, "bindings": {}, "failure_strategy": "FAIL_RUN"},), "edges": (), "needs_clarification": False, "business_goal": "product"}}


class UnrequestedWriteProvider(DeterministicRouterPlannerAdapter):
    """Plans a write node for a read-only request; must be rejected, not run."""

    name = "unrequested_write"

    def __call__(self, prompt, router_schema, plan_schema):
        return {"router": {"intents": ("AFTERSALES_CREATE",), "entities": {}, "business_goal": "write"}, "plan": {"schema_version": "r5.plan.v1", "nodes": ({"node_id": "write", "capability_ref": "aftersales/write@v1", "args": {}, "bindings": {}, "failure_strategy": "FAIL_RUN"},), "edges": (), "needs_clarification": False, "business_goal": "write"}}


class OverreachProvider:
    name = "overreach"

    def __call__(self, prompt, router_schema, plan_schema):
        return {"router": {"intents": ("ORDER_QUERY",), "entities": {}, "business_goal": "order"}, "plan": {"schema_version": "r5.plan.v1", "nodes": ({"node_id": "admin", "capability_ref": "admin/refund@v1"},), "edges": (), "needs_clarification": False, "business_goal": "order"}}


def _real_provider():
    from eval.r5_real_provider import RealRouterPlannerProvider

    return RealRouterPlannerProvider()


CANDIDATES = {
    "oracle": None,
    "keyword_router": lambda gold: KeywordProvider(),
    "deterministic_keyword": lambda gold: KeywordProvider(),
    "empty_output": lambda gold: EmptyProvider(),
    "always_clarify": lambda gold: AlwaysClarifyProvider(),
    "wrong_entity": lambda gold: WrongEntityProvider(),
    "missing_goal": lambda gold: MissingGoalProvider(),
    "unrequested_write": lambda gold: UnrequestedWriteProvider(),
    "overreach": lambda gold: OverreachProvider(),
    "real": lambda gold: _real_provider(),
}


def _f1(tp, fp, fn):
    if tp == 0:
        return 0.0
    p, r = tp / (tp + fp), tp / (tp + fn)
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def _ratio(n, d):
    return n / d if d else 0.0


def _intent_metric_counts(gold: set[str], pred: set[str], labels: tuple[str, ...]) -> dict[str, dict[str, int]]:
    """Return multilabel counts over an explicit, stable label collection."""
    return {
        label: {
            "tp": int(label in gold and label in pred),
            "fp": int(label not in gold and label in pred),
            "fn": int(label in gold and label not in pred),
        }
        for label in labels
    }


def _format_intent_metrics(counts: Mapping[str, Mapping[str, int]]) -> dict[str, dict[str, float | int]]:
    """Format per-label precision/recall/F1 without changing its label space."""
    formatted: dict[str, dict[str, float | int]] = {}
    for label, values in counts.items():
        tp = int(values.get("tp", 0))
        fp = int(values.get("fp", 0))
        fn = int(values.get("fn", 0))
        formatted[label] = {
            "precision": _ratio(tp, tp + fp),
            "recall": _ratio(tp, tp + fn),
            "f1": _f1(tp, fp, fn),
            "support": tp + fn,
        }
    return formatted


def _macro(metric_by_label: Mapping[str, Mapping[str, float | int]], key: str) -> float:
    values = [float(item[key]) for item in metric_by_label.values()]
    return sum(values) / len(values) if values else 0.0


def _handoff_legal(
    *,
    schema_valid: bool,
    predicted: set[str],
    required: set[str],
    optional: set[str],
    forbidden: set[str],
) -> bool:
    """Check the complete handoff contract, including optional and forbidden."""
    allowed = required | optional
    return bool(
        schema_valid
        and required.issubset(predicted)
        and predicted.issubset(allowed)
        and not (predicted & forbidden)
    )


def _safe_entity_summary(entities: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Persist entity presence and stable digests without persisting values."""
    if not isinstance(entities, Mapping):
        return {}
    summary: dict[str, dict[str, Any]] = {}
    for raw_key, value in sorted(entities.items(), key=lambda item: str(item[0])):
        key = str(raw_key)
        present = value is not None and str(value) != ""
        summary[key] = {
            "present": present,
            "value_digest": _digest(value) if present else None,
        }
    return summary


_MISSING_EXPECTED = object()


def _result_expected_fields(capability_ref: str, expected_entities: Mapping[str, Any]) -> dict[str, Any]:
    """Return result fields that identify the requested business target.

    A missing expected entity is represented by a presence-only assertion.  It
    commonly occurs for a derived logistics lookup where order/read supplies
    the target; the order result assertion still protects the root target.
    """
    if capability_ref == "product/read@v1":
        return {"sku": expected_entities.get("sku", _MISSING_EXPECTED)}
    if capability_ref == "order/read@v1":
        return {"order_id": expected_entities.get("order_id", _MISSING_EXPECTED)}
    if capability_ref == "logistics/read@v1":
        return {
            "carrier_code": expected_entities.get("carrier_code", _MISSING_EXPECTED),
            "tracking_no": expected_entities.get("tracking_no", _MISSING_EXPECTED),
        }
    if capability_ref == "aftersales/read@v1":
        return {"order_id": expected_entities.get("order_id", _MISSING_EXPECTED)}
    if capability_ref == "aftersales/eligibility@v1":
        order_id = expected_entities.get("order_id", _MISSING_EXPECTED)
        service = expected_entities.get("service", _MISSING_EXPECTED)
        entity_id = _MISSING_EXPECTED
        if order_id is not _MISSING_EXPECTED and service is not _MISSING_EXPECTED:
            entity_id = f"{order_id}:{service}"
        return {"entity_id": entity_id, "service": service}
    # Policy has no stable user entity in its canonical result.  The payload
    # presence check below still ensures a claimed successful read has a
    # canonical result to support it.
    return {}


def _build_result_assertions(
    expected_entities: Mapping[str, Any],
    outcome: Any,
    required_reads: set[str],
) -> tuple[list[dict[str, Any]], bool]:
    """Build safe target assertions and the required-read assertion verdict.

    The raw payload is used only while evaluating booleans/digests.  Persisted
    assertions contain no domain values, so a wrong target cannot be hidden by
    the payload redaction performed elsewhere in the report.
    """
    assertions: list[dict[str, Any]] = []
    seen_required: set[str] = set()
    for node in getattr(outcome, "node_results", ()):
        capability = str(getattr(node, "capability_ref", ""))
        status = str(getattr(node, "status", ""))
        payload = getattr(node, "payload", None)
        payload_map = payload if isinstance(payload, Mapping) else {}
        expected_fields = _result_expected_fields(capability, expected_entities)
        checks: list[dict[str, Any]] = []
        for field, expected in expected_fields.items():
            actual_present = field in payload_map and payload_map.get(field) is not None and str(payload_map.get(field)) != ""
            if expected is _MISSING_EXPECTED:
                matches = actual_present
                expected_known = False
                expected_digest = None
            else:
                expected_known = True
                matches = actual_present and str(payload_map.get(field)) == str(expected)
                expected_digest = _digest(expected)
            checks.append({
                "field": field,
                "expected_known": expected_known,
                "expected_digest": expected_digest,
                "actual_present": actual_present,
                "actual_digest": _digest(payload_map.get(field)) if actual_present else None,
                "matches": matches,
            })
        succeeded = bool(getattr(node, "ok", False)) and status == "SUCCEEDED"
        passed = bool(succeeded and isinstance(payload, Mapping) and all(check["matches"] for check in checks))
        assertion = {
            "node_id": str(getattr(node, "node_id", "")),
            "capability_ref": capability,
            "status": status,
            "payload_present": isinstance(payload, Mapping),
            "checks": checks,
            "passed": passed,
            "required": capability in required_reads,
        }
        assertions.append(assertion)
        if capability in required_reads and succeeded:
            seen_required.add(capability)

    for capability in sorted(required_reads - seen_required):
        assertions.append({
            "node_id": None,
            "capability_ref": capability,
            "status": "MISSING_REQUIRED_RESULT",
            "payload_present": False,
            "checks": [],
            "passed": False,
            "required": True,
        })

    required_ok = True
    for capability in required_reads:
        matching = [item for item in assertions if item["capability_ref"] == capability]
        if not matching or not all(bool(item["passed"]) for item in matching):
            required_ok = False
            break
    return assertions, required_ok


def _task_outcome_ok(
    *,
    expected_terminal: str,
    schema_valid: bool,
    unrequested_write: bool,
    reads_ok: bool,
    outcome_status: str,
    outcome_terminal: str,
    outcome_error: str | None,
    result_assertions_ok: bool = True,
) -> bool | None:
    """Evaluate the supported stage terminal without inspecting answer text."""
    if expected_terminal not in EXECUTABLE_TERMINALS:
        return None
    if expected_terminal == "CLARIFY":
        return schema_valid and outcome_status == "CLARIFY"
    if expected_terminal == "PENDING_CONFIRMATION":
        return bool(schema_valid and result_assertions_ok and not unrequested_write and reads_ok and outcome_status == "WAITING_CONFIRMATION")
    if expected_terminal == "REJECT":
        return schema_valid and outcome_status == "FAILED" and outcome_error in BUSINESS_REJECTION_CODES
    if expected_terminal == "HUMAN":
        return schema_valid and result_assertions_ok and outcome_terminal == "HUMAN"
    return schema_valid and result_assertions_ok and not unrequested_write and reads_ok and outcome_status == "COMPLETED"


def _write_checkpoint_header(path: Path, *, split: str, candidate: str, total: int) -> None:
    """Start a fresh per-run NDJSON checkpoint without exposing case values."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "checkpoint_version": "r5.eval.case.v1",
        "split": split,
        "candidate": candidate,
        "total_cases": total,
    }
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(header, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _append_checkpoint_case(path: Path, *, index: int, total: int, case_result: Mapping[str, Any], elapsed_ms: float) -> None:
    """Append one completed case so long real runs leave auditable progress."""
    record = {
        "record_type": "case",
        "case_index": index,
        "total_cases": total,
        "case_id": case_result.get("case_id"),
        "elapsed_ms": round(float(elapsed_ms), 3),
        "result": dict(case_result),
    }
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def evaluate(
    split: str,
    candidate: str,
    *,
    limit: int | None = None,
    stride: int | None = None,
    data_dir: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    progress: bool = False,
) -> dict[str, Any]:
    cases = load_split(split)
    if stride and stride > 1:
        cases = cases[::stride]
    if limit:
        cases = cases[:limit]
    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    if checkpoint is not None:
        _write_checkpoint_header(checkpoint, split=split, candidate=candidate, total=len(cases))
    gold_by_text = {c["user_text"]: c for c in cases}
    provider = OracleProvider(gold_by_text) if candidate == "oracle" else CANDIDATES[candidate](gold_by_text)
    boundary = R5RouterPlannerBoundary()
    executor = R5PlanExecutor(**_demo_paths(data_dir))
    trusted = SESSION[split]

    intent_tp: dict[str, int] = {label: 0 for label in R5_INTENT_LABELS}
    intent_fp: dict[str, int] = {label: 0 for label in R5_INTENT_LABELS}
    intent_fn: dict[str, int] = {label: 0 for label in R5_INTENT_LABELS}
    intent_set_exact = 0
    intent_extra_labels = 0
    intent_missing_labels = 0
    intent_pred_outside_fixed = 0
    ent_tp = ent_fp = ent_fn = 0
    necessary_total = necessary_hit = unnecessary_total = unnecessary_hit = 0
    first_schema = final_schema = 0
    coverage_hits = conformance_hits = clarif_match = legacy_hits = 0
    handoff_legal_hits = handoff_forbidden = 0
    overreach = unrequested_write = 0
    required_reads_ok = 0
    task_total = task_success = 0
    result_assertions_ok_count = 0
    result_assertions_total = 0
    model_conf_nodes = model_total_nodes = normalized_cases = unplannable_cases = 0
    excluded_reasons: dict[str, int] = defaultdict(int)
    excluded_cases: list[dict[str, Any]] = []
    provider_call_attempts = provider_returned_attempts = 0
    latencies: list[float] = []
    per_case: list[dict[str, Any]] = []

    for case_index, case in enumerate(cases, start=1):
        case_started = time.perf_counter()
        exp = case["expected"]
        ex_meta = case["execution"]
        gold_intents = set(exp["intents"])
        gold_caps = set(exp["required_capabilities"])
        optional_caps = set(exp.get("optional_capabilities", []))
        forbidden = set(exp.get("forbidden_capabilities", []))
        gold_clarify = bool(exp["needs_clarification"])
        expected_terminal = ex_meta["expected_terminal"]
        required_reads = set(ex_meta["required_reads"])
        allows_write = bool(ex_meta["allows_write"])

        started = time.perf_counter()
        result = boundary.run(user_message=case["user_text"], provider=provider)
        latencies.append((time.perf_counter() - started) * 1000)

        # The boundary records one attempt object per provider invocation.  A
        # PROVIDER_ERROR means the invocation was made but did not return a
        # usable provider result; keep called and returned counts separate.
        attempts = [_model_dump(attempt) for attempt in result.attempts]
        provider_call_attempts += len(attempts) if result.provider_called else 0
        provider_returned_attempts += sum(1 for attempt in attempts if attempt.get("error_code") != "PROVIDER_ERROR")

        if result.first_attempt_schema_valid:
            first_schema += 1
        if result.eventual_schema_valid:
            final_schema += 1

        pred_intents = set(result.router.intents)
        fixed_counts = _intent_metric_counts(gold_intents, pred_intents, R5_INTENT_LABELS)
        for label, counts in fixed_counts.items():
            intent_tp[label] += counts["tp"]
            intent_fp[label] += counts["fp"]
            intent_fn[label] += counts["fn"]
        intent_set_exact += int(pred_intents == gold_intents)
        intent_extra_labels += len(pred_intents - gold_intents)
        intent_missing_labels += len(gold_intents - pred_intents)
        intent_pred_outside_fixed += len(pred_intents - set(R5_INTENT_LABELS))

        gold_entities = {str(k): str(v) for k, v in exp["entities"].items()}
        pred_entities = {str(k): str(v) for k, v in result.router.entities.items()}
        for key, value in pred_entities.items():
            if gold_entities.get(key) == value:
                ent_tp += 1
            else:
                ent_fp += 1
        for key, value in gold_entities.items():
            if pred_entities.get(key) != value:
                ent_fn += 1

        if gold_clarify:
            necessary_total += 1
            necessary_hit += int(result.router.needs_clarification)
        else:
            unnecessary_total += 1
            unnecessary_hit += int(result.router.needs_clarification)

        # Plan-level constructs. Clarification is read from the plan, the single
        # authoritative structure, not from the router flag.
        caps = set(result.plan.derived_capabilities())
        plan_clarify = bool(result.plan.needs_clarification)
        this_clarif_match = plan_clarify == gold_clarify
        this_coverage = gold_caps.issubset(caps) if not gold_clarify else this_clarif_match
        allowed_caps = gold_caps | optional_caps
        this_handoff_legal = _handoff_legal(
            schema_valid=result.eventual_schema_valid,
            predicted=caps,
            required=gold_caps,
            optional=optional_caps,
            forbidden=forbidden,
        )
        this_conformance = this_handoff_legal and this_clarif_match
        this_overreach = bool(caps - allowed_caps)
        this_forbidden = bool(caps & forbidden)
        this_unrequested_write = bool((caps & R5_WRITE_CAPABILITIES) - (gold_caps | optional_caps))
        this_legacy = result.eventual_schema_valid and this_coverage and this_clarif_match

        coverage_hits += int(this_coverage)
        conformance_hits += int(this_conformance)
        clarif_match += int(this_clarif_match)
        legacy_hits += int(this_legacy)
        handoff_legal_hits += int(this_handoff_legal)
        handoff_forbidden += int(this_forbidden)
        overreach += int(this_overreach)
        unrequested_write += int(this_unrequested_write)

        # Real execution of the model's own plan, after deterministic argument
        # normalization (trusted context / upstream derivation). Capabilities
        # are never added or removed here.
        if result.eventual_schema_valid:
            norm = normalize_plan(result.plan, entities=result.router.entities, trusted_context=trusted, user_text=case["user_text"])
            plan_payload: Any = norm.plan
            model_conf_nodes += norm.model_arg_conformant_nodes
            model_total_nodes += norm.total_nodes
            if norm.actions:
                normalized_cases += 1
            if norm.unplannable_reason:
                unplannable_cases += 1
        else:
            plan_payload = {"invalid": True}
            norm = None

        outcome = executor.execute(plan_payload, trusted_context=trusted, allows_write=allows_write, allows_escalation=bool(ex_meta.get("allows_escalation", False)))
        reads_ok = required_reads.issubset(set(outcome.successful_reads))
        if required_reads:
            required_reads_ok += int(reads_ok)
        result_assertions, required_result_assertions_ok = _build_result_assertions(gold_entities, outcome, required_reads)
        if required_reads:
            result_assertions_total += 1
            result_assertions_ok_count += int(required_result_assertions_ok)

        # Task completion: supported stage terminal + business state.
        # Qualification-only combos accept either a plain answer or a pending
        # confirmation, since both are defensible for an informational request.
        acceptable_terminals = tuple(ex_meta.get("acceptable_terminals") or ())
        if len(acceptable_terminals) > 1:
            if "CLARIFY" in acceptable_terminals:
                task_ok = bool(result.eventual_schema_valid and outcome.status == "CLARIFY")
            else:
                task_ok = bool(
                    result.eventual_schema_valid
                    and required_result_assertions_ok
                    and not outcome.unrequested_write
                    and reads_ok
                    and outcome.terminal in set(acceptable_terminals)
                )
        else:
            task_ok = _task_outcome_ok(
                expected_terminal=expected_terminal,
                schema_valid=result.eventual_schema_valid,
                unrequested_write=outcome.unrequested_write,
                reads_ok=reads_ok,
                outcome_status=outcome.status,
                outcome_terminal=outcome.terminal,
                outcome_error=outcome.error_code,
                result_assertions_ok=required_result_assertions_ok,
            )
        if task_ok is None:
            excluded_reasons[f"terminal_not_implemented:{expected_terminal}"] += 1
        else:
            task_total += 1
            task_success += int(bool(task_ok))

        task_excluded_reason = None
        if task_ok is None:
            task_excluded_reason = f"terminal_not_implemented:{expected_terminal}"
            excluded_cases.append({"case_id": case["case_id"], "expected_terminal": expected_terminal, "reason": task_excluded_reason})

        case_result = {
            "case_id": case["case_id"],
            "family_id": case["family_id"],
            "gold_intents": sorted(gold_intents),
            "pred_intents": sorted(pred_intents),
            "intent_set_exact": pred_intents == gold_intents,
            "gold_caps": sorted(gold_caps),
            "pred_caps": sorted(caps),
            "allowed_caps": sorted(allowed_caps),
            "forbidden_caps": sorted(forbidden),
            "exact_handoff": caps == gold_caps,
            "handoff_legal": this_handoff_legal,
            "gold_clarify": gold_clarify,
            "router_clarify": result.router.needs_clarification,
            "plan_clarify": plan_clarify,
            "schema_valid": result.eventual_schema_valid,
            "stage_policy": {
                "allows_write": allows_write,
                "allows_escalation": bool(ex_meta.get("allows_escalation", False)),
                "write_requires_confirmation": bool(ex_meta.get("write_requires_confirmation", False)),
                "source": "case.execution metadata; controlled evaluator stage policy",
            },
            "first_root_cause": result.first_root_cause,
            "provider_called": result.provider_called,
            "provider_returned": result.provider_returned,
            "attempts": _redact(attempts),
            "router": _redact(_model_dump(result.router)),
            "plan": _redact(_model_dump(result.plan)),
            "gold_entities_summary": _safe_entity_summary(gold_entities),
            "pred_entities_summary": _safe_entity_summary(pred_entities),
            "plan_coverage": this_coverage,
            "plan_conformance": this_conformance,
            "overreach": this_overreach,
            "forbidden_capability": this_forbidden,
            "unrequested_write": this_unrequested_write,
            "expected_terminal": expected_terminal,
            "exec_status": outcome.status,
            "exec_terminal": outcome.terminal,
            "exec_error": outcome.error_code,
            "required_reads": sorted(required_reads),
            "successful_reads": sorted(set(outcome.successful_reads)),
            "reads_ok": reads_ok,
            "result_assertions": result_assertions,
            "result_assertions_ok": required_result_assertions_ok,
            "required_result_assertions_ok": required_result_assertions_ok,
            "normalization_actions": [] if norm is None else [a.__dict__ for a in norm.actions],
            "normalized_plan": None if norm is None else _redact(_model_dump(norm.plan)),
            "execution_outcome": _safe_execution_outcome(outcome),
            "unplannable_reason": None if norm is None else norm.unplannable_reason,
            "task_ok": task_ok,
            "task_excluded_reason": task_excluded_reason,
        }
        per_case.append(case_result)
        elapsed_ms = (time.perf_counter() - case_started) * 1000
        if checkpoint is not None:
            _append_checkpoint_case(checkpoint, index=case_index, total=len(cases), case_result=case_result, elapsed_ms=elapsed_ms)
        if progress:
            print(f"[r5-eval] case {case_index}/{len(cases)} case_id={case_result['case_id']} elapsed_ms={elapsed_ms:.1f}", flush=True)

    n = len(cases)
    per_class = _format_intent_metrics({label: {"tp": intent_tp[label], "fp": intent_fp[label], "fn": intent_fn[label]} for label in R5_INTENT_LABELS})
    macro_f1 = _macro(per_class, "f1")
    macro_recall = _macro(per_class, "recall")
    core_per_class = {label: per_class[label] for label in R5_CORE_INTENT_LABELS}
    core_macro_f1 = _macro(core_per_class, "f1")
    core_macro_recall = _macro(core_per_class, "recall")
    multi_values = per_class[MULTI_INTENT_LABEL]
    latencies.sort()

    provider_usage = None
    if hasattr(provider, "calls"):
        provider_usage = {
            "model_calls": int(getattr(provider, "calls", 0)),
            "provider_call_attempts": int(getattr(provider, "call_attempts", provider_call_attempts)),
            "provider_returned_attempts": int(getattr(provider, "returned_calls", provider_returned_attempts)),
            "input_tokens": int(getattr(provider, "total_input_tokens", 0)),
            "output_tokens": int(getattr(provider, "total_output_tokens", 0)),
            "total_tokens": int(getattr(provider, "total_tokens", 0)),
            "output_budget_tokens": getattr(provider, "max_tokens", None),
            "cost": "unknown (provider pricing not configured; not filled with 0)",
        }

    return {
        "candidate": candidate,
        "split": split,
        "unique_case_N": n,
        "observation_N": n,
        "first_pass_schema_valid": _ratio(first_schema, n),
        "eventual_schema_valid": _ratio(final_schema, n),
        "intent_label_set": list(R5_INTENT_LABELS),
        "intent_core_label_set": list(R5_CORE_INTENT_LABELS),
        "intent_macro_f1": macro_f1,
        "intent_macro_recall": macro_recall,
        "intent_core_macro_f1": core_macro_f1,
        "intent_core_macro_recall": core_macro_recall,
        "unknown_recall": per_class.get("UNKNOWN", {}).get("recall", 0.0),
        "per_class": per_class,
        "intent_core_per_class": core_per_class,
        "multi_intent_auxiliary": {
            "label": MULTI_INTENT_LABEL,
            "gold_positive": int(multi_values["support"]),
            "pred_positive": int(intent_tp[MULTI_INTENT_LABEL] + intent_fp[MULTI_INTENT_LABEL]),
            "tp": int(intent_tp[MULTI_INTENT_LABEL]),
            "fp": int(intent_fp[MULTI_INTENT_LABEL]),
            "fn": int(intent_fn[MULTI_INTENT_LABEL]),
            "precision": multi_values["precision"],
            "recall": multi_values["recall"],
            "f1": multi_values["f1"],
            "extra_pred_count": sum(1 for item in per_case if MULTI_INTENT_LABEL in item["pred_intents"] and MULTI_INTENT_LABEL not in item["gold_intents"]),
        },
        "intent_set_exact_rate": _ratio(intent_set_exact, n),
        "intent_extra_label_count": intent_extra_labels,
        "intent_missing_label_count": intent_missing_labels,
        "intent_pred_outside_fixed_label_count": intent_pred_outside_fixed,
        "entity_micro_f1": _f1(ent_tp, ent_fp, ent_fn),
        "exact_handoff": _ratio(len([c for c in per_case if set(c["pred_caps"]) == set(c["gold_caps"])]), n),
        "handoff_legal_rate": _ratio(handoff_legal_hits, n),
        "handoff_forbidden_rate": _ratio(handoff_forbidden, n),
        "plan_required_goal_coverage_rate": _ratio(coverage_hits, n),
        "plan_conformance_rate": _ratio(conformance_hits, n),
        "clarification_match_rate": _ratio(clarif_match, n),
        "plan_capability_coverage_and_clarification_match_rate": _ratio(legacy_hits, n),
        "plan_overreach_rate": _ratio(overreach, n),
        "unrequested_write_rate": _ratio(unrequested_write, n),
        "necessary_clarification_recall": _ratio(necessary_hit, necessary_total) if necessary_total else None,
        "unnecessary_clarification_rate": _ratio(unnecessary_hit, unnecessary_total) if unnecessary_total else None,
        "required_reads_success_rate": _ratio(required_reads_ok, sum(1 for c in per_case if c["required_reads"])),
        "required_result_assertions_rate": _ratio(result_assertions_ok_count, result_assertions_total),
        "model_arg_conformance_rate": _ratio(model_conf_nodes, model_total_nodes),
        "runtime_normalization_case_rate": _ratio(normalized_cases, n),
        "unplannable_case_rate": _ratio(unplannable_cases, n),
        "task_completion_rate": _ratio(task_success, task_total),
        "task_completion_denominator": task_total,
        "task_completion_supported_N": task_total,
        "task_completion_all_N": n,
        "task_completion_rate_supported_N": _ratio(task_success, task_total),
        "task_completion_rate_all_N": _ratio(task_success, n),
        "task_completion_excluded": dict(excluded_reasons),
        "task_completion_excluded_cases": excluded_cases,
        "task_completion_scope_note": "阶段执行与预期终态匹配；不检查最终用户回答内容，也不表示真实写入已提交。",
        "execution_policy_note": "allows_write/ allows_escalation 来自 case.execution 元数据，仅作为受控评测阶段策略传入 executor，不代表模型自行取得授权或该项授权判断得分。",
        "provider_call_attempts": provider_call_attempts,
        "provider_returned_attempts": provider_returned_attempts,
        "legacy_metric_note": "plan_capability_coverage_and_clarification_match_rate is a plan-text metric and does NOT represent task completion",
        "latency_ms_p50": latencies[len(latencies) // 2] if latencies else 0.0,
        "latency_ms_p95": latencies[int(len(latencies) * 0.95)] if latencies else 0.0,
        "provider_usage": provider_usage,
        "per_case": per_case,
    }


def _data_run_context(data_dir: str | Path | None) -> dict[str, Any]:
    """Describe database provenance so seeded and repository runs cannot mix."""
    paths = _demo_paths(data_dir)
    if data_dir is None:
        source_kind = "repository_demo_data"
        source_version = "repository-demo-v1"
        component_versions = dict(FIXTURE_SOURCE_VERSIONS)
        manifest_path = None
        data_root = None
    else:
        data_root = Path(data_dir).expanduser().resolve()
        manifest_path = data_root / "r5_synthetic_manifest.json"
        source_kind = "custom_data_dir"
        source_version = None
        component_versions = {}
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                manifest = {}
            source = manifest.get("source") if isinstance(manifest, Mapping) else None
            if isinstance(source, Mapping):
                source_kind = str(source.get("kind") or source_kind)
                source_version = str(source.get("version")) if source.get("version") is not None else None
            if isinstance(manifest, Mapping):
                source_version = str(manifest.get("dataset_version") or source_version or "unknown")
                versions = manifest.get("component_versions")
                if isinstance(versions, Mapping):
                    component_versions = {str(k): str(v) for k, v in versions.items()}

    return {
        "data_dir": str(data_root) if data_root is not None else "<repository-default>",
        "source_kind": source_kind,
        "source_version": source_version,
        "component_source_versions": component_versions,
        "database_hashes": {name: _file_sha256(path) for name, path in paths.items()},
        "manifest_sha256": _file_sha256(manifest_path) if manifest_path is not None else None,
    }


def _run_context(candidate: str, *, data_dir: str | Path | None = None) -> dict[str, Any]:
    """Non-secret version tuple for traceability (model, data, prompt)."""
    # Deterministic candidates must stay independent of local credentials.  The
    # legacy ``agent.llm`` module loads ``.env`` at import time, so only the
    # real provider path is allowed to import it here.
    if candidate == "real":
        import agent.llm  # noqa: F401  (loads .env into the process)

        base = os.getenv("OPENAI_BASE_URL", "")
        model = os.getenv("OPENAI_MODEL")
        credential_present = bool(os.getenv("OPENAI_API_KEY"))
    else:
        base = ""
        model = None
        credential_present = False
    return {
        "candidate": candidate,
        "model": model,
        "provider_host": urlparse(base).netloc or None if candidate == "real" else None,
        "prompt_kind": "r5.router_planner.prompt.v2",
        "output_budget_tokens": 8000,
        "model_attempt_budget": 2,
        "credential_present": credential_present,
        "data": _data_run_context(data_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev", choices=["dev", "validation"])
    parser.add_argument("--candidate", required=True, choices=list(CANDIDATES))
    parser.add_argument("--out", default=None)
    parser.add_argument("--checkpoint", default=None, help="Per-case NDJSON checkpoint; defaults beside --out in artifacts/final-20260911")
    parser.add_argument("--data-dir", default=None, help="Optional isolated fixture directory containing the four R5 databases")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    args = parser.parse_args()
    out = Path(args.out) if args.out else PROJECT_ROOT / "artifacts" / "final-20260911" / f"router_planner_{args.split}_{args.candidate}.json"
    checkpoint = Path(args.checkpoint) if args.checkpoint else out.with_suffix(".checkpoint.jsonl")
    report = evaluate(args.split, args.candidate, limit=args.limit, stride=args.stride, data_dir=args.data_dir, checkpoint_path=checkpoint, progress=True)
    report["run_context"] = _run_context(args.candidate, data_dir=args.data_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"per_case", "per_class"}}, ensure_ascii=False, indent=2, sort_keys=True))
    print("written:", out)


if __name__ == "__main__":
    main()
