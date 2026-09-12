"""Independent R1.5 router evaluation over frozen literal inputs.

The evaluator never authors prompts from gold labels.  Target and baseline
runners receive exactly the text stored in ``*-inputs.jsonl`` and emit the
same observation shape before metrics are computed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from agent.interactive_runtime import ChatOpenAIStructuredProvider, LLMRuntimeConfig, ModelClient
from agent.r1_5_router import BusinessIntent, RoutingDecisionV1, build_router_prompt, normalize_decision


LABELS = tuple(item.value for item in BusinessIntent)
ENTITY_NAMES = ("order_id", "phone_last4", "carrier_code", "tracking_no", "product_sku", "aftersales_ticket_id")


def _jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _hash_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def load_frozen(inputs_path: str | Path, gold_path: str | Path) -> list[dict[str, Any]]:
    inputs, gold = _jsonl(inputs_path), _jsonl(gold_path)
    input_by_id = {row["case_id"]: row for row in inputs}
    gold_by_id = {row["case_id"]: row for row in gold}
    if len(input_by_id) != len(inputs) or len(gold_by_id) != len(gold):
        raise ValueError("duplicate case_id in frozen data")
    if set(input_by_id) != set(gold_by_id):
        raise ValueError("input/gold case ids differ")
    return [{**input_by_id[key], "gold": gold_by_id[key]} for key in input_by_id]


def _entities(decision: RoutingDecisionV1 | Mapping[str, Any]) -> dict[str, str]:
    if isinstance(decision, RoutingDecisionV1):
        return decision.entity_map
    raw = decision.get("entities", ())
    if isinstance(raw, Mapping):
        return {str(k): str(v) for k, v in raw.items()}
    return {str(item["name"]): str(item["value"]) for item in raw if isinstance(item, Mapping) and item.get("name") and item.get("value")}


def _observation(case_id: str, decision: RoutingDecisionV1 | Mapping[str, Any], latency_ms: int = 0, usage: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if isinstance(decision, RoutingDecisionV1):
        decision = normalize_decision(decision)
        primary = decision.primary_intent.value
        secondary = [item.value for item in decision.secondary_intents]
        confidence = decision.confidence
        needs_clarification = decision.needs_clarification
    else:
        primary = str(decision["primary_intent"])
        secondary = [str(item) for item in decision.get("secondary_intents", ())]
        confidence = float(decision.get("confidence", 0.0))
        needs_clarification = bool(decision.get("needs_clarification", False))
    row = {"case_id": case_id, "primary_intent": primary, "secondary_intents": secondary,
            "entities": _entities(decision), "confidence": confidence, "latency_ms": latency_ms,
            "needs_clarification": needs_clarification, "usage": dict(usage or {}), "error_code": None, "tool_calls": 0}
    row["output_hash"] = _hash_json({key: row[key] for key in ("primary_intent", "secondary_intents", "entities", "confidence", "needs_clarification")})
    return row


def _extract_entities(text: str) -> dict[str, str]:
    patterns = {
        "order_id": r"\bORDR\d{6}\b", "phone_last4": r"(?<!\d)\d{4}(?!\d)",
        "tracking_no": r"\bTRKX\d{7}\b", "product_sku": r"\bSKU-X\d{5}\b",
        "aftersales_ticket_id": r"\bAS-X\d{6}\b",
    }
    found = {name: match.group(0) for name, pattern in patterns.items() if (match := re.search(pattern, text, re.I))}
    carrier = re.search(r"\b(sf|jd|yto|zto)\b", text, re.I)
    if carrier:
        found["carrier_code"] = carrier.group(1).lower()
    return found


def _baseline_observation(name: str, text: str, majority: str, rng: random.Random) -> Mapping[str, Any]:
    lowered = text.lower()
    if name == "majority":
        label = majority
    elif name == "constant_order":
        label = "ORDER_QUERY"
    elif name == "seeded_random":
        label = rng.choice(LABELS)
    elif name == "keyword":
        groups = (
            ("COMPLAINT", ("投诉", "人工", "poor service")),
            ("AFTERSALES_CREATE", ("申请退款", "发起退货", "换货", "提交售后", "refund for", "aftersales request")),
            ("AFTERSALES_STATUS", ("售后单", "工单", "服务单", "ticket")),
            ("POLICY_QA", ("规则", "条件", "政策", "多久到账", "支持无理由", "return policy", "refund rules")),
            ("PRODUCT_QA", ("sku", "型号", "尺寸", "颜色", "材质", "specification")),
            ("LOGISTICS_QUERY", ("物流", "包裹", "快递", "运单", "parcel", "tracking")),
            ("ORDER_QUERY", ("订单", "购买记录", "交易单", "purchase")),
            ("CHITCHAT", ("你好", "谢谢", "晚安", "hello", "good afternoon")),
        )
        hits = [label for label, words in groups if any(word in lowered for word in words)]
        if "ORDER_QUERY" in hits and "POLICY_QA" in hits:
            label = "MULTI_INTENT"
        elif "ORDER_QUERY" in hits and "LOGISTICS_QUERY" in hits:
            label = "ORDER_AND_LOGISTICS"
        else:
            label = hits[0] if hits else "UNKNOWN"
    else:
        raise ValueError(name)
    return {"primary_intent": label, "secondary_intents": [], "entities": _extract_entities(text), "confidence": 1.0}


def target_runner(db_path: str | Path) -> tuple[Callable[[str, str], dict[str, Any]], dict[str, Any]]:
    config = LLMRuntimeConfig.from_environment(db_path=db_path)
    client = ModelClient(config, ChatOpenAIStructuredProvider(config))
    def run(case_id: str, text: str) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            decision, usage, latency = client.complete(build_router_prompt(text), RoutingDecisionV1)
            normalized = normalize_decision(decision)
            row = _observation(case_id, normalized, latency, {"available": usage.available, "input": usage.input_tokens, "output": usage.output_tokens})
            row.update({"input_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(), "provider_called": True, "provider_returned": True})
            return row
        except Exception as exc:
            return {"case_id": case_id, "primary_intent": "ERROR", "secondary_intents": [], "entities": {}, "confidence": 0.0,
                    "latency_ms": int((time.perf_counter() - started) * 1000), "usage": {}, "error_code": getattr(exc, "code", type(exc).__name__),
                    "tool_calls": 0, "input_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(), "output_hash": None,
                    "provider_called": True, "provider_returned": False}
    return run, config.evidence()


def metrics(cases: Sequence[Mapping[str, Any]], observations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    observed = {row["case_id"]: row for row in observations}
    confusion = {label: {candidate: 0 for candidate in LABELS + ("ERROR",)} for label in LABELS}
    tp = Counter(); fp = Counter(); fn = Counter()
    entity_tp = entity_fp = entity_fn = 0
    errors = 0
    for case in cases:
        gold, row = case["gold"], observed[case["case_id"]]
        truth, pred = gold["primary_intent"], row["primary_intent"]
        confusion[truth][pred if pred in confusion[truth] else "ERROR"] += 1
        errors += int(bool(row.get("error_code")))
        for label in LABELS:
            tp[label] += int(truth == label and pred == label)
            fp[label] += int(truth != label and pred == label)
            fn[label] += int(truth == label and pred != label)
        expected_entities = {(k, str(v)) for k, v in gold.get("entities", {}).items()}
        actual_entities = {(k, str(v)) for k, v in row.get("entities", {}).items() if k in ENTITY_NAMES}
        entity_tp += len(expected_entities & actual_entities)
        entity_fp += len(actual_entities - expected_entities)
        entity_fn += len(expected_entities - actual_entities)
    per_label = {}
    for label in LABELS:
        precision = tp[label] / max(1, tp[label] + fp[label])
        recall = tp[label] / max(1, tp[label] + fn[label])
        per_label[label] = {"precision": precision, "recall": recall, "f1": 2 * precision * recall / max(1e-12, precision + recall), "support": tp[label] + fn[label]}
    ep = entity_tp / max(1, entity_tp + entity_fp); er = entity_tp / max(1, entity_tp + entity_fn)
    trace_complete = sum(bool(row.get("input_hash")) and bool(row.get("output_hash")) and row.get("provider_called") is not None and row.get("provider_returned") is not None for row in observations)
    return {"N": len(cases), "intent_accuracy": sum(tp.values()) / max(1, len(cases)), "route_exact_match": sum(tp.values()) / max(1, len(cases)),
            "intent_macro_f1": sum(row["f1"] for row in per_label.values()) / len(LABELS),
            "entity_precision": ep, "entity_recall": er, "entity_f1": 2 * ep * er / max(1e-12, ep + er),
            "error_count": errors, "trace_complete_rate": trace_complete / max(1, len(cases)),
            "prohibited_tool_calls": sum(int(row.get("tool_calls", 0)) for row in observations),
            "clarification_rate": sum(bool(row.get("needs_clarification", False)) for row in observations) / max(1, len(cases)),
            "per_label": per_label, "confusion_matrix": confusion}


def evaluate(*, inputs_path: str | Path, gold_path: str | Path, db_path: str | Path, output_path: str | Path,
             run_target: bool = False) -> dict[str, Any]:
    cases = load_frozen(inputs_path, gold_path)
    counts = Counter(case["gold"]["primary_intent"] for case in cases)
    majority = sorted(counts, key=lambda key: (-counts[key], key))[0]
    systems: dict[str, Any] = {}
    for name in ("majority", "constant_order", "keyword", "seeded_random"):
        rng = random.Random(1502)
        rows = []
        for case in cases:
            row = _observation(case["case_id"], _baseline_observation(name, case["text"], majority, rng))
            row.update({"input_hash": hashlib.sha256(case["text"].encode("utf-8")).hexdigest(), "provider_called": False, "provider_returned": True})
            rows.append(row)
        systems[name] = {"run_class": "executable_baseline", "metrics": metrics(cases, rows), "observations": rows}
    config_evidence = None
    if run_target:
        run, config_evidence = target_runner(db_path)
        rows = [run(case["case_id"], case["text"]) for case in cases]
        systems["target"] = {"run_class": "external_structured_model", "metrics": metrics(cases, rows), "observations": rows}
    best_baseline = max(item["metrics"]["intent_macro_f1"] for item in systems.values() if item["run_class"] == "executable_baseline")
    target_score = systems.get("target", {}).get("metrics", {}).get("intent_macro_f1")
    report = {"report_version": "r1.5.eval.v1", "dataset": {"inputs": str(inputs_path), "gold": str(gold_path),
              "inputs_sha256": _sha(inputs_path), "gold_sha256": _sha(gold_path), "N": len(cases)},
              "systems": systems, "best_baseline_macro_f1": best_baseline,
              "target_delta_over_best_baseline": None if target_score is None else target_score - best_baseline,
              "suspicious_perfect": bool(target_score == 1.0), "config_evidence": config_evidence}
    encoded = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    target = Path(output_path); target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded); handle.flush(); os.fsync(handle.fileno())
        os.replace(temp_name, target)
    finally:
        if os.path.exists(temp_name): os.unlink(temp_name)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", default="eval/datasets/r1_5/validation-inputs.jsonl")
    parser.add_argument("--gold", default="eval/datasets/r1_5/validation-gold.jsonl")
    parser.add_argument("--db-path", default="ecommerce.db")
    parser.add_argument("--output", default="artifacts/r1_5/eval-report.json")
    parser.add_argument("--target", action="store_true")
    parser.add_argument("--local-env-bootstrap", action="store_true")
    args = parser.parse_args(argv)
    if args.local_env_bootstrap:
        import agent.llm  # noqa: F401
    report = evaluate(inputs_path=args.inputs, gold_path=args.gold, db_path=args.db_path, output_path=args.output, run_target=args.target)
    print(json.dumps({"N": report["dataset"]["N"], "systems": {k: v["metrics"]["intent_macro_f1"] for k, v in report["systems"].items()}, "suspicious_perfect": report["suspicious_perfect"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
