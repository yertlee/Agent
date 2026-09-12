"""R5 capability argument contract and deterministic plan normalization.

The model chooses *which* capability a task needs and the semantic role of
each fact; it cannot be trusted to supply runtime-owned context (session
phone) or to re-derive facts that an upstream node already produced
(carrier/tracking from an order read).  This module declares, per capability,
where each argument comes from, and normalizes a model plan by filling those
deterministic sources.

It never uses gold: only the user's request entities, the trusted session
context and upstream node outputs.  It never adds or removes capabilities, so
capability-level scoring (coverage / over-reach) is unaffected.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping

from .r5_plan_contracts import R5PlanNodeV1, R5PlanV1, R5BindingV1


# Argument sources:
#   entity        - parsed from the user request (router entities)
#   context       - trusted session/runtime context (never model-authored)
#   request_text  - the raw user message
#   derived       - from an upstream node's result; falls back to entity
ENTITY = "entity"
CONTEXT = "context"
REQUEST_TEXT = "request_text"
DERIVED = "derived"
NODE_ARG = "node_arg"
# User-stated value if present, otherwise the trusted session value: ownership
# proofs (phone last-four) are user claims to be verified, not session defaults.
ENTITY_OR_CONTEXT = "entity_or_context"

CAPABILITY_ARGS: dict[str, dict[str, str]] = {
    "order/read@v1": {"order_id": ENTITY, "phone_last4": ENTITY_OR_CONTEXT},
    "aftersales/read@v1": {"order_id": ENTITY, "phone_last4": ENTITY_OR_CONTEXT},
    "aftersales/eligibility@v1": {"order_id": ENTITY, "phone_last4": ENTITY_OR_CONTEXT, "service": ENTITY},
    "logistics/read@v1": {"carrier_code": DERIVED, "tracking_no": DERIVED},
    "product/read@v1": {"sku": ENTITY},
    "policy/read@v1": {"query": REQUEST_TEXT},
    "aftersales/write@v1": {},
    "human/handoff@v1": {"summary": REQUEST_TEXT, "reason": REQUEST_TEXT},
}

# Caps whose result can satisfy a DERIVED argument.
DERIVED_SOURCE_CAPABILITIES = {"order/read@v1"}

# Entity aliases: argument name -> candidate router entity keys.
ENTITY_ALIASES: dict[str, tuple[str, ...]] = {
    "sku": ("sku", "product_sku"),
    "tracking_no": ("tracking_no",),
    "carrier_code": ("carrier_code",),
    "order_id": ("order_id",),
    "service": ("service",),
    "query": ("query",),
}

_TOKEN_RE = re.compile(r"^[^\s]{1,128}$")
_PHONE_LAST4_RE = re.compile(r"^\d{4}$")
_SERVICE_VALUES = frozenset({"refund", "return", "exchange"})
_SERVICE_ALIASES = {
    "refund": "refund", "return": "return", "exchange": "exchange",
    "退款": "refund", "退货": "return", "换货": "exchange",
}
# User-facing carrier names map to the canonical codes used by the demo
# snapshots; an unknown value is left as supplied (never invented).
_CARRIER_ALIASES = {
    "圆通": "yuantong", "圆通速递": "yuantong", "圆通快递": "yuantong",
    "中通": "zhongtong", "中通快递": "zhongtong",
    "韵达": "yunda", "韵达快递": "yunda",
    "申通": "shentong", "顺丰": "shunfeng", "京东": "jd",
}
# Common carrier codes/abbreviations a model may emit instead of the demo code.
_CARRIER_CODE_ALIASES = {
    "YTO": "yuantong", "ZTO": "zhongtong", "YD": "yunda", "YUNDA": "yunda",
    "STO": "shentong", "SF": "shunfeng", "SFEXPRESS": "shunfeng", "JD": "jd",
}


def _canonical_value(argument: str, value: Any) -> Any:
    """Map user-facing aliases (e.g. 退款 / 圆通 / YTO) to canonical values before legality."""
    if not _present(value):
        return value
    text = str(value).strip()
    if argument == "service":
        return _SERVICE_ALIASES.get(text, value)
    if argument == "carrier_code":
        if text in _CARRIER_ALIASES:
            return _CARRIER_ALIASES[text]
        if text.upper() in _CARRIER_CODE_ALIASES:
            return _CARRIER_CODE_ALIASES[text.upper()]
        return text.lower() if text.isascii() else value
    return value


def _present(value: Any) -> bool:
    return value is not None and str(value).strip() != ""


def _legal_value(argument: str, value: Any) -> bool:
    """Validate shape without inventing a domain allowlist or a value."""
    if not _present(value):
        return False
    text = str(value).strip()
    if argument == "phone_last4":
        return bool(_PHONE_LAST4_RE.fullmatch(text))
    if argument == "service":
        return text in _SERVICE_VALUES
    # IDs and carrier codes are opaque values owned by the backing demo world;
    # only reject empty/whitespace values here.  In particular, do not select a
    # carrier constant when the model/user did not provide one.
    return bool(_TOKEN_RE.fullmatch(text))


@dataclass
class NormalizationAction:
    node_id: str
    argument: str
    source: str
    value_present: bool


@dataclass
class NormalizationResult:
    plan: R5PlanV1
    actions: list[NormalizationAction]
    unplannable_reason: str | None
    model_arg_conformant_nodes: int
    total_nodes: int


def _capability_contract_ok(node: R5PlanNodeV1) -> bool:
    """Whether the model's own node args already satisfy the contract."""
    contract = CAPABILITY_ARGS.get(node.capability_ref, {})
    provided = set(node.args) | set(node.bindings)
    return set(contract).issubset(provided) and provided.issubset(set(contract))


def _entity_value(entities: Mapping[str, Any], argument: str) -> Any:
    for key in ENTITY_ALIASES.get(argument, (argument,)):
        if key in entities and str(entities[key]).strip() != "":
            return _canonical_value(argument, str(entities[key]))
    return None


def _node_value(node: R5PlanNodeV1, argument: str) -> Any:
    value = node.args.get(argument)
    return _canonical_value(argument, str(value).strip()) if _present(value) else None


def _merge_explicit_value(node: R5PlanNodeV1, entities: Mapping[str, Any], argument: str) -> tuple[Any, str | None, str | None]:
    """Merge router entities and a node literal without guessing on conflict."""
    entity = _entity_value(entities, argument)
    node_value = _node_value(node, argument)
    if entity is not None and not _legal_value(argument, entity):
        return None, None, f"invalid_entity:{argument}"
    if node_value is not None and not _legal_value(argument, node_value):
        return None, None, f"invalid_node_arg:{argument}"
    if entity is not None and node_value is not None and entity != node_value:
        return None, None, f"conflicting_entity:{argument}"
    if entity is not None:
        return entity, ENTITY, None
    if node_value is not None:
        return node_value, NODE_ARG, None
    return None, None, None


def _binding_path(binding: R5BindingV1) -> str:
    path = str(binding.path or "")
    if path.startswith("$."):
        return path[2:]
    if path.startswith("$"):
        return path[1:]
    return path


def _valid_result_binding(
    node: R5PlanNodeV1,
    argument: str,
    binding: R5BindingV1,
    *,
    ancestors: Mapping[str, set[str]],
    node_by_id: Mapping[str, R5PlanNodeV1],
) -> bool:
    if binding.kind != "result" or _binding_path(binding) != argument:
        return False
    source_id = str(binding.source_node_id or "")
    source = node_by_id.get(source_id)
    return source_id in ancestors.get(node.node_id, set()) and source is not None and source.capability_ref in DERIVED_SOURCE_CAPABILITIES


def normalize_plan(
    plan: R5PlanV1,
    *,
    entities: Mapping[str, Any],
    trusted_context: Mapping[str, Any],
    user_text: str,
) -> NormalizationResult:
    """Fill deterministic arguments; fail closed to clarification if impossible."""
    if plan.needs_clarification:
        return NormalizationResult(plan=plan, actions=[], unplannable_reason=None, model_arg_conformant_nodes=0, total_nodes=len(plan.nodes))

    ancestors = plan.ancestors()
    node_by_id = {n.node_id: n for n in plan.nodes}
    actions: list[NormalizationAction] = []
    conformant = 0

    for node in plan.nodes:
        if _capability_contract_ok(node):
            conformant += 1

    def clarify(reason: str) -> NormalizationResult:
        return NormalizationResult(
            plan=R5PlanV1(nodes=(), edges=(), needs_clarification=True, clarification_reason=reason, business_goal=plan.business_goal),
            actions=actions,
            unplannable_reason=reason,
            model_arg_conformant_nodes=conformant,
            total_nodes=len(plan.nodes),
        )

    new_nodes: list[R5PlanNodeV1] = []
    for node in plan.nodes:
        contract = CAPABILITY_ARGS.get(node.capability_ref)
        if contract is None:
            # Unknown capability cannot be normalized; leave it for the validator.
            new_nodes.append(node)
            continue
        args: dict[str, Any] = {}
        bindings: dict[str, R5BindingV1] = {}
        for argument, source in contract.items():
            if source == ENTITY_OR_CONTEXT:
                value, value_source, merge_error = _merge_explicit_value(node, entities, argument)
                if merge_error:
                    return clarify(merge_error)
                if value is not None:
                    args[argument] = value
                    actions.append(NormalizationAction(node.node_id, argument, value_source or NODE_ARG, True))
                elif str(argument) in trusted_context and str(trusted_context[argument]).strip() != "":
                    existing = node.bindings.get(argument)
                    if existing is not None and (existing.kind != "context" or existing.context_key != str(argument)):
                        return clarify(f"invalid_binding:{node.node_id}:{argument}")
                    bindings[argument] = existing or R5BindingV1(kind="context", context_key=str(argument))
                    actions.append(NormalizationAction(node.node_id, argument, CONTEXT, True))
                else:
                    return clarify(f"missing_entity:{argument}")
            elif source == CONTEXT:
                if str(argument) not in trusted_context or str(trusted_context[argument]).strip() == "":
                    return clarify(f"missing_trusted_context:{argument}")
                existing = node.bindings.get(argument)
                if existing is not None and (existing.kind != "context" or existing.context_key != str(argument)):
                    return clarify(f"invalid_binding:{node.node_id}:{argument}")
                bindings[argument] = existing or R5BindingV1(kind="context", context_key=str(argument))
                actions.append(NormalizationAction(node.node_id, argument, CONTEXT, True))
            elif source == ENTITY:
                value, value_source, merge_error = _merge_explicit_value(node, entities, argument)
                if merge_error:
                    return clarify(merge_error)
                if value is None:
                    return clarify(f"missing_entity:{argument}")
                args[argument] = value
                actions.append(NormalizationAction(node.node_id, argument, value_source or NODE_ARG, True))
            elif source == REQUEST_TEXT:
                args[argument] = user_text
                actions.append(NormalizationAction(node.node_id, argument, REQUEST_TEXT, True))
            elif source == DERIVED:
                existing = node.bindings.get(argument)
                if existing is not None:
                    if not _valid_result_binding(node, argument, existing, ancestors=ancestors, node_by_id=node_by_id):
                        return clarify(f"invalid_binding:{node.node_id}:{argument}")
                    bindings[argument] = existing
                    actions.append(NormalizationAction(node.node_id, argument, DERIVED, True))
                else:
                    upstreams = sorted(
                        nid
                        for nid in ancestors.get(node.node_id, set())
                        if node_by_id.get(nid) is not None and node_by_id[nid].capability_ref in DERIVED_SOURCE_CAPABILITIES
                    )
                    if len(upstreams) > 1:
                        return clarify(f"ambiguous_derived_source:{node.node_id}:{argument}")
                    if upstreams:
                        upstream = upstreams[0]
                        bindings[argument] = R5BindingV1(kind="result", source_node_id=upstream, path=argument)
                        actions.append(NormalizationAction(node.node_id, argument, DERIVED, True))
                        continue
                    value, value_source, merge_error = _merge_explicit_value(node, entities, argument)
                    if merge_error:
                        return clarify(merge_error)
                    if value is None:
                        return clarify(f"missing_entity:{argument}")
                    args[argument] = value
                    actions.append(NormalizationAction(node.node_id, argument, value_source or NODE_ARG, True))
        new_nodes.append(node.model_copy(update={"args": args, "bindings": bindings}))

    normalized = R5PlanV1(
        schema_version=plan.schema_version,
        nodes=tuple(new_nodes),
        edges=plan.edges,
        needs_clarification=False,
        clarification_reason=None,
        business_goal=plan.business_goal,
    )
    return NormalizationResult(plan=normalized, actions=actions, unplannable_reason=None, model_arg_conformant_nodes=conformant, total_nodes=len(plan.nodes))


__all__ = ["CAPABILITY_ARGS", "NormalizationAction", "NormalizationResult", "normalize_plan"]
