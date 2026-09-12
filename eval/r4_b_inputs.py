"""Gold-free R4-B development input builder.

The builder emits executable user/world/fault inputs only.  It never embeds
expected agents, capabilities, terminal states, error codes, answers or gold.
Callers may write the returned inputs to a file, but importing this module has
no filesystem side effect.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent.domain.objects import sha256_json


R4_B_INPUT_SCHEMA_VERSION = "r4-b.input.v1"
R4_B_BUILDER_VERSION = "r4-b.input-builder.v2"
# Inputs deliberately carry only a coarse workload family.  The exact task
# graph is execution-spec/gold material and must never be recoverable from
# the JSONL development inputs.
NORMAL_TOPOLOGY_FAMILIES = (
    "single_domain",
    "cross_domain_parallel",
    "cross_domain_dependency",
)
# Backward-compatible import name; it now exposes only coarse labels.
NORMAL_TOPOLOGIES = NORMAL_TOPOLOGY_FAMILIES
_EXACT_TOPOLOGY_TOKENS = frozenset(
    {
        "order_only",
        "logistics_only",
        "policy_only",
        "order->logistics",
        "order+policy",
        "order->logistics+policy",
    }
)
FAULT_FAMILIES = (
    "duplicate",
    "dependency_pending",
    "recoverable_timeout",
    "recoverable_lost",
    "non_retryable",
    "late",
    "schema_version",
    "semantic_wrong",
    "partial_branch",
)


class R4BWorldV1(BaseModel):
    """Synthetic case world; the evaluator materializes it in a temp DB."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    order_id: str = Field(min_length=1)
    phone_last4: str = Field(pattern=r"^\d{4}$")
    carrier_code: str = Field(min_length=1)
    tracking_no: str = Field(min_length=1)
    policy_query: str = Field(min_length=1)
    logistics_quality: str = "FRESH"


class R4BInputV1(BaseModel):
    """One gold-free, executable R4-B input."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = R4_B_INPUT_SCHEMA_VERSION
    case_id: str = Field(min_length=1)
    family_id: str = Field(min_length=1)
    topology_family: str = Field(min_length=1)
    failure_family: str = Field(min_length=1)
    source_family: str = Field(min_length=1)
    annotation_provenance: str = Field(min_length=1)
    user_text: str = Field(min_length=1)
    world: R4BWorldV1
    failure_script: dict[str, Any] = Field(default_factory=dict)
    actions: tuple[str, ...] = ()

    @model_validator(mode="after")
    def input_is_gold_free(self) -> "R4BInputV1":
        if self.schema_version != R4_B_INPUT_SCHEMA_VERSION:
            raise ValueError("R4-B input schema version mismatch")
        if self.failure_family == "normal" and self.failure_script:
            raise ValueError("normal inputs cannot carry a failure script")
        if self.failure_family != "normal" and not self.actions:
            raise ValueError("fault inputs must declare executable actions")
        if self.topology_family not in NORMAL_TOPOLOGY_FAMILIES:
            raise ValueError("unknown coarse topology family")
        forbidden = {"expected", "gold", "terminal", "error_code", "answer"}
        encoded = self.model_dump(mode="python")
        if _contains_forbidden_key(encoded, forbidden):
            raise ValueError("gold/expectation fields are not allowed in R4-B inputs")
        serialized = json.dumps(encoded, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if any(token in serialized for token in _EXACT_TOPOLOGY_TOKENS):
            raise ValueError("exact execution topology is not allowed in R4-B inputs")
        return self


def _contains_forbidden_key(value: Any, forbidden: set[str]) -> bool:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in forbidden:
                return True
            if _contains_forbidden_key(child, forbidden):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_contains_forbidden_key(item, forbidden) for item in value)
    return False


def _world(index: int, *, policy_query: str = "无需查询政策") -> R4BWorldV1:
    return R4BWorldV1(
        order_id=f"R4B-ORD-{index:03d}",
        phone_last4=f"{index % 10000:04d}",
        carrier_code=f"carrier-{index:03d}",
        tracking_no=f"R4B-TRK-{index:03d}",
        policy_query=policy_query,
    )


# These labels are builder-local only. They are never serialized into an
# input row; the row retains only its coarse topology family.
_NORMAL_SEMANTIC_COMBINATIONS = (
    "order_only",
    "logistics_only",
    "policy_only",
    "order->logistics",
    "order+policy",
    "order->logistics+policy",
)
_POLICY_QUESTIONS = {
    "policy_only": ("退货期限是多久？", "退款到账一般需要多长时间？", "什么情况下可以申请换货？"),
    "order+policy": ("退货期限是多久？", "退款到账一般需要多长时间？", "什么情况下可以申请换货？"),
    "order->logistics+policy": ("退货期限是多久？", "退款到账一般需要多长时间？", "什么情况下可以申请换货？"),
}


def _normal_text(combination: str, world: R4BWorldV1, variant: int) -> str:
    order_id = world.order_id
    phone = world.phone_last4
    carrier = world.carrier_code
    tracking = world.tracking_no
    policy = world.policy_query
    templates = {
        "order_only": (
            f"我想确认订单 {order_id} 的付款状态和订单详情，手机号后四位是 {phone}。",
            f"请帮我核对一下订单 {order_id} 目前是否已支付。",
            f"订单 {order_id} 的详情可以帮我看一下吗？我提供的手机号后四位是 {phone}。",
        ),
        "logistics_only": (
            f"麻烦查一下运单 {tracking} 现在物流配送到哪一步了。",
            f"我想了解包裹 {tracking} 的最新物流进度，承运商是 {carrier}。",
            f"请帮我追踪 {carrier} 的这件包裹（运单号 {tracking}），看看当前配送状态。",
        ),
        "policy_only": (
            f"请问{policy}我想提前了解相关售后规则。",
            f"我想先了解一下：{policy}",
            f"麻烦说明一下{policy}方便我安排后续处理。",
        ),
        "order->logistics": (
            f"订单 {order_id} 已经付款了，手机号后四位是 {phone}，请先确认订单，再根据订单中的物流信息查看包裹目前配送到哪里。",
            f"请先帮我确认订单 {order_id}，再使用订单记录中的物流信息查询配送进度。",
            f"我想核对订单 {order_id}，手机号后四位为 {phone}，随后查询订单对应的物流信息和配送进展。",
        ),
        "order+policy": (
            f"订单 {order_id} 的状态请帮我确认一下；另外，{policy}",
            f"我想查看订单 {order_id}，手机号后四位是 {phone}，并了解{policy}",
            f"请核对订单 {order_id}，手机号后四位为 {phone}，顺便告诉我{policy}",
        ),
        "order->logistics+policy": (
            f"订单 {order_id} 先帮我确认一下，手机号后四位是 {phone}，随后根据订单记录中的物流信息查看配送进度；另外，{policy}",
            f"我想先核对订单 {order_id}，再使用订单里的物流信息了解配送到哪了，也请说明{policy}",
            f"订单 {order_id} 已经下单，手机号后四位为 {phone}，请先根据订单记录中的物流信息跟进物流，然后告诉我{policy}",
        ),
    }
    return templates[combination][variant]


def _fault_case(index: int, family: str, variant: int) -> R4BInputV1:
    world = _world(100 + index)
    topology = (
        "cross_domain_dependency"
        if family in {"dependency_pending", "recoverable_timeout", "recoverable_lost"}
        else "cross_domain_parallel"
        if family == "partial_branch"
        else "single_domain"
    )
    # Fault targets and exact error kinds belong to the independent execution
    # spec.  The input remains executable metadata only and carries no error
    # oracle.
    script: dict[str, Any] = {}
    actions = {
        "duplicate": ("replay_request",),
        "dependency_pending": ("exercise_ordering",),
        "recoverable_timeout": ("inject_recoverable_fault",),
        "recoverable_lost": ("inject_recoverable_fault",),
        "non_retryable": ("inject_non_retryable_fault",),
        "late": ("freeze_and_replay",),
        "schema_version": ("mutate_message_schema",),
        "semantic_wrong": ("inject_semantic_mismatch",),
        "partial_branch": ("preserve_independent_branch",),
    }[family]
    return R4BInputV1(
        case_id=f"fault-{index:02d}",
        family_id=f"fault-{family}",
        topology_family=topology,
        failure_family=family,
        source_family=f"synthetic-{family}-world-v1",
        annotation_provenance=R4_B_BUILDER_VERSION,
        user_text=f"Please process the request for scenario {index:02d} using the available records.",
        world=world,
        failure_script=script,
        actions=actions,
    )


def build_r4_b_dev_inputs() -> tuple[R4BInputV1, ...]:
    """Build exactly 36 unique development inputs (18 normal, 18 fault)."""

    cases: list[R4BInputV1] = []
    ordinal = 1
    coarse_by_combination = {
        "order_only": "single_domain",
        "logistics_only": "single_domain",
        "policy_only": "single_domain",
        "order->logistics": "cross_domain_dependency",
        "order+policy": "cross_domain_parallel",
        "order->logistics+policy": "cross_domain_dependency",
    }
    for combination in _NORMAL_SEMANTIC_COMBINATIONS:
        topology = coarse_by_combination[combination]
        for variant in range(3):
            world = _world(ordinal)
            policy_questions = _POLICY_QUESTIONS.get(combination)
            if policy_questions is not None:
                world = _world(ordinal, policy_query=policy_questions[variant])
            cases.append(
                R4BInputV1(
                    case_id=f"normal-{ordinal:02d}",
                    family_id=f"normal-{topology}",
                    topology_family=topology,
                    failure_family="normal",
                    source_family="synthetic-normal-world-v1",
                    annotation_provenance=R4_B_BUILDER_VERSION,
                    user_text=_normal_text(combination, world, variant),
                    world=world,
                )
            )
            ordinal += 1
    fault_ordinal = 1
    for family in FAULT_FAMILIES:
        for variant in range(1, 3):
            cases.append(_fault_case(fault_ordinal, family, variant))
            fault_ordinal += 1
    if len(cases) != 36 or len({item.case_id for item in cases}) != 36:
        raise AssertionError("R4-B builder must produce 36 unique cases")
    if sum(item.failure_family == "normal" for item in cases) != 18:
        raise AssertionError("R4-B builder must produce 18 normal cases")
    return tuple(cases)


def input_manifest(cases: Sequence[R4BInputV1]) -> dict[str, Any]:
    """Return manifest/overlap evidence without consulting evaluator gold."""

    case_ids = [item.case_id for item in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("duplicate R4-B case_id")
    normal_ids = [item.case_id for item in cases if item.failure_family == "normal"]
    fault_ids = [item.case_id for item in cases if item.failure_family != "normal"]
    canonical = [item.model_dump(mode="json") for item in cases]
    encoded = "\n".join(json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")) for item in canonical) + "\n"
    return {
        "schema_version": "r4-b.manifest.v1",
        "builder_version": R4_B_BUILDER_VERSION,
        "split": "dev",
        "case_count": len(cases),
        "unique_case_N": len(set(case_ids)),
        "normal_count": len(normal_ids),
        "protocol_fault_count": len(fault_ids),
        "case_ids_sha256": sha256_json(case_ids),
        "inputs_sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        "family_counts": {
            family: sum(item.family_id == family for item in cases)
            for family in sorted({item.family_id for item in cases})
        },
        "topology_family_counts": {
            topology: sum(item.topology_family == topology for item in cases)
            for topology in NORMAL_TOPOLOGY_FAMILIES
        },
        # Compatibility alias; values remain coarse and contain no exact
        # execution topology.
        "topology_counts": {
            topology: sum(item.topology_family == topology for item in cases)
            for topology in NORMAL_TOPOLOGY_FAMILIES
        },
        "overlap": {"reference_case_N": 0, "intersection_N": 0, "intersection_rate": 0.0},
    }


def write_r4_b_dev_inputs(path: str | Path, cases: Iterable[R4BInputV1] | None = None) -> dict[str, Any]:
    """Write only executable inputs; no gold or threshold file is created."""

    values = tuple(cases or build_r4_b_dev_inputs())
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "\n".join(json.dumps(item.model_dump(mode="json"), ensure_ascii=False, sort_keys=True) for item in values) + "\n",
        encoding="utf-8",
    )
    return input_manifest(values)


def load_r4_b_inputs(path: str | Path) -> tuple[R4BInputV1, ...]:
    """Load and validate a gold-free input JSONL file."""

    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    cases = tuple(R4BInputV1.model_validate(row) for row in rows)
    if len({item.case_id for item in cases}) != len(cases):
        raise ValueError("duplicate R4-B case_id")
    return cases


__all__ = [
    "FAULT_FAMILIES",
    "NORMAL_TOPOLOGIES",
    "NORMAL_TOPOLOGY_FAMILIES",
    "R4BWorldV1",
    "R4BInputV1",
    "R4_B_BUILDER_VERSION",
    "R4_B_INPUT_SCHEMA_VERSION",
    "build_r4_b_dev_inputs",
    "input_manifest",
    "load_r4_b_inputs",
    "write_r4_b_dev_inputs",
]
