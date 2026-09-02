"""M3 candidate routing and Supervisor-owned plan lifecycle."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from .domain.objects import InputBinding, PlanRevision, PlanStatus, Task
from .domain.plan_validator import PlanLimits, PlanValidationError, PlanValidator
from .m3_registry import M3Registry, build_m3_registry
from .agents.manifest import AgentManifest, build_m3_manifests, validate_manifests


class CandidateIntent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    intent: str
    confidence: float = Field(ge=0, le=1)
    evidence_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source: str = "intent_router"


class PlanDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    draft_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    candidate_intent: CandidateIntent
    tasks: tuple[Task, ...]
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class IntentRouter:
    """Pure candidate classifier. It never activates a plan or calls a tool."""

    def classify(self, user_text: str) -> CandidateIntent:
        from .domain.objects import sha256_json

        text = str(user_text or "")
        rules = (
            ("ESCALATION", ("转人工", "人工客服", "投诉", "举报", "高风险", "审核"), 0.98),
            ("AFTERSALES", ("退款", "退货", "换货", "售后"), 0.95),
            ("LOGISTICS", ("物流", "快递", "签收", "派件", "运单"), 0.92),
            ("POLICY", ("规则", "政策", "运费", "七天无理由"), 0.91),
            ("ORDER", ("订单", "订单号", "支付", "发货"), 0.90),
            ("PRODUCT", ("商品", "产品", "库存", "价格"), 0.88),
        )
        hits = [(intent, confidence) for intent, terms, confidence in rules if any(term in text for term in terms)]
        if not hits:
            intent, confidence = "UNKNOWN", 0.2
        elif len(hits) > 1:
            intent, confidence = "MIXED", max(c for _, c in hits)
        else:
            intent, confidence = hits[0]
        return CandidateIntent(intent=intent, confidence=confidence, evidence_hash=sha256_json({"text": text, "intent": intent}))


class M3Supervisor:
    """The only M3 shared-plan writer; draft creation is pure until activation."""

    def __init__(self, *, registry: M3Registry | None = None, manifests: tuple[AgentManifest, ...] | None = None, repository: Any = None, limits: PlanLimits = PlanLimits()):
        self.registry = registry or build_m3_registry()
        self.manifests = manifests or build_m3_manifests()
        validate_manifests(self.manifests, self.registry)
        self.repository = repository
        self.validator = PlanValidator(agents=[m.agent_ref for m in self.manifests] + ["supervisor@v1"], capabilities=[s.capability_ref for s in (self.registry.get(ref) for ref in self.registry.refs())], limits=limits)

    def draft_plan(self, *, run_id: str, candidate_intent: CandidateIntent, plan_revision_id: Optional[str] = None) -> PlanDraft:
        revision_id = plan_revision_id or f"plan_{uuid4().hex}"
        intents = [candidate_intent.intent] if candidate_intent.intent != "MIXED" else ["ORDER", "POLICY"]
        specs = {
            "PRODUCT": ("product-agent@v1", "product/read@v1", "product.result.v1"),
            "ORDER": ("order-agent@v1", "order/read@v1", "order.result.v1"),
            "LOGISTICS": ("logistics-agent@v1", "logistics/read@v1", "logistics.result.v1"),
            "POLICY": ("policy-agent@v1", "policy/read@v1", "policy.result.v1"),
            "AFTERSALES": ("aftersales-agent@v1", "aftersales/read@v1", "aftersales.result.v1"),
            "ESCALATION": ("aftersales-agent@v1", "aftersales/write@v1", "handoff.result.v1"),
        }
        tasks = []
        for idx, intent in enumerate(intents, start=1):
            if intent not in specs:
                continue
            agent_ref, capability, contract = specs[intent]
            tasks.append(Task(task_id=f"task_{idx}_{revision_id}", plan_revision_id=revision_id, agent_ref=agent_ref, capability_refs=[capability], output_contract=contract, failure_strategy="FAIL_RUN", side_effect="WRITE" if intent == "ESCALATION" else "READ_ONLY", timeout_ms=5000))
        return PlanDraft(draft_id=f"draft_{uuid4().hex}", run_id=run_id, candidate_intent=candidate_intent, tasks=tuple(tasks))

    def draft_operations(self, *, run_id: str, candidate_intent: CandidateIntent, operations: list[str], plan_revision_id: Optional[str] = None) -> PlanDraft:
        """Create a plan from typed operations derived from the request.

        Evaluator expectations are intentionally not accepted by this API.  A
        caller must provide canonical ToolRefs selected by its router/business
        rules, after which activation validates the complete DAG.
        """
        revision_id = plan_revision_id or f"plan_{uuid4().hex}"
        tasks: list[Task] = []
        aftersales_chain = any(ref == "aftersales/create@v1" or ref == "aftersales/query@v1" for ref in operations)
        chain_tools = {"order/get_info@v1", "logistics/query@v1", "aftersales/create@v1", "aftersales/query@v1"}
        for index, tool_ref in enumerate(dict.fromkeys(operations), start=1):
            spec = self.registry.get(tool_ref)
            agent_ref = "supervisor@v1" if spec.owner == "supervisor" else f"{spec.owner}@v1"
            # Independent mixed-domain reads are runnable in parallel. Only
            # the order/logistics/after-sales business chain carries edges.
            prior_chain = next((t for t in reversed(tasks) if t.capability_refs[0].startswith(("order/", "logistics/", "aftersales/"))), None)
            deps = [prior_chain.task_id] if prior_chain and aftersales_chain and tool_ref in chain_tools else []
            bindings = []
            if deps:
                bindings.append(InputBinding(name=f"dependency_{index}", kind="result", source_task_id=deps[-1], path="payload", required=False))
            # Independent mixed-domain reads are partial-safe: a failed branch
            # is retained while sibling facts can still be joined.  Dependent
            # business chains retain FAIL_RUN and therefore block downstream
            # work on missing facts.
            strategy = "CONTINUE_PARTIAL" if candidate_intent.intent == "MIXED" and not deps and spec.side_effect == "READ_ONLY" else "FAIL_RUN"
            tasks.append(Task(task_id=f"task_{index}_{revision_id}", plan_revision_id=revision_id, agent_ref=agent_ref, capability_refs=[spec.capability_ref], depends_on=deps, input_bindings=bindings, output_contract=spec.result_schema, failure_strategy=strategy, side_effect=spec.side_effect, timeout_ms=spec.timeout_ms))
        return PlanDraft(draft_id=f"draft_{uuid4().hex}", run_id=run_id, candidate_intent=candidate_intent, tasks=tuple(tasks))

    def activate_plan(self, draft: PlanDraft, *, created_by: str = "supervisor", revision_reason: str = "initial", supersedes_plan_revision_id: Optional[str] = None, version: int = 1) -> PlanRevision:
        revision = PlanRevision(plan_revision_id=draft.tasks[0].plan_revision_id if draft.tasks else f"plan_{uuid4().hex}", run_id=draft.run_id, created_by=created_by, revision_reason=revision_reason, supersedes_plan_revision_id=supersedes_plan_revision_id, version=version, status=PlanStatus.ACTIVE, tasks=list(draft.tasks))
        self.validator.validate(revision)
        if self.repository is not None:
            self.repository.create_plan_revision(revision)
        return revision

    def replan(self, previous: PlanRevision, *, tasks: list[Task], reason: str = "local_replan") -> PlanRevision:
        new_id = f"plan_{uuid4().hex}"
        # Rebind the complete DAG into the new revision.  Carrying old task
        # IDs in depends_on or result bindings would make the new revision
        # refer to immutable state owned by its superseded plan.
        id_map = {task.task_id: f"{task.task_id}_r{previous.version + 1}" for task in tasks}
        rebound = []
        for task in tasks:
            bindings = [binding.model_copy(update={"source_task_id": id_map.get(binding.source_task_id, binding.source_task_id)}) if binding.kind == "result" else binding for binding in task.input_bindings]
            rebound.append(task.model_copy(update={"plan_revision_id": new_id, "task_id": id_map[task.task_id], "depends_on": [id_map.get(dep, dep) for dep in task.depends_on], "input_bindings": bindings}))
        revision = PlanRevision(plan_revision_id=new_id, run_id=previous.run_id, created_by="supervisor", revision_reason=reason, supersedes_plan_revision_id=previous.plan_revision_id, version=previous.version + 1, status=PlanStatus.ACTIVE, tasks=rebound)
        self.validator.validate(revision)
        if self.repository is not None:
            self.repository.create_plan_revision(revision)
        return revision


__all__ = ["CandidateIntent", "IntentRouter", "M3Supervisor", "PlanDraft"]
