"""R1 interactive runtime boundary.

The module is deliberately small and adapter-oriented.  It reuses the M2/M3
registry, AgentPort, executor and canonical plan objects while keeping model
providers injectable for deterministic tests.  No model provider is called by
importing this module.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Optional, Protocol, TypeVar
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .config import FIELD_SPECS, AgentConfig, compute_config_hash, load_config
from .domain.objects import PlanRevision, Run, sha256_json
from .m2_context import InvocationContext
from .m2_registry import Registry
from .m3_registry import M3Registry, build_m3_registry
from .m3_supervisor import CandidateIntent, M3Supervisor
from .agents.ports import OrderAgent
from .schemas import OrderQueryInput
from .storage.repository import SQLiteOrderRepository
from .storage.m2 import M2Repository
from .trace.events import TraceEvent, build_event, sensitive_surface_scan
from eval.harness.contracts import ExecutionMode, FailureScript, FreezeBundle, VersionTuple, WorldSnapshot
from eval.harness.trace_recorder import TraceRecorder, verify_bundle


class InteractiveRuntimeError(RuntimeError):
    """Typed fail-closed runtime error with a stable public code."""

    def __init__(self, code: str, message: str = "runtime failed") -> None:
        self.code = code
        super().__init__(message)


class UnsupportedExecutionMode(InteractiveRuntimeError):
    def __init__(self, mode: str) -> None:
        super().__init__("EXECUTION_MODE_NOT_ALLOWED", f"interactive mode is not allowed: {mode}")


class ModelConfigurationError(InteractiveRuntimeError):
    def __init__(self, message: str = "live model provider is not configured") -> None:
        super().__init__("MODEL_CONFIG_MISSING", message)


class ModelCallError(InteractiveRuntimeError):
    def __init__(self, code: str, message: str = "model call failed") -> None:
        super().__init__(code, message)


class GroundingError(InteractiveRuntimeError):
    def __init__(self, message: str = "response is not grounded in tool results") -> None:
        super().__init__("GROUNDING_FAILED", message)


class StructuredContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class IntentV1(StructuredContract):
    schema_version: str = Field(default="intent.v1", pattern=r"^intent\.v1$")
    intent: Literal["ORDER_READ"]
    order_id: Optional[str] = None
    phone_last4: Optional[str] = None
    confidence: float = Field(ge=0, le=1)
    needs_clarification: bool = False


class PlanCandidateV1(StructuredContract):
    schema_version: str = Field(default="plan-candidate.v1", pattern=r"^plan-candidate\.v1$")
    tool_ref: str = Field(min_length=1)
    order_id: Optional[str] = None
    phone_last4: Optional[str] = None


class ClaimV1(StructuredContract):
    source_field: str = Field(min_length=1)
    value: Any


class ResponseV1(StructuredContract):
    schema_version: str = Field(default="response.v1", pattern=r"^response\.v1$")
    message_code: Literal["ORDER_FACTS_V1"] = "ORDER_FACTS_V1"
    claim_fields: tuple[str, ...] = Field(min_length=1)
    evidence_ref: str = Field(pattern=r"^[0-9a-f]{64}$")


class ProviderProtocol(Protocol):
    def __call__(self, prompt: str, output_schema: type[Any]) -> Any: ...


@dataclass(frozen=True)
class ModelUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    available: bool = False


@dataclass(frozen=True)
class ModelResult:
    output: Any
    usage: ModelUsage = ModelUsage()
    model: str = "injected"


@dataclass(frozen=True)
class LLMRuntimeConfig:
    """Non-secret model configuration snapshot used by the live boundary."""

    agent_config: AgentConfig
    config_hash: str
    model: str
    base_url: str
    credential_present: bool = False
    timeout_seconds: float = 15.0
    max_retries: int = 1

    @classmethod
    def from_environment(cls, *, db_path: str | Path, env: Mapping[str, str] | None = None) -> "LLMRuntimeConfig":
        supplied = dict(env or {})
        # Consume process/supplied configuration field by field.  Secret
        # values are reduced to presence markers before load_config sees them.
        source: dict[str, str] = {}
        for spec in FIELD_SPECS:
            if spec.secret:
                value = supplied.get(spec.name) if spec.name in supplied else os.environ.get(spec.name)
                source[spec.name] = "configured" if bool(value) else ""
            else:
                value = supplied.get(spec.name) if spec.name in supplied else os.environ.get(spec.name)
                if value is not None:
                    source[spec.name] = str(value)
        source["ECOMMERCE_DB_PATH"] = str(db_path)
        # Runtime controls are intentionally kept outside AgentConfig and
        # therefore outside the user-facing env template.  They still obey
        # the same supplied-env-first, process-env-second rule and are part
        # of the non-secret runtime fingerprint below.
        for name in ("LLM_TIMEOUT_SECONDS", "LLM_MAX_RETRIES"):
            value = supplied.get(name) if name in supplied else os.environ.get(name)
            if value is not None:
                source[name] = str(value)
        cfg = load_config(source)
        non_secret = cfg.non_secret
        timeout = float(source.get("LLM_TIMEOUT_SECONDS", "15"))
        retries = int(source.get("LLM_MAX_RETRIES", "1"))
        if timeout <= 0 or timeout > 120 or retries < 0 or retries > 3:
            raise ModelConfigurationError("invalid timeout or retry configuration")
        config_hash = sha256_json({"agent_config_hash": compute_config_hash(cfg), "timeout_seconds": timeout, "max_retries": retries})
        return cls(cfg, config_hash, str(non_secret.get("OPENAI_MODEL", "gpt-4o")), str(non_secret.get("OPENAI_BASE_URL", "")), bool(cfg.secret_references.get("OPENAI_API_KEY", {}).get("present")), timeout, retries)

    def evidence(self) -> dict[str, Any]:
        return {
            "config_version": self.agent_config.config_version,
            "config_hash_sha256": self.config_hash,
            "model": self.model,
            "base_url": self.base_url,
            "credential_present": self.credential_present,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "secret_references": self.agent_config.secret_references,
        }


T = TypeVar("T")


class ModelClient:
    """Bounded, structured-output model adapter with injectable provider."""

    def __init__(self, config: LLMRuntimeConfig, provider: ProviderProtocol | None = None):
        self.config = config
        self.provider = provider

    def complete(self, prompt: str, output_schema: type[T]) -> tuple[T, ModelUsage, int]:
        if self.provider is None:
            raise ModelConfigurationError()
        last_error: Exception | None = None
        attempts = self.config.max_retries + 1
        for attempt in range(attempts):
            started = time.perf_counter()
            try:
                pool = ThreadPoolExecutor(max_workers=1)
                future = pool.submit(self.provider, prompt, output_schema)
                try:
                    raw = future.result(timeout=self.config.timeout_seconds)
                except FutureTimeoutError as exc:
                    future.cancel()
                    pool.shutdown(wait=False, cancel_futures=True)
                    raise ModelCallError("MODEL_TIMEOUT", "model call exceeded configured timeout") from exc
                else:
                    pool.shutdown(wait=True)
                latency = int((time.perf_counter() - started) * 1000)
                usage = ModelUsage()
                if isinstance(raw, ModelResult):
                    usage, raw = raw.usage, raw.output
                elif isinstance(raw, tuple) and len(raw) == 2 and isinstance(raw[1], Mapping):
                    raw, usage_map = raw
                    has_input = isinstance(usage_map.get("input_tokens"), (int, float))
                    has_output = isinstance(usage_map.get("output_tokens"), (int, float))
                    usage = ModelUsage(int(usage_map.get("input_tokens", 0)), int(usage_map.get("output_tokens", 0)), has_input and has_output)
                try:
                    return output_schema.model_validate(raw), usage, latency
                except ValidationError as exc:
                    raise ModelCallError("MODEL_SCHEMA_INVALID", "structured model output failed schema validation") from exc
            except ModelCallError as exc:
                last_error = exc
                if exc.code not in {"MODEL_TIMEOUT", "MODEL_RATE_LIMIT", "MODEL_PROVIDER_ERROR"}:
                    raise exc
            except Exception as exc:
                name = type(exc).__name__.lower()
                if "auth" in name or "permission" in name:
                    last_error = ModelCallError("MODEL_AUTH_ERROR", "model authentication failed")
                elif "rate" in name or "429" in str(exc):
                    last_error = ModelCallError("MODEL_RATE_LIMIT", "model rate limit reached")
                elif "timeout" in name or "timeout" in str(exc).lower():
                    last_error = ModelCallError("MODEL_TIMEOUT", "model call timed out")
                else:
                    last_error = ModelCallError("MODEL_PROVIDER_ERROR", "model provider failed")
                if last_error.code not in {"MODEL_TIMEOUT", "MODEL_RATE_LIMIT", "MODEL_PROVIDER_ERROR"}:
                    raise last_error
            if attempt + 1 < attempts:
                continue
        raise last_error or ModelCallError("MODEL_PROVIDER_ERROR")


class ChatOpenAIStructuredProvider:
    """Real provider bridge; construction is gated by credential presence."""

    def __init__(self, config: LLMRuntimeConfig):
        if not config.credential_present:
            raise ModelConfigurationError()
        self.config = config
        try:
            from langchain_openai import ChatOpenAI
            self.model = ChatOpenAI(model=config.model, base_url=config.base_url, timeout=config.timeout_seconds, max_retries=0)
        except Exception as exc:
            raise ModelConfigurationError("live ChatOpenAI provider could not be initialized") from exc

    def __call__(self, prompt: str, output_schema: type[Any]) -> Any:
        try:
            envelope = self.model.with_structured_output(output_schema, method="function_calling", include_raw=True).invoke(prompt)
            if isinstance(envelope, Mapping) and "parsed" in envelope:
                return ModelResult(output=envelope.get("parsed"), usage=_extract_usage(envelope.get("raw")), model=self.config.model)
            return ModelResult(output=envelope, usage=ModelUsage(), model=self.config.model)
        except Exception:
            # ModelClient classifies provider/auth/rate-limit/timeout errors;
            # raw provider text is deliberately not propagated.
            raise


def _extract_usage(message: Any) -> ModelUsage:
    """Extract only numeric usage counters from an AI message envelope."""
    candidates: list[Mapping[str, Any]] = []
    usage_metadata = getattr(message, "usage_metadata", None)
    response_metadata = getattr(message, "response_metadata", None)
    if isinstance(usage_metadata, Mapping):
        candidates.append(usage_metadata)
    if isinstance(response_metadata, Mapping):
        candidates.append(response_metadata)
        token_usage = response_metadata.get("token_usage")
        if isinstance(token_usage, Mapping):
            candidates.append(token_usage)
        usage = response_metadata.get("usage")
        if isinstance(usage, Mapping):
            candidates.append(usage)
    input_value: int | None = None
    output_value: int | None = None
    for candidate in candidates:
        if input_value is None:
            for key in ("input_tokens", "prompt_tokens", "input"):
                if isinstance(candidate.get(key), (int, float)):
                    input_value = int(candidate[key]); break
        if output_value is None:
            for key in ("output_tokens", "completion_tokens", "output"):
                if isinstance(candidate.get(key), (int, float)):
                    output_value = int(candidate[key]); break
    if input_value is None or output_value is None:
        return ModelUsage(available=False)
    return ModelUsage(input_tokens=input_value, output_tokens=output_value, available=True)


class DeterministicOrderProvider:
    """Small provider used only for explicit simulated mode and tests."""

    def __call__(self, prompt: str, output_schema: type[Any]) -> Any:
        order_id = next(iter(re.findall(r"\b\d{8,}\b", prompt)), None)
        phone = next(iter(re.findall(r"\b\d{4}\b", prompt)), None)
        name = output_schema.__name__
        if name == "IntentV1":
            return IntentV1(intent="ORDER_READ", order_id=order_id, phone_last4=phone, confidence=0.99, needs_clarification=not bool(order_id and phone))
        if name == "PlanCandidateV1":
            return PlanCandidateV1(tool_ref="order/get_info@v1")
        # The simulated provider only emits claims from the facts supplied by
        # the runtime.  This keeps simulated output subject to the same
        # grounding contract as a live provider.
        facts: dict[str, Any] = {}
        evidence_hash = ""
        marker = "facts_json="
        if marker in prompt:
            raw = prompt.split(marker, 1)[1]
            facts_text, _, evidence_hash = raw.partition(" evidence_hash=")
            try:
                parsed = json.loads(facts_text)
                if isinstance(parsed, dict):
                    facts = parsed
            except json.JSONDecodeError:
                facts = {}
        if facts:
            key = next(iter(facts))
            return ResponseV1(message_code="ORDER_FACTS_V1", claim_fields=(key,), evidence_ref=evidence_hash)
        return ResponseV1(message_code="ORDER_FACTS_V1", claim_fields=(), evidence_ref="0" * 64)


@dataclass(frozen=True)
class InteractiveResult:
    run_id: str
    session_id: str
    mode: ExecutionMode
    status: str
    answer: str
    code: str
    intent: IntentV1 | None
    plan: PlanRevision | None
    tool_result: Any
    trace_events: tuple[TraceEvent, ...]
    trace_checksum: str
    freeze_checksum: str
    trace_path: str | None = None
    freeze_bundle: FreezeBundle | None = None


class _TraceBuffer:
    def __init__(self, run_id: str, session_id: str, mode: ExecutionMode, *, model: str, prompt_version: str, repository: M2Repository, world: WorldSnapshot, version_tuple: VersionTuple, run_context: Mapping[str, Any], artifact_root: Path):
        self.run_id, self.session_id, self.mode = run_id, session_id, mode
        self.model, self.prompt_version = model, prompt_version
        self.repository, self.world, self.version_tuple = repository, world, version_tuple
        self.run_context = dict(run_context)
        self.artifact_root = artifact_root
        self.recorder = TraceRecorder(run_id=run_id, session_id=session_id, repository=repository, scene_clock=world.scene_clock, version_tuple=version_tuple, failure_script=FailureScript())
        self.plan: PlanRevision | None = None
        self.result_payload: dict[str, Any] | None = None
        self.result_code: str | None = None
        self.result_status: str = "FAILED"
        self.final_response: str = "请求无法安全完成。"
        self.evidence_refs: list[str] = []
        self.claim_refs: list[str] = []

    @property
    def events(self) -> tuple[TraceEvent, ...]:
        return self.recorder.events

    def record(self, event_type: str, payload: Mapping[str, Any], actor: str = "runtime") -> TraceEvent:
        return self.recorder.record(event_type, payload=dict(payload), actor=actor)

    def set_plan(self, plan: PlanRevision | None) -> None:
        self.plan = plan

    def set_result(self, value: Any, *, code: str | None, status: str) -> None:
        payload = getattr(value, "payload", None)
        self.result_payload = dict(payload or {}) if isinstance(payload, Mapping) else None
        self.result_code, self.result_status = code, status

    def set_evidence(self, *, evidence_refs: list[str] | tuple[str, ...] = (), claim_refs: list[str] | tuple[str, ...] = ()) -> None:
        self.evidence_refs = [str(item) for item in evidence_refs]
        self.claim_refs = [str(item) for item in claim_refs]

    def set_response(self, answer: str) -> None:
        self.final_response = str(answer)

    def model_called(self, schema: type[Any], prompt: str) -> None:
        self.record("MODEL_CALLED", {"mode": self.mode.value, "model": self.model, "prompt_version": self.prompt_version, "schema": schema.__name__, "prompt_hash": sha256_json(prompt)}, "agent")

    def model_returned(self, schema: type[Any], output: Any = None, *, usage: ModelUsage = ModelUsage(), latency_ms: int = 0, error_code: str | None = None) -> None:
        # The trace sensitivity scanner reserves the word ``token`` for raw
        # secret material.  Usage counters therefore use neutral keys while
        # remaining explicit, numeric and safe to aggregate.
        payload: dict[str, Any] = {"mode": self.mode.value, "model": self.model, "schema": schema.__name__, "output_hash": sha256_json(output.model_dump(mode="json") if hasattr(output, "model_dump") else output), "usage": {"input": usage.input_tokens, "output": usage.output_tokens, "available": usage.available}, "latency_ms": latency_ms}
        if error_code:
            payload["error_class"] = error_code
        self.record("MODEL_RETURNED", payload, "agent")

    def finalize(self) -> tuple[tuple[TraceEvent, ...], str, str, str | None, FreezeBundle]:
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        trace_path = self.artifact_root / "trace.jsonl"
        plan_rows = [self.plan.model_dump(mode="json")] if self.plan is not None else []
        result_rows = []
        if self.plan is not None:
            task = self.plan.tasks[0] if self.plan.tasks else None
            result_rows.append({"result_id": f"result_{self.run_id}", "schema_version": "m1.result.v1", "run_id": self.run_id, "plan_revision_id": self.plan.plan_revision_id, "task_id": task.task_id if task else "none", "attempt_id": f"attempt_{self.run_id}", "status": self.result_status, "output_contract": task.output_contract if task else "interactive.result.v1", "payload": self.result_payload, "payload_hash": sha256_json(self.result_payload), "business_code": self.result_code, "evidence_refs": self.evidence_refs, "claim_refs": self.claim_refs})
        manifest = self.recorder.drain_jsonl(trace_path)
        bundle = self.recorder.freeze_bundle(world_snapshot=self.world, final_response=self.final_response, final_fingerprint=self.world.snapshot_hash, failure_script=FailureScript(), run_context=self.run_context, version_tuple=self.version_tuple, plan_revisions=plan_rows, results=result_rows)
        verified = verify_bundle(bundle, repository=self.repository)
        if not verified.get("ok"):
            raise RuntimeError("interactive runtime produced an unverifiable FreezeBundle")
        return self.recorder.events, str(manifest["trace_checksum"]), bundle.checksum, str(trace_path), bundle


def _order_tool(db_path: str) -> Callable[..., dict[str, Any]]:
    def read(order_id: str, phone_last4: str) -> dict[str, Any]:
        try:
            request = OrderQueryInput(order_id=order_id, phone_last4=phone_last4)
        except ValidationError:
            return {"success": False, "code": "INVALID_PARAMS", "message": "invalid order query", "data": None, "user_hint": "请提供有效的订单号和手机号后四位。"}
        try:
            with SQLiteOrderRepository(db_path) as repository:
                data = repository.get_order_for_owner(request.order_id, request.phone_last4)
                if data is None:
                    exists = repository.get_order_by_id(request.order_id)
                    code = "PHONE_MISMATCH" if exists is not None else "ORDER_NOT_FOUND"
                    return {"success": False, "code": code, "message": "order is unavailable", "data": None, "user_hint": "手机号后四位不匹配。" if code == "PHONE_MISMATCH" else "未查询到该订单。"}
                # Do not expose the ownership proof in the model-facing result.
                # Ownership is proven inside the repository call.  The
                # business identifier and proof are intentionally omitted
                # from model-facing payloads and frozen artifacts.
                safe = {k: v for k, v in data.items() if k not in {"order_id", "phone_last4"}}
                safe["source"] = "ecommerce.db"
                safe["source_version"] = "sqlite.orders.v1"
                return {"success": True, "code": "OK", "message": "order found", "data": safe, "user_hint": ""}
        except FileNotFoundError:
            return {"success": False, "code": "CONFIG_MISSING", "message": "order database unavailable", "data": None, "user_hint": "系统暂时无法访问订单数据。"}
        except Exception:
            return {"success": False, "code": "INFRA_UNAVAILABLE", "message": "order database unavailable", "data": None, "user_hint": "系统暂时无法访问订单数据。"}
    return read


def _live_registry(db_path: str, mode: ExecutionMode) -> M3Registry:
    base = build_m3_registry()
    specs = []
    for ref in base.refs():
        spec = base.get(ref)
        if ref == "order/get_info@v1":
            spec = spec.model_copy(update={"implementation_mode": "LIVE" if mode is ExecutionMode.LIVE else "SIMULATED", "callable": _order_tool(db_path)})
        specs.append(spec)
    return M3Registry(canonical=Registry(specs))


def _r1_version_tuple(config_hash: str, model: str) -> VersionTuple:
    return VersionTuple(
        schema="r1.schema.v1", model=model, prompt="r1.prompt.v1", code="r1.code.v1",
        registry="r1.registry.v1", tool_impl="r1.tool.v1", config="r1.config." + _safe_artifact_label(config_hash),
        policy_catalog="r1.policy.v1", kb="r1.kb.v1", dataset="dev-order-live-v1",
        harness="r1.harness.v1", trace_schema="m1.trace.v1", evaluator="r1.evaluator.v1",
        simulator="r1.simulator.v1", world_template="r1.interactive.v1", seed=1,
    )


def _safe_artifact_label(run_id: str) -> str:
    """Map an execution id to an artifact directory without long digit runs."""
    digits = str.maketrans("0123456789", "abcdefghij")
    return str(run_id).translate(digits)


class InteractiveRuntime:
    """Explicit live/simulated interactive order-read runtime."""

    INTERACTIVE_MODES = frozenset({ExecutionMode.LIVE, ExecutionMode.SIMULATED})

    def __init__(self, *, db_path: str | Path, provider: ProviderProtocol | None = None, artifact_root: str | Path | None = None, timeout_seconds: float = 15.0, max_retries: int = 1):
        self.db_path = str(db_path)
        self.artifact_root = Path(artifact_root) if artifact_root is not None else None
        self.config = LLMRuntimeConfig.from_environment(db_path=self.db_path, env={"ECOMMERCE_DB_PATH": self.db_path, "LLM_TIMEOUT_SECONDS": str(timeout_seconds), "LLM_MAX_RETRIES": str(max_retries)})
        self.provider = provider

    def _mode(self, mode: ExecutionMode | str) -> ExecutionMode:
        try:
            selected = mode if isinstance(mode, ExecutionMode) else ExecutionMode(str(mode))
        except ValueError as exc:
            raise UnsupportedExecutionMode(str(mode)) from exc
        if selected not in self.INTERACTIVE_MODES:
            raise UnsupportedExecutionMode(selected.value)
        return selected

    def chat(self, *, user_id: str, message: str, session_id: str | None = None, mode: ExecutionMode | str = ExecutionMode.LIVE, run_id: str | None = None) -> InteractiveResult:
        selected = self._mode(mode)
        if not str(user_id or "").strip():
            raise InteractiveRuntimeError("AUTH_REQUIRED", "authenticated user is required")
        if not str(message or "").strip():
            raise InteractiveRuntimeError("INVALID_INPUT", "message must not be blank")
        sid = session_id or f"session_{uuid4().hex}"
        rid = run_id or f"r1_{uuid4().hex}"
        runtime_root = self.artifact_root or Path(tempfile.mkdtemp(prefix="r1-runtime-"))
        run_root = runtime_root / _safe_artifact_label(rid)
        run_root.mkdir(parents=True, exist_ok=True)
        repository = M2Repository(str(run_root / "runtime.db"))
        owner_ref = "owner_" + hashlib.sha256(("r1-owner-v1:" + str(user_id)).encode("utf-8")).hexdigest()[:32]
        world = WorldSnapshot(world_fixture_ref=f"interactive-{rid}", world_template_version="r1.interactive.v1", seed=1, scene_clock=datetime.now(timezone.utc), entities=[{"entity_type": "interactive", "entity_id": rid}])
        repository.create_session(sid, owner_ref)
        repository.create_run(Run(run_id=rid, session_id=sid, initial_world_hash=world.snapshot_hash))
        model_name = self.config.model if selected is ExecutionMode.LIVE else "deterministic-r1"
        versions = _r1_version_tuple(self.config.config_hash, model_name)
        trace = _TraceBuffer(rid, sid, selected, model=model_name, prompt_version="r1.prompt.v1", repository=repository, world=world, version_tuple=versions, run_context={"mode": selected.value, "model": model_name, "config_hash": self.config.config_hash, "owner_ref": owner_ref}, artifact_root=run_root)
        provider = self.provider
        if selected is ExecutionMode.SIMULATED and provider is None:
            provider = DeterministicOrderProvider()
        elif selected is ExecutionMode.LIVE and provider is None and self.config.credential_present:
            provider = ChatOpenAIStructuredProvider(self.config)
        client = ModelClient(self.config, provider=provider)
        intent: IntentV1 | None = None
        plan: PlanRevision | None = None
        tool_result: Any = None

        def finish(status: str, answer: str, code: str) -> InteractiveResult:
            trace.set_response(answer)
            events, checksum, freeze, path, bundle = trace.finalize()
            repository.close()
            return InteractiveResult(rid, sid, selected, status, answer, code, intent, plan, tool_result, events, checksum, freeze, path, bundle)

        try:
            prompt = f"intent_enum=ORDER_READ; canonical_intent_only=ORDER_READ; do_not_use_synonyms\nmessage: {message[:1000]}"
            trace.model_called(IntentV1, prompt)
            try:
                intent, usage, latency = client.complete(prompt, IntentV1)
                trace.model_returned(IntentV1, intent, usage=usage, latency_ms=latency)
            except InteractiveRuntimeError as exc:
                trace.model_returned(IntentV1, {}, error_code=exc.code)
                return finish("FAILED", "模型调用失败。", exc.code)
            if intent.needs_clarification or not intent.order_id or not intent.phone_last4:
                trace.record("RESULT_WRITTEN", {"status": "NEEDS_CLARIFICATION", "code": "CLARIFICATION_REQUIRED"})
                return finish("NEEDS_CLARIFICATION", "请提供订单号和手机号后四位。", "CLARIFICATION_REQUIRED")
            OrderQueryInput(order_id=intent.order_id, phone_last4=intent.phone_last4)
            plan_prompt = "plan_contract=single_allowed_tool; allowed_tool=order/get_info@v1; entity_fields_must_be_empty; do_not_copy_customer_identifiers"
            trace.model_called(PlanCandidateV1, plan_prompt)
            candidate, usage, latency = client.complete(plan_prompt, PlanCandidateV1)
            trace.model_returned(PlanCandidateV1, candidate, usage=usage, latency_ms=latency)
            if candidate.tool_ref != "order/get_info@v1" or (candidate.order_id is not None and candidate.order_id != intent.order_id) or (candidate.phone_last4 is not None and candidate.phone_last4 != intent.phone_last4):
                raise InteractiveRuntimeError("PLAN_CONTRACT_INVALID", "model plan does not match trusted intent")
            registry = _live_registry(self.db_path, selected)
            candidate_intent = CandidateIntent(intent="ORDER", confidence=intent.confidence, evidence_hash=sha256_json(intent.model_dump(mode="json")))
            supervisor = M3Supervisor(registry=registry, repository=repository)
            draft = supervisor.draft_operations(run_id=rid, candidate_intent=candidate_intent, operations=[candidate.tool_ref])
            plan = supervisor.activate_plan(draft)
            trace.set_plan(plan)
            trace.record("PLAN_CREATED", {"plan_revision_id": plan.plan_revision_id, "task_count": len(plan.tasks), "schema_version": plan.schema_version}, "agent")
            trace.record("PLAN_VALIDATED", {"plan_revision_id": plan.plan_revision_id, "status": plan.status.value}, "runtime")
            task = plan.tasks[0]
            context = InvocationContext(session_id=sid, user_id=str(user_id), run_id=rid, plan_revision_id=plan.plan_revision_id, task_id=task.task_id, attempt_id=f"attempt_{uuid4().hex}", agent_ref="order-agent@v1", auth_scope="order/read@v1", idempotency_key=f"idemp_{uuid4().hex}", deadline=datetime.now(timezone.utc) + timedelta(seconds=10), config_version=self.config.agent_config.config_version, registry_version="r1.registry.v1", dataset_version="dev-order-live-v1", trace_id=f"trace_{uuid4().hex}")
            args = {"order_id": intent.order_id, "phone_last4": intent.phone_last4}
            trace.record("TOOL_CALLED", {"tool_ref": "order/get_info@v1", "args_hash": sha256_json(args), "mode": selected.value}, "tool")
            port_result = OrderAgent(registry=registry).invoke(args, context=context)
            tool_result = port_result
            trace.record("TOOL_RETURNED", {"tool_ref": "order/get_info@v1", "ok": port_result.ok, "error_code": port_result.error_code, "payload_hash": port_result.payload_hash}, "tool")
            if not port_result.ok:
                answer = "手机号后四位不匹配。" if port_result.error_code in {"PHONE_MISMATCH", "AUTH_IDENTITY_MISMATCH"} else ("未查询到该订单。" if port_result.error_code == "ORDER_NOT_FOUND" else "系统暂时无法访问订单数据。")
                trace.record("RESULT_WRITTEN", {"status": "FAILED", "code": port_result.error_code or "TOOL_FAILED"})
                trace.set_result(port_result, code=port_result.error_code or "TOOL_FAILED", status="FAILED")
                return finish("FAILED", answer, port_result.error_code or "TOOL_FAILED")
            payload = dict(port_result.payload or {})
            allowed = {str(k): v for k, v in payload.items() if k not in {"source", "source_version", "phone_last4", "address", "recipient_name"}}
            evidence_hash = port_result.payload_hash
            facts_json = json.dumps(allowed, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            synthesis_prompt = f"response_contract=select_fields_only; message_code=ORDER_FACTS_V1; source_field_allowlist={json.dumps(sorted(allowed), ensure_ascii=False)}; claim_fields_must_be_nonempty_unique_allowlisted_strings; evidence_ref_must_equal={evidence_hash}; do_not_return_fact_values_or_answer; facts_json={facts_json} evidence_hash={evidence_hash}"
            trace.model_called(ResponseV1, synthesis_prompt)
            response, usage, latency = client.complete(synthesis_prompt, ResponseV1)
            trace.model_returned(ResponseV1, response, usage=usage, latency_ms=latency)
            if response.message_code != "ORDER_FACTS_V1" or not response.claim_fields or len(set(response.claim_fields)) != len(response.claim_fields) or response.evidence_ref != evidence_hash:
                raise GroundingError()
            if any(field not in allowed for field in response.claim_fields):
                raise GroundingError()
            claims = tuple(ClaimV1(source_field=field, value=allowed[field]) for field in response.claim_fields)
            claim_refs = [sha256_json(claim.model_dump(mode="json")) for claim in claims]
            trace.set_result(port_result, code="OK", status="SUCCEEDED")
            trace.set_evidence(evidence_refs=[evidence_hash], claim_refs=claim_refs)
            trace.record("RESULT_WRITTEN", {"status": "SUCCEEDED", "code": "OK", "payload_hash": evidence_hash, "evidence_refs": [evidence_hash], "claim_count": len(claims)})
            rendered = "订单信息已核验：" + "；".join(f"{claim.source_field}={claim.value}" for claim in claims) + "。"
            return finish("SUCCEEDED", rendered, "OK")
        except InteractiveRuntimeError as exc:
            trace.record("ERROR", {"code": exc.code, "error_class": type(exc).__name__})
            if tool_result is not None and plan is not None:
                trace.set_result(tool_result, code=exc.code, status="FAILED")
            return finish("FAILED", "请求无法安全完成。", exc.code)
        except ValidationError:
            trace.record("ERROR", {"code": "SCHEMA_INVALID", "error_class": "ValidationError"})
            if tool_result is not None and plan is not None:
                trace.set_result(tool_result, code="SCHEMA_INVALID", status="FAILED")
            return finish("FAILED", "请求格式无法验证。", "SCHEMA_INVALID")


__all__ = ["ClaimV1", "DeterministicOrderProvider", "ExecutionMode", "GroundingError", "InteractiveResult", "InteractiveRuntime", "InteractiveRuntimeError", "IntentV1", "LLMRuntimeConfig", "ModelCallError", "ModelClient", "ModelConfigurationError", "ModelResult", "ModelUsage", "PlanCandidateV1", "ResponseV1", "UnsupportedExecutionMode"]
