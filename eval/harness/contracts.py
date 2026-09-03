"""Stable M4 harness contracts.

These models separate executable runtime input from evaluator-only gold.  The
models are frozen and reject unknown fields so that accidental contract drift
is visible at the boundary.
"""
from __future__ import annotations

import hashlib
import re
import json
from pathlib import Path
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agent.domain.objects import canonical_json, sha256_json


class ExecutionMode(str, Enum):
    LIVE = "live"
    SIMULATED = "simulated"
    FAULT = "fault"
    REPLAY = "replay"
    RE_EXECUTE = "re_execute"


class VersionTuple(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(min_length=1, alias="schema", serialization_alias="schema")
    model: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    code: str = Field(min_length=1)
    registry: str = Field(min_length=1)
    tool_impl: str = Field(min_length=1)
    config: str = Field(min_length=1)
    policy_catalog: str = Field(min_length=1)
    kb: str = Field(min_length=1)
    dataset: str = Field(min_length=1)
    harness: str = Field(min_length=1)
    trace_schema: str = Field(min_length=1)
    evaluator: str = Field(min_length=1)
    simulator: str = Field(min_length=1)
    world_template: str = Field(min_length=1)
    seed: int
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    @property
    def dataset_version(self) -> str:
        return self.dataset

    @property
    def registry_version(self) -> str:
        return self.registry

    @property
    def policy_version(self) -> str:
        return self.policy_catalog

    @property
    def simulator_version(self) -> str:
        return self.simulator

    @property
    def world_template_version(self) -> str:
        return self.world_template

    def as_tuple(self) -> tuple[Any, ...]:
        return tuple(self.model_dump(mode="python", by_alias=True).values())

    @property
    def fingerprint(self) -> str:
        return sha256_json(self.model_dump(mode="json", by_alias=True))


class Scenario(BaseModel):
    """Only executable fields; evaluator gold is deliberately forbidden."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    turns: List[str] = Field(min_length=1)
    intent_label: Optional[str] = None
    world_fixture_ref: str = Field(min_length=1)
    split: str = Field(min_length=1)
    version_tuple: VersionTuple
    failure_script: "FailureScript" = Field(default_factory=lambda: FailureScript())
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)

    @field_validator("turns")
    @classmethod
    def nonempty_turns(cls, value: List[str]) -> List[str]:
        if any(not str(item).strip() for item in value):
            raise ValueError("turns must contain non-empty strings")
        return value


class FailureTrigger(BaseModel):
    """One deterministic logical/physical fault trigger.

    ``count`` counts matching invocations, not retries.  The controller never
    changes logical call numbering; it only returns an injected outcome.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    tool_ref: str = Field(min_length=1)
    action: Literal["RETURN_ERROR", "BLOCK_TASK"]
    error_code: Optional[str] = None
    logical_call: Optional[int] = Field(default=None, gt=0)
    physical_attempt: Optional[int] = Field(default=None, gt=0)
    layer: Optional[Literal["infra", "contract", "auth", "business", "data_quality"]] = None
    count: int = Field(default=1, gt=0)
    priority: int = 0
    exhaustion: Literal["ignore", "repeat", "error"] = "ignore"
    trigger_id: str = "trigger-v1"

    @model_validator(mode="after")
    def validate_action(self) -> "FailureTrigger":
        if self.action == "RETURN_ERROR" and not self.error_code:
            raise ValueError("RETURN_ERROR requires error_code")
        if self.action == "BLOCK_TASK" and self.error_code is not None:
            raise ValueError("BLOCK_TASK does not accept error_code")
        if self.logical_call is None and self.physical_attempt is None and self.layer is None:
            # An unscoped trigger remains valid and applies to the first count
            # matching calls; this is useful for a compact fixture.
            return self
        return self


class FailureScript(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str = "failure-script.v1"
    triggers: List[FailureTrigger] = Field(default_factory=list)

    @model_validator(mode="after")
    def sorted_priority_contract(self) -> "FailureScript":
        if len({t.trigger_id for t in self.triggers}) != len(self.triggers):
            raise ValueError("failure trigger IDs must be unique")
        return self

    @classmethod
    def from_mapping(cls, value: Any) -> "FailureScript":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            raise TypeError("failure_script must be an object")
        # M3's compact single-trigger fixture remains accepted as an input
        # adapter, while the canonical stored form is always triggers[].
        if "triggers" in value:
            return cls.model_validate(value)
        raw = dict(value)
        raw.setdefault("trigger_id", "trigger-0")
        raw["tool_ref"] = raw.pop("tool_ref", raw.pop("target_tool", "*"))
        return cls(version=str(raw.pop("version", "failure-script.v1")), triggers=[FailureTrigger.model_validate(raw)])


class WorldSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    world_fixture_ref: str = Field(min_length=1)
    world_template_version: str = Field(min_length=1)
    seed: int
    scene_clock: datetime
    entities: List[Dict[str, Any]] = Field(default_factory=list)

    @field_validator("scene_clock")
    @classmethod
    def utc_clock(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("scene_clock must include timezone")
        return value.astimezone(timezone.utc)

    @property
    def canonical_entities(self) -> List[Dict[str, Any]]:
        return sorted(
            [dict(item) for item in self.entities],
            key=lambda item: (
                str(item.get("entity_type", item.get("type", ""))),
                str(item.get("entity_id", item.get("id", ""))),
                canonical_json(item),
            ),
        )

    def canonical_projection(self) -> Dict[str, Any]:
        return {
            "world_fixture_ref": self.world_fixture_ref,
            "world_template_version": self.world_template_version,
            "seed": self.seed,
            "scene_clock": self.scene_clock.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
            "entities": self.canonical_entities,
        }

    @property
    def snapshot_hash(self) -> str:
        return sha256_json(self.canonical_projection())


class ArtifactRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(min_length=1)
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    event_count: Optional[int] = Field(default=None, ge=0)
    initial_world_hash: Optional[str] = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    canonical_json: Optional[Dict[str, Any]] = None


class FreezeLock(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    owner: str = Field(min_length=1)
    acquired_at: datetime
    immutable: Literal[True] = True


class FreezeBundle(BaseModel):
    """Immutable gold-free execution evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "m4.freeze.v1"
    bundle_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    trace: ArtifactRef
    plan_revisions: List[ArtifactRef] = Field(default_factory=list)
    results: List[ArtifactRef] = Field(default_factory=list)
    final_response: ArtifactRef
    world_snapshot: ArtifactRef
    final_world_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    failure_script: Optional[ArtifactRef] = None
    run_context: ArtifactRef
    freeze_lock: FreezeLock
    version_tuple: VersionTuple
    frozen_at: datetime

    @staticmethod
    def _contains_forbidden(value: Any, path: str = "", skip: set[str] | None = None) -> str | None:
        skip = skip or set()
        if isinstance(value, dict):
            for key, child in value.items():
                key_text = str(key).lower()
                # ``version_tuple.evaluator`` is a required runtime version
                # coordinate, not evaluator gold.  Every other evaluator key
                # remains forbidden, including run_context/artifact payloads.
                version_tuple_evaluator = key_text == "evaluator" and path == "version_tuple"
                if key_text in {"gold", "expected", "rubric"} and key_text not in skip:
                    return f"{path}.{key}" if path else str(key)
                if key_text == "evaluator" and not version_tuple_evaluator and key_text not in skip:
                    return f"{path}.{key}" if path else str(key)
                found = FreezeBundle._contains_forbidden(child, f"{path}.{key}" if path else str(key), skip)
                if found:
                    return found
        elif isinstance(value, list):
            for index, child in enumerate(value):
                found = FreezeBundle._contains_forbidden(child, f"{path}[{index}]", skip)
                if found:
                    return found
        return None

    @model_validator(mode="after")
    def no_gold_and_valid_lock(self) -> "FreezeBundle":
        forbidden = self._contains_forbidden(self.model_dump(mode="python"))
        if forbidden:
            raise ValueError(f"gold/evaluator fields cannot enter FreezeBundle: {forbidden}")
        sensitive = self._contains_sensitive(self.model_dump(mode="python"))
        if sensitive:
            raise ValueError(f"raw PII/sensitive field cannot enter FreezeBundle: {sensitive}")
        refs = [self.trace, *self.plan_revisions, *self.results, self.final_response, self.world_snapshot, self.run_context]
        if self.failure_script is not None:
            refs.append(self.failure_script)
        for ref in refs:
            content = ref.canonical_json
            if content is None and Path(ref.path).is_file():
                try:
                    if Path(ref.path).suffix.lower() == ".jsonl":
                        content = [json.loads(line) for line in Path(ref.path).read_text(encoding="utf-8").splitlines() if line.strip()]
                    else:
                        content = json.loads(Path(ref.path).read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    content = None
            if content is not None:
                forbidden = self._contains_forbidden(content)
                if forbidden:
                    raise ValueError(f"gold/evaluator fields cannot enter FreezeBundle artifact: {forbidden}")
                sensitive = self._contains_sensitive(content)
                if sensitive:
                    raise ValueError(f"raw PII/sensitive field cannot enter FreezeBundle artifact: {sensitive}")
        if not self.freeze_lock.immutable:
            raise ValueError("freeze lock must be immutable")
        return self

    @staticmethod
    def _contains_sensitive(value: Any, path: str = "") -> str | None:
        pattern = re.compile(r"(?:token|secret|password|phone|address|payment|raw_payload)", re.I)
        if isinstance(value, dict):
            for key, child in value.items():
                text = str(key).lower()
                safe_metadata = text.endswith(("_hash", "_ref", "_purpose", "_status", "_state", "_type", "_version", "_required"))
                if pattern.search(text) and not safe_metadata and text not in {"token_id_hash", "payload_hash", "snapshot_hash"}:
                    return f"{path}.{key}" if path else str(key)
                found = FreezeBundle._contains_sensitive(child, f"{path}.{key}" if path else str(key))
                if found:
                    return found
        elif isinstance(value, list):
            for index, child in enumerate(value):
                found = FreezeBundle._contains_sensitive(child, f"{path}[{index}]")
                if found:
                    return found
        elif isinstance(value, str):
            # Freeze artifacts may contain synthetic references, hashes and
            # redaction markers, but never phone/card-like raw values hidden
            # inside otherwise innocuous text fields such as a user turn.
            leaf = path.rsplit(".", 1)[-1].lower()
            # UUIDs/event references are identifiers, not payment/phone
            # values.  A UUID can contain a 15--19 digit run after the
            # separators are removed, so never apply value-level PII checks
            # to identifier/hash/timestamp fields.  Sensitive *field names*
            # are still checked above (e.g. ``phone`` or ``payment``).
            if leaf.endswith(("_id", "_ids", "_ref", "_refs", "_hash", "_checksum", "_fingerprint")) or leaf in {
                "id", "ids", "trace", "timestamp", "occurred_at", "scene_clock",
            }:
                return None
            if leaf in {"checksum", "manifest_hash"}:
                return None
            compact = re.sub(r"[\s-]", "", value)
            if re.search(r"(?<!\d)1[3-9]\d{9}(?!\d)", compact):
                return path or "<value>"
            if re.search(r"(?<!\d)\d{15,19}(?!\d)", compact):
                return path or "<value>"
        return None

    @staticmethod
    def compute_lock(**parts: Any) -> str:
        return sha256_json(parts)

    @property
    def checksum(self) -> str:
        return sha256_json(self.model_dump(mode="json"))


class EvaluationInput(BaseModel):
    """Evaluator binding kept outside the runtime FreezeBundle."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "m4.evaluation-input.v1"
    bundle: Optional[FreezeBundle] = None
    bundle_id: str = Field(min_length=1)
    dataset_manifest: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    gold_version: str = Field(min_length=1)
    gold: Dict[str, Any] = Field(default_factory=dict)
    evaluator_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def assert_isolation(self) -> "EvaluationInput":
        if self.bundle is not None and self.bundle_id != self.bundle.bundle_id:
            raise ValueError("EvaluationInput bundle_id does not match FreezeBundle")
        if self.bundle is not None:
            forbidden = FreezeBundle._contains_forbidden(self.bundle.model_dump(mode="python"))
            if forbidden:
                raise ValueError(f"gold must remain outside FreezeBundle: {forbidden}")
        return self


Scenario.model_rebuild()

__all__ = [
    "EvaluationInput", "ExecutionMode", "FailureScript", "FailureTrigger",
    "ArtifactRef", "FreezeBundle", "FreezeLock", "Scenario", "VersionTuple", "WorldSnapshot",
]
