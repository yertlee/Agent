"""Stable public runtime port used by service and demo callers.

Only this port invokes the existing M3 runtime.  The HTTP layer is a thin
adapter and cannot append TraceEvents or mutate domain tables itself.
"""
from __future__ import annotations

import json
import hashlib
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eval.harness import FailureScript, TraceRecorder, VersionTuple, WorldStateBuilder, verify_bundle
from eval.harness.contracts import FreezeBundle
from eval.harness.contracts import ExecutionMode
from .interactive_runtime import InteractiveResult, InteractiveRuntime
from .m3_runtime import M3ScenarioRunner
from .storage.m2 import M2Repository


@dataclass(frozen=True)
class RuntimeChatResult:
    run_id: str
    session_id: str
    answer: str
    status: str
    bundle_path: str


def _versions() -> VersionTuple:
    return VersionTuple(schema="m5.schema.v1", model="m5.runtime.v1", prompt="m5.prompt.v1", code="m5.code.v1",
                        registry="m5.registry.v1", tool_impl="m5.tool.v1", config="m5.config.v1",
                        policy_catalog="m5.policy.v1", kb="m5.kb.v1", dataset="m5.demo.v1", harness="m5.harness.v1",
                        trace_schema="m5.trace.v1", evaluator="m5.evaluator.v1", simulator="m5.simulator.v1",
                        world_template="m5.world.v1", seed=5005)


def owner_ref(credential: str) -> str:
    """Return a stable, irreversible reference for an auth credential."""
    value = str(credential or "").strip()
    if not value:
        raise PermissionError("authenticated user is required")
    return "owner_" + hashlib.sha256(("m5-owner-v1:" + value).encode("utf-8")).hexdigest()[:32]


def _safe_run_id(value: str) -> str:
    if not value or len(value) > 128 or not __import__("re").fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError("invalid run_id")
    return value


def _public_value(value: Any) -> Any:
    """Drop sensitive keys from runtime Result payloads before freezing."""
    if isinstance(value, dict):
        import re
        result = {}
        for key, child in value.items():
            if re.search(r"(?:token|secret|password|phone|address|payment|card|raw_payload)", str(key), re.I):
                continue
            result[str(key)] = _public_value(child)
        return result
    if isinstance(value, list):
        return [_public_value(item) for item in value]
    return value


class RuntimePort:
    """The sole public write entry point for a chat run."""
    def __init__(self, *, artifact_root: str | Path | None = None, runner: M3ScenarioRunner | None = None, interactive_runtime: InteractiveRuntime | None = None):
        self.artifact_root = Path(artifact_root or os.getenv("M5_ARTIFACT_ROOT", "runtime/m5"))
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.runner = runner or M3ScenarioRunner(db_dir=self.artifact_root / "db")
        self.interactive_runtime = interactive_runtime

    def interactive_chat(self, *, session_id: str | None, user_id: str, message: str,
                         mode: ExecutionMode | str = ExecutionMode.LIVE,
                         run_id: str | None = None) -> InteractiveResult:
        """Explicit R1 live/simulated boundary; no mode fallback is permitted."""
        runtime = self.interactive_runtime
        if runtime is None:
            db_path = os.getenv("ECOMMERCE_DB_PATH") or str(Path("ecommerce.db"))
            runtime = InteractiveRuntime(db_path=db_path, artifact_root=self.artifact_root / "interactive")
        authenticated_owner = owner_ref(user_id)
        inherited_session = session_id
        if run_id:
            _safe_run_id(run_id)
            metadata_path = self._metadata(run_id)
            if not metadata_path.is_file():
                raise KeyError(run_id)
            prior = json.loads(metadata_path.read_text(encoding="utf-8"))
            if prior.get("owner_ref") != authenticated_owner:
                raise PermissionError("run ownership does not match authenticated principal")
            inherited_session = str(prior["session_id"])
            # A continuation is always a fresh immutable runtime run.  The
            # supplied id is only an ownership/session reference.
            run_id = None
        result = runtime.chat(session_id=inherited_session, user_id=user_id, message=message, mode=mode, run_id=run_id)
        bundle = result.freeze_bundle
        if bundle is None:
            raise RuntimeError("interactive runtime did not return a frozen bundle")
        verified = verify_bundle(bundle)
        if not verified.get("ok"):
            raise RuntimeError("interactive runtime produced an unverifiable freeze bundle")
        root = self.artifact_root / _safe_run_id(result.run_id)
        root.mkdir(parents=True, exist_ok=True)
        bundle_path = root / "bundle.json"
        fd, temp_name = __import__("tempfile").mkstemp(prefix=".bundle.", suffix=".tmp", dir=str(root), text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(bundle.model_dump_json(indent=2))
                handle.flush(); os.fsync(handle.fileno())
            os.replace(temp_name, bundle_path)
        except Exception:
            try: os.unlink(temp_name)
            except OSError: pass
            raise
        metadata_path = self._metadata(result.run_id)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {"owner_ref": authenticated_owner, "session_id": result.session_id,
                    "created_at": datetime.now(timezone.utc).isoformat(), "bundle": str(bundle_path)}
        fd, temp_name = __import__("tempfile").mkstemp(prefix=".ownership.", suffix=".tmp", dir=str(metadata_path.parent), text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(metadata, handle, sort_keys=True)
                handle.flush(); os.fsync(handle.fileno())
            os.replace(temp_name, metadata_path)
        except Exception:
            try: os.unlink(temp_name)
            except OSError: pass
            raise
        return result

    def _metadata(self, run_id: str) -> Path:
        return self.artifact_root / _safe_run_id(run_id) / "ownership.json"

    def chat(self, *, session_id: str | None, user_id: str, message: str, run_id: str | None = None,
             scenario_id: str | None = None, world_fixture: dict[str, Any] | None = None,
             failure_script: dict[str, Any] | None = None, version_tuple: VersionTuple | None = None) -> RuntimeChatResult:
        credential = str(user_id or "")
        authenticated_owner = owner_ref(credential)
        if not str(message or "").strip():
            raise ValueError("message must not be empty")
        if run_id:
            _safe_run_id(run_id)
            metadata = self._metadata(run_id)
            if not metadata.is_file():
                raise KeyError(run_id)
            prior = json.loads(metadata.read_text(encoding="utf-8"))
            if prior.get("owner_ref") != authenticated_owner:
                raise PermissionError("run ownership does not match authenticated principal")
            # A continuation gets a new runtime execution identity while the
            # caller keeps the original run immutable; this avoids appending
            # after a freeze and preserves the canonical event sequence.
            session_id = str(prior["session_id"])
        session_id = session_id or f"session_{uuid.uuid4().hex}"
        scenario_id = scenario_id or f"api-{uuid.uuid4().hex[:12]}"
        fixture = dict(world_fixture or {"world_fixture_ref": f"world-{scenario_id}", "world_template_version": "m5.world.v1",
                   "seed": 5005, "scene_clock": "2026-01-01T00:00:00Z",
                   "entities": [{"entity_type": "api", "entity_id": scenario_id}]})
        case = {"scenario_id": scenario_id, "category": "mixed", "turns": [str(message)],
                "world_fixture_ref": fixture["world_fixture_ref"], "world_snapshot": fixture,
                "expected_intent": None, "expected_tool_path": [], "expected_business_code": None,
                "failure_script": dict(failure_script or {})}
        run = self.runner.run(case)
        root = self.artifact_root / run.run_id
        root.mkdir(parents=True, exist_ok=True)
        # Runtime owns recorder creation, draining and freezing.  The service
        # only gets the immutable bundle after this operation completes.
        repo = M2Repository(run.db_path)
        try:
            repo_session = repo.conn.execute("SELECT session_id FROM runs WHERE run_id=?", (run.run_id,)).fetchone()
            actual_session = str(repo_session[0]) if repo_session else session_id
            world = WorldStateBuilder().build(fixture)
            frozen_versions = version_tuple or _versions()
            frozen_script = FailureScript.from_mapping(failure_script) if failure_script else FailureScript()
            recorder = TraceRecorder(run_id=run.run_id, session_id=actual_session, repository=repo,
                                     scene_clock=world.scene_clock, version_tuple=frozen_versions,
                                     failure_script=frozen_script)
            recorder.drain_jsonl(root / "trace.jsonl")
            plan_rows = []
            for row in repo.conn.execute("SELECT plan_revision_id,run_id,schema_version,created_by,revision_reason,supersedes_plan_revision_id,version,status,tasks_json,created_at,updated_at FROM plan_revisions WHERE run_id=? ORDER BY version", (run.run_id,)).fetchall():
                item = dict(row)
                item["tasks"] = json.loads(item.pop("tasks_json"))
                plan_rows.append(item)
            result_rows = []
            for row in repo.conn.execute("SELECT result_id,schema_version,run_id,plan_revision_id,task_id,attempt_id,status,output_contract,payload_json,payload_hash,business_code,error_ref,evidence_refs_json,claim_refs_json,usage_json,created_at,updated_at FROM results WHERE run_id=? ORDER BY created_at,result_id", (run.run_id,)).fetchall():
                item = dict(row)
                for field in ("payload_json", "evidence_refs_json", "claim_refs_json", "usage_json"):
                    raw = item.pop(field)
                    # Usage may contain ``input_tokens`` counters; the
                    # canonical FreezeBundle sensitivity scanner reserves the
                    # word token for secret material, so counters are omitted
                    # from the public evidence projection.
                    if field != "usage_json":
                        item[field.removesuffix("_json")] = _public_value(json.loads(raw)) if raw else None
                result_rows.append(item)
            bundle = recorder.freeze_bundle(world_snapshot=world, final_response=run.business_code or "completed",
                                            final_fingerprint=run.terminal_fingerprint or run.world_fingerprint,
                                            failure_script=frozen_script,
                                            run_context={"scenario_id": scenario_id, "mode": "simulated"},
                                            version_tuple=frozen_versions, results=result_rows,
                                            plan_revisions=plan_rows)
            verified = verify_bundle(bundle, repository=repo)
            if not verified.get("ok"):
                raise RuntimeError("runtime produced an unverifiable freeze bundle")
            bundle_path = root / "bundle.json"
            bundle_path.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")
            metadata_path = self._metadata(run.run_id)
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            metadata = {"owner_ref": authenticated_owner, "session_id": actual_session,
                        "created_at": datetime.now(timezone.utc).isoformat(), "bundle": str(bundle_path)}
            fd, temp_name = __import__("tempfile").mkstemp(prefix=".ownership.", suffix=".tmp", dir=str(metadata_path.parent), text=True)
            try:
                with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                    json.dump(metadata, handle, sort_keys=True)
                    handle.flush(); os.fsync(handle.fileno())
                os.replace(temp_name, metadata_path)
            except Exception:
                try: os.unlink(temp_name)
                except OSError: pass
                raise
            return RuntimeChatResult(run_id=run.run_id, session_id=actual_session,
                                     answer=run.business_code or "completed", status=run.observed_terminal_class,
                                     bundle_path=str(root / "bundle.json"))
        finally:
            repo.close()

    def load_owned_bundle(self, *, run_id: str, user_id: str) -> FreezeBundle:
        metadata = self._metadata(run_id)
        if not metadata.is_file():
            raise KeyError(run_id)
        details = json.loads(metadata.read_text(encoding="utf-8"))
        if details.get("owner_ref") != owner_ref(user_id):
            raise PermissionError("run ownership does not match authenticated principal")
        bundle_path = Path(str(details.get("bundle", "")))
        if not bundle_path.is_file():
            raise ValueError("frozen bundle is unavailable")
        return FreezeBundle.model_validate_json(bundle_path.read_text(encoding="utf-8"))


__all__ = ["RuntimeChatResult", "RuntimePort", "owner_ref"]
