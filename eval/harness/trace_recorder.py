"""Canonical TraceEvent recorder and crash-safe outbox JSONL drain."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from agent.domain.objects import canonical_json, sha256_json
from agent.m2_context import InvocationContext
from agent.storage.m2 import M2Repository
from agent.trace.events import TraceEvent, build_event, sensitive_surface_scan

from .contracts import ArtifactRef, FailureScript, FreezeBundle, FreezeLock, VersionTuple, WorldSnapshot


class TraceDrainManifest(dict):
    """JSON-compatible delivery manifest, retained as a dict for callers."""


class TraceRecorder:
    def __init__(self, *, run_id: str, session_id: str, repository: M2Repository | None = None,
                 scene_clock: datetime | None = None, version_tuple: VersionTuple | None = None,
                 failure_script: FailureScript | None = None):
        self.run_id = run_id
        self.session_id = session_id
        self.repository = repository
        self.scene_clock = scene_clock
        self.version_tuple = version_tuple or VersionTuple(schema="m4.schema.v1", model="m4.model.v1", prompt="m4.prompt.v1",
            code="m4.code.v1", registry="m4.registry.v1", tool_impl="m4.tool.v1", config="m4.config.v1",
            policy_catalog="m4.policy.v1", kb="m4.kb.v1", dataset="m4.dataset.v1", harness="m4.harness.v1",
            trace_schema="m4.trace.v1", evaluator="m4.evaluator.v1", simulator="m4.simulator.v1",
            world_template="world-template.v1", seed=0)
        self.failure_script = failure_script or FailureScript()
        self._events: list[TraceEvent] = []
        self._delivered: set[str] = set()
        self.late_events: list[dict[str, str]] = []
        self._frozen = False
        self._manifest: TraceDrainManifest | None = None
        self._sealed = False
        if self.repository is not None:
            rows = self.repository.conn.execute("SELECT envelope_json,delivered_at FROM trace_outbox WHERE run_id=? ORDER BY seq_no", (self.run_id,)).fetchall()
            self._events = [TraceEvent.model_validate(json.loads(str(row[0]))) for row in rows]
            self._delivered = {str(json.loads(str(row[0]))["trace_id"]) for row in rows if row[1] is not None}
            sealed = self.repository.conn.execute("SELECT 1 FROM trace_manifests WHERE run_id=?", (self.run_id,)).fetchone()
            self._sealed = bool(sealed)

    @property
    def events(self) -> tuple[TraceEvent, ...]:
        return tuple(self._events)

    def _late_audit(self, event_id: str, reason: str) -> None:
        self.late_events.append({"event_id": event_id, "reason": reason})
        if self.repository is None:
            return
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        self.repository.conn.execute("BEGIN IMMEDIATE")
        try:
            self.repository.conn.execute(
                "INSERT INTO late_event_audit(audit_id,run_id,event_id,reason,received_at,created_at,updated_at) VALUES (?,?,?,?,?,?,?)",
                (f"late-{uuid.uuid4().hex}", self.run_id, event_id, reason, now, now, now),
            )
            self.repository.conn.commit()
        except Exception:
            self.repository.conn.rollback()
            raise

    def record(self, event_type: str, *, payload: Mapping[str, Any] | None = None,
               plan_revision_id: str | None = None, task_id: str | None = None,
               attempt_id: str | None = None, parent_event_id: str | None = None,
               actor: str = "recorder") -> TraceEvent:
        seq = (self._events[-1].seq_no + 1) if self._events else 1
        if self.repository is not None:
            row = self.repository.conn.execute("SELECT next_seq_no FROM runs WHERE run_id=?", (self.run_id,)).fetchone()
            if row:
                seq = int(row[0])
        event_payload = dict(payload or {})
        sensitive = FreezeBundle._contains_sensitive(event_payload)
        if sensitive:
            raise ValueError(f"raw PII/sensitive field in trace payload: {sensitive}")
        event = build_event(run_id=self.run_id, session_id=self.session_id, event_type=event_type,
            seq_no=seq, payload=event_payload, plan_revision_id=plan_revision_id, task_id=task_id,
            attempt_id=attempt_id, parent_event_id=parent_event_id, actor=actor, scene_clock=self.scene_clock)
        if self._frozen or self._sealed:
            self._late_audit(event.trace_id, "TRACE_FROZEN_OR_MANIFEST_SEALED")
            raise RuntimeError("trace manifest is sealed; event recorded only in late_event_audit")
        if self.repository is not None:
            self.repository.append_m2_event(event)
        self._events.append(event)
        return event

    def pending(self) -> int:
        if self.repository is not None:
            row = self.repository.conn.execute("SELECT COUNT(*) FROM trace_outbox WHERE run_id=? AND delivered_at IS NULL", (self.run_id,)).fetchone()
            return int(row[0])
        return sum(1 for event in self._events if event.trace_id not in self._delivered)

    def drain_jsonl(self, output_path: str | Path, *, snapshot_hash: str = "") -> TraceDrainManifest:
        if self._sealed and self._manifest is not None:
            return self._manifest
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if self.repository is not None:
            rows = self.repository.conn.execute("SELECT event_id,envelope_json FROM trace_outbox WHERE run_id=? ORDER BY seq_no", (self.run_id,)).fetchall()
            records = [(str(row[0]), str(row[1])) for row in rows]
        else:
            records = [(event.trace_id, canonical_json(event)) for event in self._events]
        # The temp file is in the destination directory, so os.replace is an
        # atomic same-filesystem commit after flush+fsync.
        fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent), text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                for _event_id, envelope in records:
                    handle.write(envelope)
                    handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, target)
        except Exception:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
        payload = target.read_bytes()
        if self.repository is not None:
            for event_id, _ in records:
                self.repository.conn.execute("UPDATE trace_outbox SET delivered_at=?,updated_at=? WHERE event_id=? AND delivered_at IS NULL", (datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), event_id))
            now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            if snapshot_hash:
                self.repository.conn.execute(
                    "INSERT INTO trace_manifests(manifest_id,run_id,trace_checksum,event_count,version_tuple_json,snapshot_hash,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(run_id) DO UPDATE SET trace_checksum=excluded.trace_checksum,event_count=excluded.event_count,version_tuple_json=excluded.version_tuple_json,snapshot_hash=excluded.snapshot_hash,updated_at=excluded.updated_at",
                    (f"trace-manifest-{uuid.uuid4().hex}", self.run_id, hashlib.sha256(payload).hexdigest(), len(records), canonical_json(self.version_tuple.model_dump(mode="json", by_alias=True)), snapshot_hash, now, now),
                )
            self.repository.conn.commit()
        self._delivered.update(event_id for event_id, _ in records)
        manifest = TraceDrainManifest(path=str(target), trace_checksum=hashlib.sha256(payload).hexdigest(), event_count=len(records), run_id=self.run_id, version_tuple=self.version_tuple.model_dump(mode="json", by_alias=True), sealed=bool(snapshot_hash))
        self._manifest = manifest
        return manifest

    def freeze_bundle(self, *, world_snapshot: WorldSnapshot, final_response: str,
                      final_fingerprint: str, failure_script: FailureScript,
                      run_context: Mapping[str, Any], version_tuple: VersionTuple,
                      plan_revisions: list[Mapping[str, Any]] | None = None,
                      results: list[Mapping[str, Any]] | None = None) -> FreezeBundle:
        if self.pending() != 0:
            raise RuntimeError("cannot freeze while outbox has undelivered events")
        if self._manifest is None:
            raise RuntimeError("trace must be drained before freeze")
        if self._sealed:
            raise RuntimeError("trace manifest is already sealed")
        if version_tuple.fingerprint != self.version_tuple.fingerprint:
            raise ValueError("freeze version_tuple must match recorder version_tuple")
        if self.pending() != 0:
            raise RuntimeError("cannot freeze while outbox has undelivered events")
        # RUN_FROZEN is the final canonical event and is included in the
        # second drain before the manifest is sealed.
        self.record("RUN_FROZEN", payload={"freeze": "pending_zero", "manifest_hash": self._manifest["trace_checksum"], "checksum": self._manifest["trace_checksum"]}, actor="recorder")
        self.drain_jsonl(self._manifest["path"], snapshot_hash=world_snapshot.snapshot_hash)
        root = Path(self._manifest["path"]).parent
        root.mkdir(parents=True, exist_ok=True)

        def write_artifact(name: str, value: Any) -> ArtifactRef:
            path = root / name
            encoded = canonical_json(value).encode("utf-8")
            forbidden = FreezeBundle._contains_forbidden(value)
            if forbidden:
                raise ValueError(f"gold/evaluator fields cannot be frozen: {forbidden}")
            sensitive = FreezeBundle._contains_sensitive(value)
            if sensitive:
                raise ValueError(f"raw PII/sensitive field cannot be frozen: {sensitive}")
            fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(root))
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temp_name, path)
            except Exception:
                try:
                    os.unlink(temp_name)
                except OSError:
                    pass
                raise
            return ArtifactRef(path=str(path), checksum=hashlib.sha256(encoded).hexdigest())

        plans = [write_artifact(f"plan-{idx}.json", dict(item)) for idx, item in enumerate(plan_revisions or [])]
        result_refs = [write_artifact(f"result-{idx}.json", dict(item)) for idx, item in enumerate(results or [])]
        response_ref = write_artifact("final_response.json", {"response": final_response})
        world_ref = write_artifact("world_snapshot.json", world_snapshot.canonical_projection())
        world_ref = world_ref.model_copy(update={"initial_world_hash": world_snapshot.snapshot_hash})
        failure_ref = write_artifact("failure_script.json", failure_script.model_dump(mode="json")) if failure_script.triggers else None
        context_ref = write_artifact("run_context.json", dict(run_context))
        lock = FreezeLock(owner="harness", acquired_at=datetime.now(timezone.utc), immutable=True)
        bundle = FreezeBundle(bundle_id=f"bundle-{uuid.uuid4().hex}", run_id=self.run_id,
            trace=ArtifactRef(path=str(self._manifest["path"]), checksum=str(self._manifest["trace_checksum"]), event_count=int(self._manifest["event_count"])),
            plan_revisions=plans, results=result_refs, final_response=response_ref, world_snapshot=world_ref,
            final_world_fingerprint=final_fingerprint, failure_script=failure_ref, run_context=context_ref,
            freeze_lock=lock, version_tuple=version_tuple, frozen_at=datetime.now(timezone.utc))
        self._frozen = True
        self._sealed = True
        return bundle


def verify_bundle(bundle: FreezeBundle, *, repository: M2Repository | None = None) -> dict[str, Any]:
    """Verify path checksums and terminal freeze invariants without mutation."""
    refs = [bundle.trace, *bundle.plan_revisions, *bundle.results, bundle.final_response, bundle.world_snapshot, bundle.run_context]
    if bundle.failure_script is not None:
        refs.append(bundle.failure_script)
    mismatches: list[str] = []
    world_hash_matches = False
    for ref in refs:
        path = Path(ref.path)
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != ref.checksum:
            mismatches.append(ref.path)
        elif path.is_file():
            try:
                if path.suffix.lower() == ".jsonl":
                    value = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
                else:
                    value = json.loads(path.read_text(encoding="utf-8"))
                if value is not None and (FreezeBundle._contains_forbidden(value) or FreezeBundle._contains_sensitive(value)):
                    mismatches.append(f"unsafe:{ref.path}")
            except (OSError, ValueError):
                mismatches.append(f"unreadable:{ref.path}")
    try:
        world_value = json.loads(Path(bundle.world_snapshot.path).read_text(encoding="utf-8"))
        world_hash_matches = bool(bundle.world_snapshot.initial_world_hash) and bundle.world_snapshot.initial_world_hash == sha256_json(world_value)
        if not world_hash_matches:
            mismatches.append("world_snapshot:initial_world_hash")
    except (OSError, TypeError, ValueError):
        mismatches.append("world_snapshot:unreadable")
    lines = Path(bundle.trace.path).read_text(encoding="utf-8").splitlines() if Path(bundle.trace.path).is_file() else []
    event_rows: list[dict[str, Any]] = []
    try:
        event_rows = [json.loads(line) for line in lines if line.strip()]
    except (TypeError, ValueError):
        # The checksum mismatch/unreadable artifact is reported below; the
        # verifier itself must remain a total read-only check for corrupt JSON.
        event_rows = []
    frozen_last = bool(event_rows) and event_rows[-1].get("event_type") == "RUN_FROZEN"
    seq_contiguous = [row.get("seq_no") for row in event_rows] == list(range(1, len(event_rows) + 1))
    event_count_matches = len(event_rows) == int(bundle.trace.event_count or 0)
    # RUN_FROZEN commits the checksum of the already-drained prefix.  The
    # event itself necessarily changes the final JSONL checksum, so checking
    # the prefix makes the marker verifiable without a self-referential hash.
    prefix_checksum = ""
    try:
        raw_lines = Path(bundle.trace.path).read_bytes().splitlines(keepends=True)
        if raw_lines:
            prefix_checksum = hashlib.sha256(b"".join(raw_lines[:-1])).hexdigest()
    except OSError:
        pass
    freeze_payload = event_rows[-1].get("payload", {}) if event_rows else {}
    freeze_payload_ok = bool(
        frozen_last
        and freeze_payload.get("manifest_hash") == prefix_checksum
        and freeze_payload.get("checksum") == prefix_checksum
    )
    pending = 0
    manifest_matches = True
    if repository is not None:
        pending = int(repository.conn.execute("SELECT COUNT(*) FROM trace_outbox WHERE run_id=? AND delivered_at IS NULL", (bundle.run_id,)).fetchone()[0])
        manifest = repository.conn.execute("SELECT trace_checksum,event_count,snapshot_hash FROM trace_manifests WHERE run_id=?", (bundle.run_id,)).fetchone()
        manifest_matches = bool(manifest) and str(manifest[0]) == bundle.trace.checksum and int(manifest[1]) == len(event_rows)
    version_complete = len(bundle.version_tuple.model_dump(by_alias=True)) == 16
    return {"ok": not mismatches and frozen_last and freeze_payload_ok and seq_contiguous and event_count_matches and pending == 0 and manifest_matches and world_hash_matches and version_complete, "checksum_mismatches": mismatches, "run_frozen_last": frozen_last, "freeze_payload_ok": freeze_payload_ok, "seq_contiguous": seq_contiguous, "event_count_matches": event_count_matches, "pending": pending, "manifest_matches": manifest_matches, "world_hash_matches": world_hash_matches, "version_tuple_complete": version_complete}


__all__ = ["TraceDrainManifest", "TraceRecorder", "verify_bundle"]
