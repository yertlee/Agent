"""Security boundary for the M5 report surface.

The API deliberately consumes a frozen bundle produced by the runtime port.
It never opens a writable runtime repository and never accepts ownership from
the request body.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

from eval.harness import FreezeBundle, verify_bundle


class OwnershipDenied(PermissionError):
    code = "AUTH_OWNERSHIP_DENIED"


class ChecksumMismatch(ValueError):
    code = "TRACE_CHECKSUM_MISMATCH"


_SENSITIVE_KEY = re.compile(r"(?:token|secret|password|phone|address|payment|card|raw_payload)", re.I)
_PII_VALUE = re.compile(r"(?<!\d)(?:1[3-9]\d{9}|\d{15,19})(?!\d)")


def _redacted(value: Any, path: str) -> Any:
    """Stable projection which does not preserve the sensitive original."""
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            name = str(key)
            if _SENSITIVE_KEY.search(name) and not name.lower().endswith(("_hash", "_ref", "_status")):
                result[name] = "[REDACTED:" + hashlib.sha256((path + "." + name).encode()).hexdigest()[:12] + "]"
            else:
                result[name] = _redacted(child, path + "." + name)
        return result
    if isinstance(value, list):
        return [_redacted(item, f"{path}[{idx}]") for idx, item in enumerate(value)]
    if isinstance(value, str) and _PII_VALUE.search(value):
        return "[REDACTED:" + hashlib.sha256(path.encode()).hexdigest()[:12] + "]"
    return value


def project_trace(events: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return the allow-listed, deterministic trace projection."""
    allowed = {"trace_id", "run_id", "session_id", "plan_revision_id", "task_id", "attempt_id",
               "parent_event_id", "seq_no", "event_type", "occurred_at", "scene_clock", "actor", "payload",
               "schema_version", "payload_hash"}
    projection: list[dict[str, Any]] = []
    for event in events:
        row = {key: _redacted(event[key], f"event.{key}") for key in allowed if key in event}
        projection.append(row)
    return projection


def build_read_model(bundle: FreezeBundle, events: list[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Build the eight object views from the same verified frozen inputs."""
    projected = project_trace(events)
    def where(fragment: str) -> list[dict[str, Any]]:
        return [row for row in projected if fragment in str(row.get("event_type", ""))]
    run = [{"run_id": bundle.run_id, "bundle_id": bundle.bundle_id, "status": "FROZEN", "trace_checksum": bundle.trace.checksum}]
    def artifact(item: Any) -> dict[str, Any]:
        try:
            content = load_json(item.path)
        except (OSError, ValueError):
            content = None
        return {"name": Path(item.path).name, "checksum": item.checksum, "content": _redacted(content, "artifact")}
    plans = [artifact(item) for item in bundle.plan_revisions]
    tasks = where("TASK")
    attempts = where("ATTEMPT")
    tools = where("TOOL")
    results = [artifact(item) for item in bundle.results] or where("RESULT")
    reviews = where("REVIEW")
    evidence = [{"kind": "trace", "name": Path(bundle.trace.path).name, "checksum": bundle.trace.checksum},
                {"kind": "world_snapshot", "name": Path(bundle.world_snapshot.path).name, "checksum": bundle.world_snapshot.checksum},
                {"kind": "final_response", "name": Path(bundle.final_response.path).name, "checksum": bundle.final_response.checksum}]
    return {"Run": run, "Plan": plans, "Task": tasks, "Attempt": attempts, "Tool": tools,
            "Result": results, "Review": reviews, "Evidence": evidence}


def verify_bundle_for_owner(bundle: FreezeBundle, *, owner: str, expected_owner: str) -> dict[str, Any]:
    if not owner or owner != expected_owner:
        raise OwnershipDenied("run ownership does not match authenticated principal")
    result = verify_bundle(bundle)
    if not result.get("ok"):
        raise ChecksumMismatch("frozen bundle is not readable or checksum verification failed")
    return result


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


__all__ = ["ChecksumMismatch", "OwnershipDenied", "project_trace", "build_read_model", "verify_bundle_for_owner", "load_json"]
