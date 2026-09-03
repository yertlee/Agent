"""Canonical deterministic WorldSnapshot construction."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

from .contracts import WorldSnapshot


class WorldStateBuilder:
    def __init__(self, *, world_template_version: str | None = None):
        self.world_template_version = world_template_version

    def from_fixture_path(self, path: str | Path, *, world_fixture_ref: str | None = None) -> WorldSnapshot:
        value = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if isinstance(value, dict) and "world_fixtures" in value:
            rows = value["world_fixtures"].get("fixtures", [])
            ref = world_fixture_ref or (rows[0].get("world_fixture_ref") if rows else None)
            value = next((row for row in rows if row.get("world_fixture_ref") == ref), None)
            if value is None:
                raise ValueError(f"world fixture not found: {ref}")
        return self.build(value, world_fixture_ref=world_fixture_ref)

    def build(self, fixture: Mapping[str, Any], *, world_fixture_ref: str | None = None) -> WorldSnapshot:
        raw = dict(fixture)
        clock = raw.get("scene_clock")
        if clock is None:
            raise ValueError("fixture must provide scene_clock; wall clock is not a scene clock")
        parsed = datetime.fromisoformat(str(clock).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("fixture scene_clock must be timezone-aware UTC RFC3339")
        parsed = parsed.astimezone(timezone.utc)
        entities = raw.get("entities") or []
        if not isinstance(entities, list) or any(not isinstance(item, Mapping) for item in entities):
            raise ValueError("fixture entities must be a list of objects")
        return WorldSnapshot(
            world_fixture_ref=str(world_fixture_ref or raw.get("world_fixture_ref") or "world-inline-v1"),
            world_template_version=str(self.world_template_version or raw.get("world_template_version") or raw.get("template_version") or "world-template.v1"),
            seed=int(raw.get("seed", 0)),
            scene_clock=parsed,
            entities=[dict(item) for item in entities],
        )


__all__ = ["WorldStateBuilder"]
