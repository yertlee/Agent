"""Immutable scenario loading with runtime/evaluator gold separation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from .contracts import FailureScript, Scenario, VersionTuple


class ScenarioLoader:
    def __init__(self, *, require_version_tuple: bool = True):
        self.require_version_tuple = require_version_tuple

    def load(self, path: str | Path) -> Scenario:
        value = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return self.from_mapping(value)

    def from_mapping(self, value: Mapping[str, Any]) -> Scenario:
        raw = dict(value)
        for wrapper in ("scenario", "scenarios"):
            if wrapper in raw and isinstance(raw[wrapper], dict):
                raw = dict(raw[wrapper])
                break
        if "scenarios" in raw and isinstance(raw["scenarios"], list):
            raise ValueError("from_mapping accepts one scenario; use load_many for a collection")
        if self.require_version_tuple and "version_tuple" not in raw:
            raise ValueError("scenario must bind a version_tuple")
        version = raw.get("version_tuple")
        if not isinstance(version, VersionTuple):
            version = VersionTuple.model_validate(version or {})
        runtime = {
            "scenario_id": raw.get("scenario_id"),
            "category": raw.get("category", "uncategorized"),
            "turns": raw.get("turns") or [],
            "intent_label": raw.get("intent_label"),
            "world_fixture_ref": raw.get("world_fixture_ref"),
            "split": raw.get("split", "dev"),
            "version_tuple": version,
            "failure_script": FailureScript.from_mapping(raw.get("failure_script")),
            "tool_calls": raw.get("tool_calls") or raw.get("operations") or [],
        }
        # Never pass gold/expected/rubric keys into Scenario.  They can be
        # loaded separately by the evaluator from its manifest.
        return Scenario.model_validate(runtime)

    def load_many(self, path: str | Path) -> list[Scenario]:
        value = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        rows = value.get("scenarios", value.get("cases", value)) if isinstance(value, dict) else value
        if not isinstance(rows, list):
            return [self.from_mapping(value)]
        return [self.from_mapping(row) for row in rows]


def load_scenario(path: str | Path) -> Scenario:
    return ScenarioLoader().load(path)


__all__ = ["ScenarioLoader", "load_scenario"]
