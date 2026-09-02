"""M3 dev-44 manifest loader and deterministic leakage/quota validator."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from agent.domain.objects import canonical_json


class ManifestValidationError(ValueError):
    pass


def load_manifest(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if isinstance(value, dict) and isinstance(value.get("dev44_manifest"), dict):
        value = value["dev44_manifest"]
    if not isinstance(value, dict) or not isinstance(value.get("cases"), list):
        raise ManifestValidationError("manifest must contain cases")
    return value


def manifest_checksum(value: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def validate_dev44(path: str | Path) -> dict[str, Any]:
    value = load_manifest(path)
    rubric = value.get("rubric")
    if not isinstance(rubric, dict) or set(rubric.get("goal_assertions", [])) != {"intent", "plan", "business_code", "tool_path", "terminal", "world_fingerprint"}:
        raise ManifestValidationError("manifest rubric must freeze the six M3 goal assertions")
    cases = value["cases"]
    if len(cases) != 44:
        raise ManifestValidationError(f"expected 44 cases, got {len(cases)}")
    ids = [str(case.get("scenario_id") or "") for case in cases]
    if not all(ids) or len(set(ids)) != len(ids):
        raise ManifestValidationError("scenario IDs must be non-empty and unique")
    if not isinstance(value.get("gold_plan"), dict):
        raise ManifestValidationError("manifest must freeze a normalized gold plan projection")
    expected_codes = rubric.get("expected_business_codes")
    if not isinstance(expected_codes, dict) or not isinstance(expected_codes.get("default"), list):
        raise ManifestValidationError("manifest must freeze expected business-code defaults")
    required = {"scenario_id", "category", "world_fixture_ref", "split", "expected_terminal_class", "expected_tool_path"}
    for case in cases:
        missing = required - set(case)
        if missing:
            raise ManifestValidationError(f"case missing fields: {sorted(missing)}")
        if case["expected_terminal_class"] not in {"PASS", "FAILED", "BLOCKED", "CANCELLED"}:
            raise ManifestValidationError("invalid terminal class")
        if not case.get("turns"):
            raise ManifestValidationError(f"case must embed executable turns: {case['scenario_id']}")
        if str(case["scenario_id"]) not in value["gold_plan"]:
            raise ManifestValidationError(f"case missing gold plan: {case['scenario_id']}")
    fixture_path = Path(__file__).resolve().parent / "scenarios" / "world_fixtures.yaml"
    fixtures_doc = yaml.safe_load(fixture_path.read_text(encoding="utf-8")) or {}
    gold = fixtures_doc.get("world_fixtures", {}).get("gold_fingerprints", {})
    if not isinstance(gold, dict) or any(str(case["world_fixture_ref"]) not in gold for case in cases):
        raise ManifestValidationError("every case must reference a frozen world fingerprint")
    source = [case for case in cases if not case.get("synthetic", False)]
    synthetic = [case for case in cases if case.get("synthetic", False)]
    if len(source) != 21 or len(synthetic) != 23:
        raise ManifestValidationError(f"manifest must contain 21 legacy and 23 synthetic cases, got {len(source)} and {len(synthetic)}")
    refs = [str(case.get("source_case_ref") or "") for case in source]
    if any(not ref for ref in refs) or len(set(refs)) != 21:
        raise ManifestValidationError("legacy source refs must be one-to-one")
    migration_path = Path(__file__).resolve().parents[1] / "reports" / "m3" / "legacy21_migration_map.yaml"
    if migration_path.exists():
        migration = yaml.safe_load(migration_path.read_text(encoding="utf-8"))["legacy21_migration_map"]
        mapped = {str(row["target_scenario_id"]): row for row in migration["rows"]}
        if set(refs) != set(mapped) or len(mapped) != 21:
            raise ManifestValidationError("manifest legacy refs do not match the 21-row migration map")
        for case in source:
            row = mapped[str(case["source_case_ref"])]
            if case["category"] != row["category"] or case["expected_intent"] != row["expected_intent"] or case["expected_tool_path"] != row["expected_tool_path"]:
                raise ManifestValidationError(f"legacy case mapping mismatch: {case['scenario_id']}")
    quota = value.get("quota", {})
    counts = {category: sum(1 for case in cases if case["category"] == category) for category in quota}
    if counts != {str(k): int(v) for k, v in quota.items()}:
        raise ManifestValidationError(f"quota mismatch: {counts} vs {quota}")
    keys = []
    for case in cases:
        key = canonical_json({"turns": case.get("turns", []), "intent": case.get("expected_intent"), "category": case["category"], "tool_path": case["expected_tool_path"], "fixture": case["world_fixture_ref"], "policy_version": case.get("policy_version", "m3.policy.v1")})
        keys.append(hashlib.sha256(key.encode()).hexdigest())
    if len(keys) != len(set(keys)):
        raise ManifestValidationError("duplicate scenario semantic fingerprint")
    fixture_splits: dict[str, str] = {}
    for case in cases:
        fixture = str(case["world_fixture_ref"])
        split = str(case["split"])
        if fixture in fixture_splits and fixture_splits[fixture] != split:
            raise ManifestValidationError("world fixture leakage across split")
        fixture_splits[fixture] = split
    return {"case_count": len(cases), "legacy_count": len(source), "synthetic_count": len(synthetic), "category_counts": counts, "semantic_fingerprint_count": len(set(keys)), "coverage": {"goal_assertions": sorted(rubric["goal_assertions"]), "world_fingerprints": len(gold)}, "manifest_sha256": manifest_checksum(value)}


__all__ = ["ManifestValidationError", "load_manifest", "manifest_checksum", "validate_dev44"]
