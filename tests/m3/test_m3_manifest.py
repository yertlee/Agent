from pathlib import Path

from eval.m3_manifest import validate_dev44


def test_dev44_manifest_schema_quota_and_leakage_checks():
    result = validate_dev44(Path("eval/manifests/dev44.yaml"))
    assert result["case_count"] == 44
    assert result["legacy_count"] == 21
    assert result["synthetic_count"] == 23
    assert result["semantic_fingerprint_count"] == 44
    assert sum(result["category_counts"].values()) == 44
