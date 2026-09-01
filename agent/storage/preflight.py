"""Read-only SQLite preflight and postflight checks for M0.

The checks inspect schema metadata, counts and integrity constraints only.  No
row values are selected, so this module is safe to run against the source
database and against an isolated backup/restore copy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .backup import EXPECTED_BASELINE_COUNTS, _ro_connect, get_table_counts, sha256_file

MILESTONE = "M0"
OWNER = "m0-implementation-model"


def schema_fingerprint(conn: sqlite3.Connection) -> str:
    rows = conn.execute(
        "SELECT sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY name"
    ).fetchall()
    material = "".join(str(row[0]) for row in rows).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _index_summary(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    tables = [
        str(row[0])
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
    ]
    for table in tables:
        for row in conn.execute(f"PRAGMA index_list({table})").fetchall():
            result.append({"table": table, "name": str(row[1]), "unique": bool(row[2])})
    return result


def inspect_database(path: str | Path) -> dict[str, Any]:
    """Return metadata-only checks for a database opened in read-only mode."""
    db_path = Path(path)
    conn = _ro_connect(db_path)
    try:
        integrity = str(conn.execute("PRAGMA integrity_check").fetchone()[0])
        foreign_key_violations = len(conn.execute("PRAGMA foreign_key_check").fetchall())
        counts = get_table_counts(conn)
        fingerprint = schema_fingerprint(conn)
        indexes = _index_summary(conn)
        foreign_keys = {
            table: len(conn.execute(f"PRAGMA foreign_key_list({table})").fetchall())
            for table in counts
        }
    finally:
        conn.close()
    return {
        "path": str(db_path),
        "sha256": sha256_file(db_path),
        "schema_fingerprint": fingerprint,
        "counts": counts,
        "integrity_check": integrity,
        "foreign_key_violations": foreign_key_violations,
        "foreign_key_counts": foreign_keys,
        "indexes": indexes,
    }


def _load_schema_expectations(path: str | Path | None) -> Mapping[str, Any]:
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("schema manifest must be a JSON object")
    return payload


def run_preflight(
    path: str | Path,
    schema_manifest: str | Path | None = None,
    expected_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    observed = inspect_database(path)
    expected = _load_schema_expectations(schema_manifest)
    errors: list[str] = []
    if observed["integrity_check"] != "ok":
        errors.append("integrity_check_failed")
    if observed["foreign_key_violations"]:
        errors.append("foreign_key_check_failed")
    expected_fingerprint = expected.get("schema_fingerprint")
    if expected_fingerprint and observed["schema_fingerprint"] != expected_fingerprint:
        errors.append("schema_fingerprint_mismatch")
    expected_table_counts = expected_counts or expected.get("counts") or {}
    for table, count in expected_table_counts.items():
        if observed["counts"].get(table) != int(count):
            errors.append(f"count_mismatch:{table}")
    # M0's current schema has unique primary-key indexes.  Report, and assert
    # that metadata was actually inspected rather than silently skipping it.
    if not observed["indexes"]:
        errors.append("no_index_metadata")
    expected_unique = expected.get("unique_index_count")
    if expected_unique is not None:
        actual_unique = sum(1 for index in observed["indexes"] if index["unique"])
        if actual_unique != int(expected_unique):
            errors.append("unique_index_count_mismatch")
    expected_foreign_keys = expected.get("foreign_key_counts")
    if isinstance(expected_foreign_keys, dict) and expected_foreign_keys != observed["foreign_key_counts"]:
        errors.append("foreign_key_metadata_mismatch")
    return {
        "kind": "preflight",
        "milestone": MILESTONE,
        "owner": OWNER,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "input": observed,
        "expected": {
            "schema_fingerprint": expected_fingerprint,
            "counts": dict(expected_table_counts),
            "unique_index_count": expected_unique,
            "foreign_key_counts": expected_foreign_keys,
        },
        "errors": errors,
        "status": "pass" if not errors else "fail",
    }


def run_postflight(
    path: str | Path,
    schema_manifest: str | Path | None = None,
    expected_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Postflight uses the same metadata checks against an isolated output."""
    report = run_preflight(path, schema_manifest, expected_counts)
    report["kind"] = "postflight"
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m agent.storage.preflight")
    parser.add_argument("--input", required=True, help="database path; opened read-only")
    parser.add_argument("--schema", default=None, help="JSON schema manifest with fingerprint/counts")
    parser.add_argument("--report", default=None, help="JSON report output path")
    parser.add_argument("--postflight", action="store_true", help="label check as postflight")
    args = parser.parse_args(argv)
    report = run_postflight(args.input, args.schema) if args.postflight else run_preflight(args.input, args.schema)
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.report:
        target = Path(args.report)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    print(text)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
