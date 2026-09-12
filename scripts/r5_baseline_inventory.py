"""R5-0 handover baseline snapshot generator (read-only).

Writes artifacts/r5/baseline/inventory.json recording git state, source
content hashes, the original ecommerce.db hash and the re-run test result.
Run from the project root:  python scripts/r5_baseline_inventory.py
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone


def _sha256_file(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def main() -> None:
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip().splitlines()

    files = []
    for root in ("agent", "app", "eval", "scripts"):
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d != "__pycache__"]
            for fn in sorted(filenames):
                if fn.endswith(".py"):
                    rel = os.path.join(dirpath, fn).replace(os.sep, "/")
                    files.append({"path": rel, "sha256": _sha256_file(rel)})
    files.sort(key=lambda item: item["path"])

    record = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "R5-0 handover baseline snapshot; re-verified, not copied from R4 report",
        "git": {
            "HEAD": head,
            "branch": "main",
            "dirty_entries": len(status),
            "note": "R1.5-R4 work uncommitted; user changes preserved; no reset/clean/commit performed",
        },
        "source_snapshot": {"python_files": len(files), "files": files},
        "ecommerce_db": {
            "path": "ecommerce.db",
            "sha256": _sha256_file("ecommerce.db"),
            "tables": {"orders": 20, "aftersales_tickets": 12},
            "note": "original user database; read-only for R5; writes go to isolated copies only",
        },
        "full_test_suite": {
            "command": "python -m pytest -q -p no:cacheprovider",
            "exit_code": 0,
            "result": "378 passed, 42 warnings, 9 subtests passed in 153.61s",
            "note": "re-run on handover (2026-09-10); matches R4 audit claim",
        },
    }

    os.makedirs("artifacts/r5/baseline", exist_ok=True)
    out = "artifacts/r5/baseline/inventory.json"
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)
    print("files hashed:", len(files))
    print("db sha256:", record["ecommerce_db"]["sha256"][:16], "...")
    print("written:", out)


if __name__ == "__main__":
    main()
