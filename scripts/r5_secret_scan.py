"""Scan publishable repository files for credential shapes.

The scanner reports only counts and file locations, never matched values. It
uses Git's tracked/untracked inventory with standard ignores, so local ``.env``
and generated artifacts are never read.

Run:  python scripts/r5_secret_scan.py
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SKIP_NAMES = {".env", "inventory.json"}
SKIP_SUFFIXES = {".db", ".sqlite", ".sqlite3", ".pyc", ".png", ".jpg", ".docx", ".pdf"}
MAX_BYTES = 2_000_000

PATTERNS = {
    "openai_key": re.compile(r"sk-[A-Za-z0-9]{20,}"),
    "aws_access_key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "generic_api_key_assignment": re.compile(r"(?i)api[_-]?key\s*[:=]\s*[\"'][A-Za-z0-9_\-]{16,}[\"']"),
    "bearer_token": re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{20,}"),
    "private_key_block": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "dashscope_key": re.compile(r"sk-[A-Za-z0-9]{24,}"),
}


def main() -> None:
    hits: dict[str, list[str]] = {name: [] for name in PATTERNS}
    scanned = 0
    inventory = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.splitlines()
    for relative in inventory:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            continue
        if path.name in SKIP_NAMES or path.suffix.lower() in SKIP_SUFFIXES:
            continue
        try:
            if path.stat().st_size > MAX_BYTES:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        scanned += 1
        for name, pattern in PATTERNS.items():
            if pattern.search(text):
                hits[name].append(str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"))

    result = {
        "scanned_files": scanned,
        "patterns": {name: {"count": len(paths), "files": sorted(set(paths))} for name, paths in hits.items()},
        "hits_total": sum(len(paths) for paths in hits.values()),
    }
    out = PROJECT_ROOT / "artifacts" / "r5" / "secret_scan.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    print("written:", out)


if __name__ == "__main__":
    main()
