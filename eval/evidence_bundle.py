"""Build and verify a checksummed M5 evidence directory."""
from __future__ import annotations
import argparse, hashlib, json
import shutil
from pathlib import Path
from typing import Iterable

def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def build_bundle(output_dir: str | Path, *, artifacts: Iterable[str | Path] = (), status: str = "BLOCKED", heldout_status: str = "NOT_RUN") -> dict:
    root = Path(output_dir); root.mkdir(parents=True, exist_ok=True)
    artifact_dir = root / "artifacts"; artifact_dir.mkdir(parents=True, exist_ok=True)
    refs = []
    for index, item in enumerate(artifacts):
        path = Path(item)
        if path.is_file():
            destination = artifact_dir / f"{index:03d}-{path.name}"
            shutil.copy2(path, destination)
            refs.append({"path": str(destination.relative_to(root)).replace("\\", "/"), "sha256": _sha(destination)})
    manifest = {"bundle_version": "m5.evidence.v1", "status": status, "heldout_status": heldout_status, "artifacts": refs}
    manifest["checksum"] = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    target = root / "manifest.json"
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest

def verify_bundle(path: str | Path) -> dict:
    source = Path(path)
    manifest_path = source / "manifest.json" if source.is_dir() else source
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = []
    expected_manifest_checksum = hashlib.sha256(json.dumps({k: v for k, v in value.items() if k != "checksum"}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if value.get("checksum") != expected_manifest_checksum:
        failures.append("manifest.json:checksum")
    root = manifest_path.parent.resolve()
    for ref in value.get("artifacts", []):
        relative = Path(str(ref["path"]))
        p = (root / relative).resolve()
        if relative.is_absolute() or root not in p.parents:
            failures.append(str(ref.get("path")) + ":path_escape")
            continue
        if not p.is_file() or _sha(p) != ref.get("sha256"): failures.append(ref.get("path"))
    return {"ok": not failures, "failures": failures, "bundle_checksum": value.get("checksum"), "status": value.get("status")}

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="verify M5 evidence bundle")
    parser.add_argument("--bundle")
    parser.add_argument("--output", help="build a bundle directory")
    parser.add_argument("--artifact", action="append", default=[])
    parser.add_argument("--status", default="BLOCKED")
    parser.add_argument("--heldout-status", default="NOT_RUN")
    args = parser.parse_args(argv)
    if args.output:
        result = build_bundle(args.output, artifacts=args.artifact, status=args.status, heldout_status=args.heldout_status)
        print(json.dumps({"status": result["status"], "bundle": str(Path(args.output) / "manifest.json")}, ensure_ascii=False)); return 0
    if not args.bundle:
        parser.error("--bundle or --output is required")
    result = verify_bundle(args.bundle); print(json.dumps(result, ensure_ascii=False)); return 0 if result["ok"] else 1

if __name__ == "__main__": raise SystemExit(main())
