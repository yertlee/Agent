"""CLI for the five independent M5 demos."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import yaml
from .demos import build_demo_manifest, run_demos

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="M5 demo runner")
    parser.add_argument("--manifest", default="eval/manifests/m5_demos.yaml")
    parser.add_argument("--output", default="reports/m5_demos.json")
    args = parser.parse_args(argv)
    path = Path(args.manifest)
    manifest = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else build_demo_manifest()
    report = run_demos(manifest)
    overall = "PASS" if report["outputs"] and all(row.get("status") == "PASS" and row.get("assertion_status") == "PASS" and row.get("bundle_verified") for row in report["outputs"]) else "FAIL"
    report["status"] = overall
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": overall, "report": str(Path(args.output)), "case_count": report["case_count"]}, ensure_ascii=False))
    return 0 if overall == "PASS" else 1

if __name__ == "__main__":
    raise SystemExit(main())
