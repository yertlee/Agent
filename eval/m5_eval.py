"""M5 heldout evaluator entry point with a safe sealed-metadata state."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from .datasets import load_dataset_manifest, validate_test20_metadata

def evaluate_heldout(manifest_path: str | Path, *, output: str | Path | None = None) -> dict:
    manifest = load_dataset_manifest(manifest_path)
    validate_test20_metadata(manifest)
    # The repository intentionally contains only sealed test metadata.  The
    # evaluator cannot invent owner-provided payloads or gold, so every case is
    # explicitly NOT_RUN and the gate remains BLOCKED.
    statuses = [{"scenario_id": case.scenario_id, "status": "NOT_RUN", "reason": "owner-provided frozen payload/gold unavailable"} for case in manifest.cases]
    report = {"report_version": "m5.evaluator.v1", "gate": "M5", "status": "BLOCKED",
              "blocked_reason": "sealed test20 metadata has no owner-provided frozen payload/gold",
              "dataset_version": manifest.dataset_version, "manifest_checksum": manifest.checksum,
              "N": len(statuses), "status_counts": {"NOT_RUN": len(statuses)}, "outputs": statuses,
              "metrics": {}, "test_execution": "NOT_RUN"}
    report["checksum"] = hashlib.sha256(json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if output:
        target = Path(output); target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="M5 heldout evaluation")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default="reports/m5_test20.json")
    args = parser.parse_args(argv)
    report = evaluate_heldout(args.manifest, output=args.output)
    print(json.dumps({"status": report["status"], "report": args.output, "N": report["N"]}, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 2

if __name__ == "__main__":
    raise SystemExit(main())
