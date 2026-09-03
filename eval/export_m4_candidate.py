"""Export reproducible candidate frozen-outcome projections."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--manifest", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from .datasets import load_dataset_manifest
    from .m4_eval import run_dev72
    manifest = load_dataset_manifest(args.manifest)
    evaluation, outputs = run_dev72(manifest, mode="simulated", artifact_dir="reports/candidate_bundles")
    Path(args.output).write_text(json.dumps({"manifest_checksum": manifest.checksum, "evaluation": evaluation, "outputs": outputs}, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__": raise SystemExit(main())
