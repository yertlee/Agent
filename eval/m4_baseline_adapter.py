"""Input-only adapter for running the frozen M3 baseline on M4 cases."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


def install_inline_world_adapter() -> None:
    """Teach the legacy loader to consume the M4 deterministic world input."""
    import agent.m3_runtime as runtime

    if hasattr(runtime, "_load_fixture_original"):
        return
    runtime._load_fixture_original = runtime._load_fixture

    def fixture(ref: str) -> dict[str, object]:
        match = re.match(r"^world-m4-(?P<category>[a-z_]+)-(?P<index>\d+)$", str(ref))
        if not match:
            return runtime._load_fixture_original(ref)  # type: ignore[attr-defined]
        category, index = match.group("category"), match.group("index")
        return {"world_fixture_ref": str(ref), "world_template_version": "world-template.m4.v1", "seed": 4072,
                "scene_clock": "2026-01-01T00:00:00Z", "entities": [{"entity_type": category, "entity_id": f"m4-{category}-{index}"}]}

    runtime._load_fixture = fixture


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export M3 baseline outcomes on an M4 manifest")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    install_inline_world_adapter()
    from .datasets import load_dataset_manifest
    from .m4_eval import run_dev72
    manifest = load_dataset_manifest(args.manifest)
    evaluation, outputs = run_dev72(manifest, mode="simulated", artifact_dir="reports/baseline_bundles")
    target = Path(args.output); target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"manifest_checksum": manifest.checksum, "evaluation": evaluation,
                                  "outputs": outputs}, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
