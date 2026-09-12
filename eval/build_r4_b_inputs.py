"""CLI/public entry point for the gold-free R4-B input builder."""
from __future__ import annotations

import argparse

from .r4_b_inputs import (
    FAULT_FAMILIES,
    NORMAL_TOPOLOGIES,
    R4BInputV1,
    R4BWorldV1,
    build_r4_b_dev_inputs,
    input_manifest,
    load_r4_b_inputs,
    write_r4_b_dev_inputs,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write exactly 36 gold-free R4-B dev inputs.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    write_r4_b_dev_inputs(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FAULT_FAMILIES",
    "NORMAL_TOPOLOGIES",
    "R4BInputV1",
    "R4BWorldV1",
    "build_r4_b_dev_inputs",
    "input_manifest",
    "load_r4_b_inputs",
    "main",
    "write_r4_b_dev_inputs",
]
