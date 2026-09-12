"""Compatibility entry point for the corrected R2 evaluator.

The former manifest/category evaluator was retired because it generated
answer-bearing phrases and world state from the same category field.
"""
from .r2_correction_eval import evaluate, main

__all__ = ["evaluate", "main"]

if __name__ == "__main__":
    raise SystemExit(main())
