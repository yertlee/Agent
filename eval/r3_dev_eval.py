"""R3 development evaluator entry point.

It delegates to the shared runtime in ``agent.r3_rag_runtime`` and never
creates validation or heldout gold.
"""

from .r3_eval import MODES, evaluate, main, run_case

__all__ = ["MODES", "evaluate", "main", "run_case"]

if __name__ == "__main__":
    raise SystemExit(main())
