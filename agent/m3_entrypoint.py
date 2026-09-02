"""Feature-flagged isolated M3 entrypoint; legacy/M2 callers remain unchanged."""
from __future__ import annotations

import os
from typing import Any, Mapping

from .m3_runtime import M3ScenarioRunner


def m3_enabled() -> bool:
    return os.getenv("M3_EXECUTION_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}


def run_m3_case(case: Mapping[str, Any]):
    if not m3_enabled():
        raise RuntimeError("M3 execution is disabled")
    return M3ScenarioRunner().run(case)


__all__ = ["m3_enabled", "run_m3_case"]
