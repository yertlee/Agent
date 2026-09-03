"""M4 deterministic evaluation harness public contracts and runners.

The package is intentionally an adapter around the public M2/M3 runtime ports.
It does not import or mutate private runner state and never treats evaluator
gold as runtime input.
"""

from .contracts import (
    ArtifactRef,
    ExecutionMode,
    EvaluationInput,
    FailureScript,
    FailureTrigger,
    FreezeBundle,
    Scenario,
    VersionTuple,
    WorldSnapshot,
    FreezeLock,
)
from .scenario_loader import ScenarioLoader, load_scenario
from .world import WorldStateBuilder
from .simulator import HarnessToolResult, ToolSimulator
from .runner import AgentRunner, HarnessRun
from .trace_recorder import TraceRecorder, TraceDrainManifest, verify_bundle
from .replay import DivergenceReport, ReplayResult, ReplayRunner

__all__ = [
    "AgentRunner", "ArtifactRef", "DivergenceReport", "EvaluationInput", "ExecutionMode",
    "FailureScript", "FailureTrigger", "FreezeBundle", "FreezeLock", "HarnessRun",
    "HarnessToolResult", "ReplayResult", "ReplayRunner", "Scenario",
    "ScenarioLoader", "ToolSimulator", "TraceDrainManifest", "TraceRecorder",
    "VersionTuple", "WorldSnapshot", "WorldStateBuilder", "load_scenario", "verify_bundle",
]
