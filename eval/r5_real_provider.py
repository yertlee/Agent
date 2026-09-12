"""Real-LLM Router/Planner provider for the R5 evaluator.

Uses the project's existing model configuration (``agent.llm.build_chat_model``)
and JSON response mode.  The provider receives only the prompt and returns a
plain mapping with ``router`` and ``plan`` keys; the boundary owns schema
validation and bounded repair, so raw model casing/enum mistakes are visible
to the repair loop instead of being swallowed by an eager parser.
"""
from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict

from agent.r5_plan_contracts import R5PlanV1
from agent.r5_router_planner import R5RouterDecisionV1


class R5RouterPlannerEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    router: R5RouterDecisionV1
    plan: R5PlanV1


def _strip_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[-1]
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[:-3]
    return stripped.strip()


class RealRouterPlannerProvider:
    """JSON-mode provider with lazy model construction."""

    name = "real_llm"

    def __init__(self, *, temperature: float = 0.0, max_tokens: int = 8000):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._model = None
        self.last_usage: dict[str, Any] | None = None
        self.call_attempts = 0
        self.returned_calls = 0
        # Backwards-compatible alias used by older report consumers.
        self.calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    def _structured(self):
        if self._model is None:
            from agent.llm import build_chat_model

            model = build_chat_model(temperature=self.temperature, max_tokens=self.max_tokens, tags=["r5", "router_planner"])
            self._model = model.bind(response_format={"type": "json_object"})
        return self._model

    def __call__(self, prompt: str, router_schema=None, plan_schema=None):
        self.call_attempts += 1
        result = self._structured().invoke(prompt)
        self.returned_calls += 1
        self.calls = self.returned_calls
        usage = getattr(result, "usage_metadata", None) or {}
        self.last_usage = usage
        self.total_input_tokens += int(usage.get("input_tokens") or 0)
        self.total_output_tokens += int(usage.get("output_tokens") or 0)
        content = getattr(result, "content", "")
        if not content:
            raise ValueError("empty model content")
        data = json.loads(_strip_fences(str(content)))
        if not isinstance(data, dict):
            raise ValueError("model output is not a JSON object")
        return data


__all__ = ["RealRouterPlannerProvider", "R5RouterPlannerEnvelope"]
