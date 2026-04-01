from __future__ import annotations

from copy import deepcopy
import importlib
from pathlib import Path
from typing import Any, Dict, Optional

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from . import planner as planner_module
from . import runtime as runtime_module
from . import specialists as specialists_module
from . import state as state_module
from . import verifier as verifier_module
from .langsmith_utils import traceable


CHECKPOINTER = InMemorySaver()
GRAPH = None
_GRAPH_SIGNATURE = None
_WATCHED_MODULES = {
    "state": state_module,
    "planner": planner_module,
    "specialists": specialists_module,
    "verifier": verifier_module,
    "runtime": runtime_module,
}


def _module_signature() -> tuple[tuple[str, int], ...]:
    signature = []
    base_dir = Path(__file__).resolve().parent
    for name in _WATCHED_MODULES:
        file_path = base_dir / f"{name}.py"
        signature.append((name, file_path.stat().st_mtime_ns))
    return tuple(signature)


def _reload_runtime_modules_if_needed() -> None:
    global GRAPH, _GRAPH_SIGNATURE
    global state_module, planner_module, specialists_module, verifier_module, runtime_module

    signature = _module_signature()
    if GRAPH is not None and signature == _GRAPH_SIGNATURE:
        return

    if _GRAPH_SIGNATURE is not None:
        state_module = importlib.reload(state_module)
        planner_module = importlib.reload(planner_module)
        specialists_module = importlib.reload(specialists_module)
        verifier_module = importlib.reload(verifier_module)
        runtime_module = importlib.reload(runtime_module)

    GRAPH = runtime_module.build_agent_graph().compile(checkpointer=CHECKPOINTER)
    _GRAPH_SIGNATURE = signature


def _get_graph():
    _reload_runtime_modules_if_needed()
    return GRAPH


def _config(session_id: str) -> Dict[str, Any]:
    return {"configurable": {"thread_id": session_id}}


def _safe_snapshot(session_id: str):
    try:
        return _get_graph().get_state(_config(session_id))
    except Exception:
        return None


def load_v3_state(session_id: str) -> state_module.AgentState:
    state = state_module.initial_state(session_id)
    snapshot = _safe_snapshot(session_id)
    if snapshot and snapshot.values:
        state.update(deepcopy(snapshot.values))
    return state


def get_interrupt_payload(session_id: str) -> Optional[Dict[str, Any]]:
    snapshot = _safe_snapshot(session_id)
    if snapshot and snapshot.interrupts:
        interrupt_obj = snapshot.interrupts[0]
        value = getattr(interrupt_obj, "value", None)
        if isinstance(value, dict):
            return value
    return None


def has_pending_interrupt(session_id: str) -> bool:
    snapshot = _safe_snapshot(session_id)
    return bool(snapshot and snapshot.interrupts)


def get_runtime_snapshot(session_id: str):
    return _safe_snapshot(session_id)


@traceable(name="agent_v3_invoke_turn")
def invoke_turn_v3(session_id: str, user_input: str) -> state_module.AgentState:
    graph = _get_graph()
    config = _config(session_id)
    if has_pending_interrupt(session_id):
        graph.invoke(Command(resume={"user_input": user_input}), config=config)
    else:
        graph.invoke({"session_id": session_id, "user_input": user_input}, config=config)

    state = load_v3_state(session_id)
    interrupt_payload = get_interrupt_payload(session_id)
    if interrupt_payload:
        pending_question = str(interrupt_payload.get("pending_question") or state.get("pending_question") or "")
        state["final_response"] = pending_question
    return state


if __name__ == "__main__":
    sid = "agent_v3_demo"
    print("输入 'exit' 退出。")
    while True:
        user_in = input("用户: ").strip()
        if user_in.lower() in {"exit", "quit"}:
            break
        current_state = invoke_turn_v3(sid, user_in)
        print("系统:", current_state.get("final_response") or current_state.get("pending_question") or "")
