from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from .runtime import build_agent_graph
from .state import AgentState, initial_state
from .langsmith_utils import traceable


CHECKPOINTER = InMemorySaver()
GRAPH = build_agent_graph().compile(checkpointer=CHECKPOINTER)


def _config(session_id: str) -> Dict[str, Any]:
    return {"configurable": {"thread_id": session_id}}


def _safe_snapshot(session_id: str):
    try:
        return GRAPH.get_state(_config(session_id))
    except Exception:
        return None


def load_v3_state(session_id: str) -> AgentState:
    state = initial_state(session_id)
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
def invoke_turn_v3(session_id: str, user_input: str) -> AgentState:
    config = _config(session_id)
    if has_pending_interrupt(session_id):
        GRAPH.invoke(Command(resume={"user_input": user_input}), config=config)
    else:
        GRAPH.invoke({"session_id": session_id, "user_input": user_input}, config=config)

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
