from __future__ import annotations

import uuid

import sys
from pathlib import Path

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agent.state import state_to_debug_dict
from agent.graph_agent import get_interrupt_payload, get_runtime_snapshot, invoke_turn_v3, load_v3_state


def _safe_rerun() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()


def _init_ui_state() -> None:
    if "sessions" not in st.session_state:
        st.session_state.sessions = []
    if "current_session_id" not in st.session_state:
        session_id = "graph_" + uuid.uuid4().hex[:8]
        st.session_state.sessions.append(session_id)
        st.session_state.current_session_id = session_id
    if "show_debug" not in st.session_state:
        st.session_state.show_debug = True


def _new_session() -> None:
    session_id = "graph_" + uuid.uuid4().hex[:8]
    st.session_state.sessions.append(session_id)
    st.session_state.current_session_id = session_id


def _render_sidebar() -> None:
    st.sidebar.title("Graph Sessions")
    if st.sidebar.button("新建会话"):
        _new_session()
        _safe_rerun()
        return

    sessions = st.session_state.sessions
    current = st.session_state.current_session_id
    if sessions:
        index = sessions.index(current) if current in sessions else 0
        selected = st.sidebar.selectbox("选择 Session", options=sessions, index=index)
        st.session_state.current_session_id = selected

    st.session_state.show_debug = st.sidebar.checkbox("显示 Runtime 面板", value=st.session_state.show_debug)


def _render_messages(state: dict) -> None:
    st.markdown("### Chat")
    messages = state.get("messages") or []
    interrupt_payload = get_interrupt_payload(st.session_state.current_session_id)
    last_ai_text = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            last_ai_text = str(msg.content or "")
            break
    if not messages:
        st.info("输入订单、售后、规则或闲聊问题，我会通过 LangGraph runtime 进行处理。")
    else:
        for msg in messages:
            if isinstance(msg, HumanMessage):
                with st.chat_message("user"):
                    st.markdown(msg.content)
            elif isinstance(msg, AIMessage):
                with st.chat_message("assistant"):
                    st.markdown(msg.content)

    if interrupt_payload:
        pending_question = interrupt_payload.get("pending_question") or state.get("pending_question")
        if pending_question and pending_question != last_ai_text:
            with st.chat_message("assistant"):
                st.markdown(pending_question)


def _render_plan(state: dict) -> None:
    st.markdown("#### Current Plan")
    plan = state.get("current_plan") or []
    if not plan:
        st.caption("当前没有活动计划")
        return
    rows = []
    current_index = int(state.get("current_step_index") or 0)
    for idx, step in enumerate(plan):
        rows.append(
            {
                "idx": idx,
                "current": ">>" if idx == current_index else "",
                "step_id": step.step_id,
                "owner": step.owner_agent.value,
                "action": step.action_type.value,
                "status": step.status.value,
                "goal": step.goal,
                "required": ", ".join(step.required_inputs),
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)


def _render_runtime_panel(state: dict) -> None:
    if not st.session_state.show_debug:
        return

    st.markdown("### Agent Runtime")
    snapshot = get_runtime_snapshot(st.session_state.current_session_id)
    interrupt_payload = get_interrupt_payload(st.session_state.current_session_id)

    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Current Graph Node**: {(state.get('trace_tags') or {}).get('current_node', '')}")
        st.write(f"**Active Specialist**: {getattr(state.get('active_agent'), 'value', state.get('active_agent'))}")
        st.write(f"**Plan Mode**: {getattr(state.get('plan_mode'), 'value', state.get('plan_mode'))}")
        st.write(f"**Current Step Index**: {state.get('current_step_index')}")
        st.write(f"**Blocked Step**: {state.get('blocked_step_id')}")
        st.write(f"**Awaited Slots**: {state.get('awaited_slots')}")
        st.write(f"**Pending Question**: {state.get('pending_question')}")
        st.write(f"**Next Nodes**: {list(snapshot.next) if snapshot else []}")
    with col2:
        st.write(f"**Tool Retries**: {state.get('tool_retry_counts')}")
        st.write(f"**RAG Retry Count**: {state.get('rag_retry_count')}")
        st.write(f"**Replan Count**: {state.get('replan_count')}")
        st.write(f"**Response Mode**: {getattr(state.get('response_mode'), 'value', state.get('response_mode'))}")
        st.write(f"**Handoff Reason**: {state.get('handoff_reason')}")
        st.write(f"**LangSmith Project**: {(state.get('trace_tags') or {}).get('langsmith_project', '')}")
        st.write(f"**Trace Thread ID**: {(state.get('trace_tags') or {}).get('thread_id', '')}")
        if interrupt_payload:
            st.write(f"**Interrupt Payload**: {interrupt_payload}")

    _render_plan(state)

    st.markdown("#### Verifier Result")
    verifier = state.get("verification_status")
    st.json(verifier.model_dump() if verifier else {})

    st.markdown("#### Last Observation")
    last_observation = state.get("last_observation")
    st.json(last_observation.model_dump() if last_observation else {})

    with st.expander("Verified Facts", expanded=False):
        st.json([fact.model_dump() for fact in (state.get("verified_facts") or [])])

    with st.expander("Retrieval Evidence", expanded=False):
        st.json([item.model_dump() for item in (state.get("retrieval_evidence") or [])])

    with st.expander("Raw State", expanded=False):
        st.json(state_to_debug_dict(state))


def main() -> None:
    st.set_page_config(page_title="LangGraph Agent V3", layout="wide")
    st.title("基于 LangGraph 的电商客服智能体 V3")
    st.caption("Supervisor + Specialist Graph, verifier-controlled finalization, LangSmith-ready tracing")

    _init_ui_state()
    _render_sidebar()

    session_id = st.session_state.current_session_id
    state = load_v3_state(session_id)

    left, right = st.columns([3, 2])
    with left:
        _render_messages(state)
        user_text = st.chat_input("请输入订单、售后、规则或闲聊问题...")
        if user_text:
            with st.spinner("Graph 正在运行..."):
                invoke_turn_v3(session_id, user_text)
            _safe_rerun()

    with right:
        _render_runtime_panel(state)


if __name__ == "__main__":
    main()
