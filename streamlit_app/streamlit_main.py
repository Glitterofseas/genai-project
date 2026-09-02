"""Streamlit proof of concept for the SMS recruiting chatbot.

    streamlit run streamlit_app/streamlit_main.py

Two entry points, as in the workflow diagram: fill the registration form, or go
straight into the conversation.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import streamlit as st

# Streamlit runs this by path, so the project root needs to be importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from streamlit_app import utils  # noqa: E402
from app.modules.agents.types import Action  # noqa: E402

st.set_page_config(page_title="SMS Recruiting Bot", page_icon="", layout="centered")


def sidebar() -> tuple[str, bool]:
    with st.sidebar:
        st.subheader("Configuration")
        kind = st.radio(
            "Agent",
            options=("llm", "rule"),
            format_func=lambda k: "LLM (LangChain + OpenAI)" if k == "llm" else "Rule-based (free)",
            help="The rule-based agent runs with no API calls - useful for a quick demo.",
        )
        show_trace = st.checkbox("Show advisor trace", value=True)

        st.divider()
        st.caption("Environment")
        for label, value in utils.env_banner():
            st.caption(f"**{label}:** {value}")

        if st.button("Restart conversation", use_container_width=True):
            for key in ("profile", "memory", "finished", "trace"):
                st.session_state.pop(key, None)
            st.rerun()
    return kind, show_trace


def registration_form() -> None:
    st.title("Python Developer - candidate screening")
    st.caption(
        "Fill in your details to begin, or skip straight to the conversation. "
        "The conversation date anchors relative phrases like *next Friday*."
    )

    with st.form("registration"):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Name", value="")
            phone = st.text_input("Phone", value="+1-555-0100")
        with col2:
            position = st.selectbox(
                "Position", utils.POSITION_CHOICES,
                index=utils.POSITION_CHOICES.index("Python Dev"),
            )
            date = st.date_input("Conversation date", value=dt.date.today())

        warning = utils.day_warning(date)
        if warning:
            st.info(warning)

        submitted = st.form_submit_button("Start conversation", use_container_width=True)

    skip = st.button("Skip registration and just chat", use_container_width=True)

    if submitted or skip:
        utils.start_conversation(
            {
                "name": name or "Candidate",
                "phone": phone or "+1-555-0100",
                "position": position if submitted else "Python Dev",
                "date": date if submitted else dt.date.today(),
            }
        )
        st.rerun()


def render_trace(entry: dict) -> None:
    label = f"{entry['action']} - advisors: {', '.join(entry['advisors']) or 'none'}"
    with st.expander(label, expanded=False):
        for line in entry["verdicts"]:
            st.markdown(f"- {line}")
        if entry["slots"]:
            st.markdown("**Slots offered from the SQL schedule:**")
            for slot in entry["slots"]:
                st.markdown(f"- {slot}")


def conversation_view(kind: str, show_trace: bool) -> None:
    profile = st.session_state.profile
    st.title("Python Developer - candidate screening")
    st.caption(
        f"{profile['name']} · {profile['phone']} · {profile['position']} · "
        f"conversation date {profile['date']:%A %d %B %Y}"
    )

    agent = utils.load_agent(kind, utils.fine_tuned_model_id() if kind == "llm" else None)

    traces = {t["turn_id"]: t for t in st.session_state.trace}
    for turn in utils.conversation_turns():
        role = "assistant" if turn.speaker == "recruiter" else "user"
        with st.chat_message(role):
            st.write(turn.text)
            if show_trace and turn.turn_id in traces:
                render_trace(traces[turn.turn_id])

    if st.session_state.finished:
        st.success("Conversation ended. Use *Restart conversation* in the sidebar.")
        return

    reply = st.chat_input("Reply as the candidate...")
    if not reply:
        return

    utils.append_turn("candidate", reply)
    with st.chat_message("user"):
        st.write(reply)

    with st.chat_message("assistant"):
        with st.spinner("Consulting advisors..."):
            decision = agent.decide(utils.current_context())
        st.write(decision.message)

    utils.append_turn("recruiter", decision.message)
    st.session_state.trace.append(
        {
            "turn_id": utils.next_turn_id(),
            "action": decision.action.value,
            "advisors": decision.advisors_used,
            "verdicts": [v.summary().replace("\n", "  \n") for v in decision.consulted],
            "slots": [s.label() for v in decision.consulted for s in v.slots],
        }
    )
    if decision.action is Action.END:
        st.session_state.finished = True
    st.rerun()


def main() -> None:
    kind, show_trace = sidebar()
    if "profile" not in st.session_state:
        registration_form()
    else:
        conversation_view(kind, show_trace)


main()
