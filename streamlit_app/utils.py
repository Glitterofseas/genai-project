"""Helpers for the Streamlit app: session state, agent construction, formatting."""
from __future__ import annotations

import datetime as dt

import streamlit as st

from app.modules.agents.memory import ConversationMemory
from app.modules.config.settings import get_settings
from app.modules.db.rules import POSITIONS, is_valid_day
from app.modules.db.store import get_store

OPENING = (
    "Thanks for applying to our Python Developer opening. "
    "What kinds of Python projects have you worked on recently?"
)


@st.cache_resource(show_spinner=False)
def load_store():
    return get_store()


@st.cache_resource(show_spinner=False)
def load_retriever():
    try:
        from app.modules.embedding.retriever import get_retriever

        retriever = get_retriever()
        return retriever if retriever.is_ready() else None
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def load_agent(kind: str, exit_model: str | None):
    store = load_store()
    if kind == "rule":
        from app.modules.agents.rule_based import build_rule_based_agent

        return build_rule_based_agent(store, load_retriever())

    from app.modules.agents.llm import build_llm_agent

    return build_llm_agent(store, load_retriever(), exit_model=exit_model)


def fine_tuned_model_id() -> str | None:
    from app.modules.fine_tuning.run_job import load_model_id

    return load_model_id()


def start_conversation(profile: dict) -> None:
    """Seed session state from the registration form.

    The dialogue lives in LangChain-backed ConversationMemory; session state
    just holds it across reruns.
    """
    memory = ConversationMemory(
        anchor=dt.datetime.combine(profile["date"], dt.time(10, 0)),
        position=profile["position"],
        candidate_phone=profile["phone"],
    )
    memory.add_recruiter(OPENING)
    st.session_state.profile = profile
    st.session_state.memory = memory
    st.session_state.finished = False
    st.session_state.trace = []


def append_turn(speaker: str, text: str) -> None:
    st.session_state.memory.add(speaker, text)


def current_context():
    return st.session_state.memory.context()


def conversation_turns():
    return st.session_state.memory.turns


def next_turn_id() -> int:
    return len(st.session_state.memory)


def day_warning(date: dt.date) -> str | None:
    """db_Tech.sql generates no Mondays or Saturdays."""
    if not is_valid_day(date):
        return (
            f"{date:%A}s have no interview slots - the schedule covers "
            "Tuesday to Friday and Sunday. The bot will offer the nearest valid day."
        )
    return None


def index_exists() -> bool:
    """Cheap on-disk check.

    Deliberately does NOT instantiate the retriever: building a Chroma client
    plus an embeddings client takes seconds, and doing it just to render a
    status line blocks the whole app on first paint.
    """
    from app.modules.config.settings import CHROMA_DIR

    return CHROMA_DIR.exists() and any(CHROMA_DIR.glob("*.sqlite3"))


def env_banner() -> list[tuple[str, str]]:
    """Small status rows so a reviewer can see what is wired up."""
    settings = get_settings()
    rows = [
        ("Schedule backend", settings.schedule_backend),
        ("Chat model", settings.chat_model),
        ("Vector index", "ready" if index_exists() else "not built"),
        ("OpenAI key", "set" if settings.has_api_key else "missing"),
    ]
    model = fine_tuned_model_id()
    rows.append(("Exit advisor", model if model else "few-shot (no fine-tune)"))
    return rows


POSITION_CHOICES = list(POSITIONS)
