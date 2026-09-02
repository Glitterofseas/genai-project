"""Entry point for the SMS recruiting chatbot.

    python -m app.main                 # terminal chat, LLM agent
    python -m app.main --agent rule    # the free rule-based agent
    python -m app.main --date 2024-04-30

--date sets the conversation anchor, which is what "next Friday" resolves
against. The schedule covers 2024 plus the current and next year.
"""
from __future__ import annotations

import argparse
import datetime as dt

from .modules.agents.memory import ConversationMemory
from .modules.agents.types import Action
from .modules.config.settings import get_settings
from .modules.db.store import get_store


def build_agent(kind: str, store):
    if kind == "rule":
        from .modules.agents.rule_based import build_rule_based_agent

        return build_rule_based_agent(store, _retriever_or_none())

    from .modules.agents.llm import build_llm_agent
    from .modules.fine_tuning.run_job import load_model_id

    return build_llm_agent(store, _retriever_or_none(), exit_model=load_model_id())


def _retriever_or_none():
    """The vector index is optional - the bot still runs without it."""
    try:
        from .modules.embedding.retriever import get_retriever

        retriever = get_retriever()
        return retriever if retriever.is_ready() else None
    except Exception:
        return None


def run_cli(agent_kind: str, anchor_date: dt.date, position: str) -> None:
    store = get_store()
    agent = build_agent(agent_kind, store)

    # Memory holds the live dialogue and produces the advisors' context.
    memory = ConversationMemory(
        anchor=dt.datetime.combine(anchor_date, dt.time(10, 0)), position=position
    )

    print(f"SMS recruiting bot - {position} - conversation date {anchor_date:%A %d %B %Y}")
    print("Type your replies as the candidate. Ctrl-C to quit.\n")

    opening = "Thanks for applying to our Python Developer opening. What kinds of Python projects have you worked on recently?"
    memory.add_recruiter(opening)
    print(f"bot> {opening}")

    while True:
        try:
            reply = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return
        if not reply:
            continue

        memory.add_candidate(reply)
        decision = agent.decide(memory.context())
        memory.add_recruiter(decision.message)

        print(f"bot> {decision.message}")
        print(f"     [{decision.action.value} | advisors: {', '.join(decision.advisors_used) or 'none'}]")
        if decision.action is Action.END:
            print("\n(conversation ended)")
            return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", choices=("llm", "rule"), default="llm")
    parser.add_argument("--date", default=None, help="conversation date, YYYY-MM-DD")
    parser.add_argument("--position", default="Python Dev")
    args = parser.parse_args()

    anchor = (
        dt.date.fromisoformat(args.date) if args.date else dt.date.today()
    )
    if args.agent == "llm" and not get_settings().has_api_key:
        raise SystemExit("OPENAI_API_KEY is not set - add it to .env, or use --agent rule")
    run_cli(args.agent, anchor, args.position)


if __name__ == "__main__":
    main()
