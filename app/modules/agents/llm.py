"""LangChain versions of the Main Agent, the advisors and the composer.

Each advisor decides yes/no first; only a yes triggers its tool.
"""
from __future__ import annotations

from functools import lru_cache

from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from ..config.settings import get_settings
from ..db import dates
from ..db.store import ScheduleStore
from . import prompts
from .tools import build_schedule_toolkit
from .types import (
    Action,
    AdvisorName,
    AdvisorVerdict,
    ConversationContext,
)


# --- structured outputs ---------------------------------------------------


class BinaryVerdict(BaseModel):
    decision: bool = Field(description="the advisor's yes/no answer")
    reason: str = Field(description="one short clause justifying the decision")


class RouteChoice(BaseModel):
    advisor: str = Field(description="one of: exit, sched, info, none")
    reason: str = Field(description="one short clause")


def _chat(model: str | None = None, **params) -> ChatOpenAI:
    settings = get_settings()
    return ChatOpenAI(
        model=model or settings.chat_model,
        api_key=settings.openai_api_key,
        **params,
    )


@lru_cache(maxsize=8)
def _binary_chain(system: str, few_shot: str, model: str | None = None):
    """A cached classifier chain. Cached so few-shot blocks are built once."""
    template = ChatPromptTemplate.from_messages(
        [
            ("system", system + ("\n\nWorked examples:\n" + few_shot if few_shot else "")),
            ("human", "Conversation so far:\n{history}\n\nToday is {anchor}.\n{extra}"),
        ]
    )
    return template | _chat(model, **prompts.CLASSIFIER_PARAMS).with_structured_output(
        BinaryVerdict
    )


# --- advisors -------------------------------------------------------------


class LLMExitAdvisor:
    """End / Don't End.

    `model` can be a fine-tuned model id - same contract either way, so the two
    variants can be swapped and compared.
    """

    name = AdvisorName.EXIT

    def __init__(self, model: str | None = None, use_few_shot: bool = True):
        self.model = model
        # A fine-tuned model already has the examples baked in.
        self.few_shot = (
            prompts.few_shot_block("end", "YES - end", "NO - continue")
            if use_few_shot and not model
            else ""
        )

    def evaluate(self, context: ConversationContext) -> AdvisorVerdict:
        chain = _binary_chain(prompts.EXIT_ADVISOR_SYSTEM, self.few_shot, self.model)
        result = chain.invoke(
            {
                "history": prompts.render_history(context.history),
                "anchor": context.anchor.strftime("%A %d %B %Y"),
                "extra": "Should this conversation end now?",
            }
        )
        return AdvisorVerdict(self.name, bool(result.decision), result.reason)


class LLMSchedulingAdvisor:
    """Sched / Don't Sched, then function-calls the SQL store when it says yes."""

    name = AdvisorName.SCHED

    def __init__(self, store: ScheduleStore, model: str | None = None):
        self.store = store
        self.model = model
        self.few_shot = prompts.few_shot_block("schedule", "YES - schedule", "NO - not yet")

    def evaluate(self, context: ConversationContext) -> AdvisorVerdict:
        chain = _binary_chain(prompts.SCHED_ADVISOR_SYSTEM, self.few_shot, self.model)
        result = chain.invoke(
            {
                "history": prompts.render_history(context.history),
                "anchor": context.anchor.strftime("%A %d %B %Y"),
                "extra": "Should the next message propose or confirm an interview time?",
            }
        )
        if not result.decision:
            return AdvisorVerdict(self.name, False, result.reason)

        # Only now is SQL queried.
        return self._with_slots(context, result.reason)

    def _with_slots(self, context: ConversationContext, reason: str) -> AdvisorVerdict:
        """Phase 2: the SQL calendar, via function calling.

        Only reached on a yes. The slots attached below are the rows the tool
        calls returned - never anything the model wrote in prose.
        """
        offers = dates.parse_offers(context.last_candidate_message, context.anchor)

        # A local read, so the composer's "not available" prefix is reliable
        # whichever tools the model chose to call.
        if offers:
            date, time = offers[0]
            exact = self.store.check_slot(date, time, context.position) if time else None
            if exact is None:
                reason += f"; {date:%A %d %b} {time or ''} is not on the calendar"
            elif not exact.available:
                reason += f"; {date:%A %d %b} {time} is already booked"

        toolkit = build_schedule_toolkit(self.store, context)
        slots: list = []
        try:
            request = (
                f"Conversation so far:\n{prompts.render_history(context.history)}\n\n"
                f"Today is {context.anchor:%A %d %B %Y}.\n"
                f"The candidate's last message: "
                f"{context.last_candidate_message or '(nothing yet)'}\n\n"
                "Find up to three interview slots that can actually be offered."
            )
            self._agent(toolkit).invoke(
                {"messages": [{"role": "user", "content": request}]},
                config={"recursion_limit": 8},
            )
            slots = toolkit.unique_slots(3)
        except Exception as exc:  # tool calling is a network call; never fail the turn
            reason += f"; tool calling unavailable ({type(exc).__name__})"

        if not slots:
            # Fall back rather than degrade.
            slots = toolkit.fallback_slots(3)
        if toolkit.call_log:
            reason += f"; via {len(toolkit.call_log)} SQL tool call(s)"

        return AdvisorVerdict(self.name, True, reason, slots=slots, proposed=offers)

    def _agent(self, toolkit):
        """Tool-calling agent bound to this turn's schedule tools.

        create_agent is the LangChain 1.x API; AgentExecutor is deprecated.
        """
        return create_agent(
            _chat(self.model, **prompts.TOOL_AGENT_PARAMS),
            toolkit.as_tools(),
            system_prompt=prompts.SCHED_TOOL_SYSTEM,
        )


class LLMInfoAdvisor:
    """Info Needed / Not Needed, then queries Chroma when it says yes."""

    name = AdvisorName.INFO

    def __init__(self, retriever=None, model: str | None = None):
        self.retriever = retriever
        self.model = model
        self.few_shot = ""

    def evaluate(self, context: ConversationContext) -> AdvisorVerdict:
        chain = _binary_chain(prompts.INFO_ADVISOR_SYSTEM, self.few_shot, self.model)
        result = chain.invoke(
            {
                "history": prompts.render_history(context.history, 4),
                "anchor": context.anchor.strftime("%A %d %B %Y"),
                "extra": "Does answering the last message need facts about the role?",
            }
        )
        if not result.decision:
            return AdvisorVerdict(self.name, False, result.reason)

        retrieved = []
        if self.retriever is not None:
            retrieved = self.retriever.search(context.last_candidate_message, k=3)
        return AdvisorVerdict(self.name, True, result.reason, retrieved=retrieved)


# --- main agent parts -----------------------------------------------------


class LLMRouter:
    """The 'Decides 1 out of 3 Options' and 'Decides 1 out of 2 Options' diamonds."""

    def __init__(self, model: str | None = None):
        self.model = model

    # Only ever offer advisors that haven't been consulted this turn.
    PRIORITY = (AdvisorName.EXIT, AdvisorName.SCHED, AdvisorName.INFO)

    def choose(self, context: ConversationContext, consulted) -> AdvisorName | None:
        seen = {v.advisor for v in consulted}
        remaining = [a for a in self.PRIORITY if a not in seen]
        if not remaining:
            return None

        template = ChatPromptTemplate.from_messages(
            [
                ("system", prompts.ROUTER_SYSTEM),
                (
                    "human",
                    "Conversation so far:\n{history}\n\n"
                    "Advisors already consulted this turn: {already}\n"
                    "Their findings:\n{findings}\n\n"
                    "You may still consult: {remaining}.\n"
                    "Which advisor next? Answer with one of {remaining}, or none.",
                ),
            ]
        )
        chain = template | _chat(self.model, **prompts.CLASSIFIER_PARAMS).with_structured_output(
            RouteChoice
        )
        result = chain.invoke(
            {
                "history": prompts.render_history(context.history),
                "already": ", ".join(a.value for a in seen) or "none yet",
                "findings": "\n".join(v.summary() for v in consulted) or "(none)",
                "remaining": ", ".join(a.value for a in remaining),
            }
        )
        answer = (result.advisor or "").strip().lower()
        if answer == "none":
            # Finishing a turn without ever asking "should we propose a time?"
            # is how every schedule turn gets missed.
            return AdvisorName.SCHED if AdvisorName.SCHED in remaining else None
        mapping = {a.value: a for a in AdvisorName}
        choice = mapping.get(answer)
        # Naming an exhausted advisor still means a question is open.
        if choice is None or choice in seen:
            return remaining[0]
        return choice

    def consult_again(self, context, consulted) -> bool:
        # A yes settles the turn; a no may still leave a question open.
        return not consulted[-1].decision


class LLMComposer:
    def __init__(self, model: str | None = None):
        self.model = model

    def compose(self, context: ConversationContext, consulted, action: Action, booking=None) -> str:
        slots, retrieved = [], []
        for verdict in consulted:
            if verdict.slots:
                slots = verdict.slots
            if verdict.retrieved:
                retrieved = verdict.retrieved

        template = ChatPromptTemplate.from_messages(
            [
                ("system", prompts.COMPOSER_SYSTEM),
                (
                    "human",
                    "Conversation so far:\n{history}\n\n"
                    "Decision for this turn: {action}\n"
                    "Advisor findings:\n{findings}\n\n"
                    "Available interview slots (use these exact times, if any):\n{slots}\n\n"
                    "Retrieved role information (use only this, if any):\n{retrieved}\n\n"
                    "Write the next SMS.",
                ),
            ]
        )
        chain = template | _chat(self.model, **prompts.COMPOSER_PARAMS)
        result = chain.invoke(
            {
                "history": prompts.render_history(context.history),
                "action": action.value,
                "findings": "\n".join(v.summary() for v in consulted) or "(none)",
                "slots": "\n".join(s.label() for s in slots) or "(none)",
                "retrieved": "\n".join(retrieved) or "(none)",
            }
        )
        return result.content.strip()


def build_llm_agent(store: ScheduleStore, retriever=None, exit_model: str | None = None):
    """The full LangChain agent. `exit_model` swaps in the fine-tuned advisor."""
    from .booking import SlotBooker
    from .main_agent import MainAgent

    return MainAgent(
        router=LLMRouter(),
        advisors={
            AdvisorName.EXIT: LLMExitAdvisor(model=exit_model),
            AdvisorName.SCHED: LLMSchedulingAdvisor(store),
            AdvisorName.INFO: LLMInfoAdvisor(retriever),
        },
        composer=LLMComposer(),
        booker=SlotBooker(store),
    )
