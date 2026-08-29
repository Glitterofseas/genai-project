"""Rule-based advisors: a free, deterministic implementation of the same contracts.

Two jobs:

1. Development. The whole orchestration loop can be built and tested without
   spending a single token, and without an API key present.
2. Evaluation baseline. Any LLM advisor must beat these to have earned its cost.
   Reporting the LLM against a real baseline - rather than against nothing - is
   what stops a mediocre score from looking impressive.
"""
from __future__ import annotations

import datetime as dt
import re

from ..db import dates
from ..db.store import ScheduleStore
from .types import (
    Action,
    AdvisorName,
    AdvisorVerdict,
    ConversationContext,
)

DISINTEREST = re.compile(
    r"no longer interested|not interested|remove me|stop texting|unsubscribe"
    r"|found (another|a) job|accepted (another|an) offer|withdraw|take me off",
    re.IGNORECASE,
)
ACCEPTANCE = re.compile(
    r"\b(works|work for me|sounds (great|good)|that works|is good|perfect"
    r"|see you then|yes,? absolutely|confirmed|book it)\b",
    re.IGNORECASE,
)
QUESTION = re.compile(
    r"\?|could you (share|tell)|what (is|are|kind)|can i get more|more (details|about)"
    r"|tell me about",
    re.IGNORECASE,
)
SCHEDULING_INTENT = re.compile(
    r"\b(schedule|appointment|interview|meet|meeting|availability|calendar)\b",
    re.IGNORECASE,
)
REJECTION = re.compile(
    r"can'?t|cannot|unavailable|don'?t work|doesn'?t work|busy|other commitments"
    r"|another time|other times|not good for me",
    re.IGNORECASE,
)


def previously_offered(context: ConversationContext) -> set[dt.datetime]:
    """Every slot the recruiter has already put on the table this conversation.

    Re-offering a time the candidate has just declined reads as not listening,
    and candidates decline at least once in a third of the transcripts.
    """
    offered: set[dt.datetime] = set()
    for turn in context.history:
        if turn.speaker != "recruiter":
            continue
        for date, time in dates.parse_offers(turn.text, context.anchor):
            if time is not None:
                offered.add(dt.datetime.combine(date, time))
    return offered


class RuleBasedExitAdvisor:
    """End / Don't End.

    Fires on two distinct situations, because the labels contain both:
      * the candidate has disengaged, and
      * the candidate has just accepted a slot, so the thread is done.
    """

    name = AdvisorName.EXIT

    def evaluate(self, context: ConversationContext) -> AdvisorVerdict:
        last = context.last_candidate_message
        if DISINTEREST.search(last):
            return AdvisorVerdict(self.name, True, "candidate has disengaged")
        # A rejection outranks an acceptance: "Those slots don't work for me"
        # contains "work for me", and reading that as a yes ends the
        # conversation at exactly the moment the candidate is still trying to
        # find a time.
        if REJECTION.search(last):
            return AdvisorVerdict(self.name, False, "candidate declined the proposed time")
        if ACCEPTANCE.search(last) and self._slot_was_offered(context):
            return AdvisorVerdict(self.name, True, "candidate accepted a proposed slot")
        return AdvisorVerdict(self.name, False, "candidate is still engaged")

    @staticmethod
    def _slot_was_offered(context: ConversationContext) -> bool:
        for turn in reversed(context.history):
            if turn.speaker == "recruiter":
                return bool(dates.parse_offers(turn.text, context.anchor))
        return False


class RuleBasedSchedulingAdvisor:
    """Sched / Don't Sched. Queries SQL only when it decides to schedule."""

    name = AdvisorName.SCHED

    def __init__(self, store: ScheduleStore):
        self.store = store

    def evaluate(self, context: ConversationContext) -> AdvisorVerdict:
        last = context.last_candidate_message
        if not last:
            return AdvisorVerdict(self.name, False, "conversation has not started")

        offers = dates.parse_offers(last, context.anchor)
        already_offered = self._recruiter_has_offered(context)

        # Scheduling is the recruiter's initiative, not a reply to a request:
        # 10 of the 19 'schedule' turns land immediately after the candidate
        # first describes their experience, with no prompting at all. The spec
        # says so too - the bot must "drive the conversation toward the end
        # goal: scheduling an interview".
        if already_offered and REJECTION.search(last):
            rationale = "candidate rejected the proposed slot; offer alternatives"
        elif offers or SCHEDULING_INTENT.search(last):
            rationale = "candidate raised timing"
        elif not already_offered:
            rationale = "candidate has answered; time to propose an interview"
        else:
            return AdvisorVerdict(self.name, False, "a slot is already on the table")

        # Only now does the SQL store get touched, per the workflow diagram.
        target = None
        if offers:
            date, time = offers[0]
            target = dt.datetime.combine(date, time or dt.time(9))
        slots = self.store.nearest_available(
            target or context.anchor,
            position=context.position,
            n=3,
            not_before=context.anchor,
            exclude=previously_offered(context),
        )
        if offers:
            date, time = offers[0]
            exact = self.store.check_slot(date, time, context.position) if time else None
            if exact is None:
                rationale += f"; {date:%A} {time or ''} is not on the calendar"
            elif not exact.available:
                rationale += f"; {date:%A} {time} is already taken"
        return AdvisorVerdict(self.name, True, rationale, slots=slots, proposed=offers)

    @staticmethod
    def _recruiter_has_offered(context: ConversationContext) -> bool:
        """Has a concrete slot already been put on the table?"""
        return any(
            turn.speaker == "recruiter" and dates.parse_offers(turn.text, context.anchor)
            for turn in context.history
        )


class RuleBasedInfoAdvisor:
    """Info Needed / Not Needed. Queries the vector store only when needed."""

    name = AdvisorName.INFO

    def __init__(self, retriever=None):
        self.retriever = retriever

    def evaluate(self, context: ConversationContext) -> AdvisorVerdict:
        last = context.last_candidate_message
        if not QUESTION.search(last):
            return AdvisorVerdict(self.name, False, "no question asked")
        retrieved: list[str] = []
        if self.retriever is not None:
            retrieved = self.retriever.search(last, k=3)
        return AdvisorVerdict(
            self.name, True, "candidate asked about the role", retrieved=retrieved
        )


class RuleBasedRouter:
    """The Main Agent's two diamonds, as deterministic rules."""

    def choose(self, context, consulted):
        seen = {v.advisor for v in consulted}
        if AdvisorName.EXIT not in seen:
            return AdvisorName.EXIT
        if AdvisorName.SCHED not in seen:
            return AdvisorName.SCHED
        if AdvisorName.INFO not in seen:
            return AdvisorName.INFO
        return None

    def consult_again(self, context, consulted):
        # A positive verdict settles the turn; a negative one sends us back.
        return not consulted[-1].decision


class TemplateComposer:
    """Canned SMS copy. The LLM composer replaces this; the shape is identical."""

    def compose(self, context, consulted, action, booking=None) -> str:
        if action is Action.END:
            reason = next(
                (v.rationale for v in consulted if v.advisor is AdvisorName.EXIT), ""
            )
            if "disengaged" in reason:
                return "Understood - I'll close your application for now. Best of luck!"
            return self._closing(booking)

        if action is Action.SCHEDULE:
            verdict = next(v for v in consulted if v.advisor is AdvisorName.SCHED)
            if not verdict.slots:
                return "I couldn't find an opening near that time - could you suggest another day?"
            options = " or ".join(s.label() for s in verdict.slots[:2])
            prefix = ""
            if "not on the calendar" in verdict.rationale or "already taken" in verdict.rationale:
                prefix = "That time isn't available, unfortunately. "
            return f"{prefix}Could we schedule your interview for {options}?"

        return self._informational(consulted)

    @staticmethod
    def _closing(booking) -> str:
        """Only claim a confirmation when a slot was genuinely written."""
        from .booking import BookingStatus

        if booking is None or booking.status is BookingStatus.UNCLEAR:
            # No specific slot could be resolved, so promise a confirmation of
            # the time rather than implying one has already been made.
            return "Great - I'll confirm the exact time and send your calendar invite shortly."
        if booking.status is BookingStatus.BOOKED:
            return (
                f"Great, your interview is confirmed for {booking.slot.label()}. "
                "You'll receive a calendar invite shortly."
            )
        alternatives = booking.alternatives or []
        options = " or ".join(s.label() for s in alternatives[:2])
        problem = (
            "we don't have that day in the calendar"
            if booking.status is BookingStatus.NOT_ON_CALENDAR
            else "that slot has just been taken"
        )
        if not options:
            return f"Apologies - {problem}. Could you suggest another time?"
        return f"Apologies - {problem}. Could we do {options} instead?"

    @staticmethod
    def _informational(consulted) -> str:
        info = next(
            (v for v in consulted if v.advisor is AdvisorName.INFO and v.decision), None
        )
        if info and info.retrieved:
            return f"{info.retrieved[0][:220]} Does that help? Shall we find a time to talk?"
        return "Thanks for the detail. Could you tell me more about your Python experience?"


def build_rule_based_agent(store: ScheduleStore, retriever=None):
    """A complete, zero-cost agent implementing the full workflow."""
    from .booking import SlotBooker
    from .main_agent import MainAgent

    return MainAgent(
        router=RuleBasedRouter(),
        advisors={
            AdvisorName.EXIT: RuleBasedExitAdvisor(),
            AdvisorName.SCHED: RuleBasedSchedulingAdvisor(store),
            AdvisorName.INFO: RuleBasedInfoAdvisor(retriever),
        },
        composer=TemplateComposer(),
        booker=SlotBooker(store),
    )
