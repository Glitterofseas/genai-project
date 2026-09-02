"""The Main Agent: one turn of the conversation.

Route to one advisor, read its verdict, then either consult another or reply -
the loop from docs/workflow/. Capped so it can't spin, and to bound cost.
"""
from __future__ import annotations

from typing import Mapping, Protocol

from .types import (
    Action,
    AdvisorName,
    AdvisorVerdict,
    AgentDecision,
    ConversationContext,
    action_from_verdicts,
)

MAX_ADVISOR_CALLS = 3


class Router(Protocol):
    """The two decision points: which advisor, then whether to keep going."""

    def choose(
        self, context: ConversationContext, consulted: list[AdvisorVerdict]
    ) -> AdvisorName | None:
        """Which advisor next, or None to reply now."""

    def consult_again(
        self, context: ConversationContext, consulted: list[AdvisorVerdict]
    ) -> bool:
        """Loop back for more advice, or answer the candidate?"""


class Composer(Protocol):
    def compose(
        self,
        context: ConversationContext,
        consulted: list[AdvisorVerdict],
        action: Action,
        booking=None,
    ) -> str:
        """The SMS the candidate receives."""


class MainAgent:
    def __init__(
        self,
        router: Router,
        advisors: Mapping[AdvisorName, object],
        composer: Composer,
        max_advisor_calls: int = MAX_ADVISOR_CALLS,
        booker=None,
    ):
        self.router = router
        self.advisors = advisors
        self.composer = composer
        self.max_advisor_calls = max_advisor_calls
        self.booker = booker

    def decide(self, context: ConversationContext) -> AgentDecision:
        consulted: list[AdvisorVerdict] = []

        for _ in range(self.max_advisor_calls):
            choice = self.router.choose(context, consulted)
            if choice is None:
                break
            advisor = self.advisors.get(choice)
            if advisor is None:
                break
            consulted.append(advisor.evaluate(context))
            if not self.router.consult_again(context, consulted):
                break

        action = action_from_verdicts(consulted)

        # Ending on an accepted slot has to actually write the booking, and
        # must not confirm a time the calendar lacks. Only runs on END, and
        # never changes the action, so evaluation is unaffected.
        booking = None
        if action is Action.END and self.booker is not None:
            booking = self.booker.book_from(context)

        message = self.composer.compose(context, consulted, action, booking)

        return AgentDecision(
            action=action,
            message=message,
            consulted=consulted,
            booked_slot=booking.slot if booking else None,
            booking=booking,
        )
