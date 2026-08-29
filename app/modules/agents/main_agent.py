"""The Main Agent: one turn of the conversation.

Implements the loop drawn in docs/workflow/ exactly:

    Receives and Process Input
        -> Decides 1 of 3 Options          (route to ONE advisor)
        -> advisor processes chat history, returns a binary verdict
        -> Receives and Process Input
        -> Decides 1 of 2 Options          (consult again | answer the user)
        -> Sends Output to User

The loop is capped so a router that keeps asking for advice cannot spin
forever - and so per-turn cost stays bounded.
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
    """The Main Agent's two decision diamonds."""

    def choose(
        self, context: ConversationContext, consulted: list[AdvisorVerdict]
    ) -> AdvisorName | None:
        """Which advisor to consult next; None to answer immediately."""

    def consult_again(
        self, context: ConversationContext, consulted: list[AdvisorVerdict]
    ) -> bool:
        """After a verdict: loop back, or send output to the user?"""


class Composer(Protocol):
    def compose(
        self,
        context: ConversationContext,
        consulted: list[AdvisorVerdict],
        action: Action,
        booking=None,
    ) -> str:
        """The SMS the candidate actually receives."""


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

        # A conversation that ends because a slot was accepted must actually
        # write that booking, and must not confirm a time the calendar does not
        # have. Booking runs only on END and never changes the action, so the
        # evaluation is unaffected by it.
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

    def predict_action(self, context: ConversationContext) -> Action:
        """Action only - used by the evaluation harness."""
        return self.decide(context).action
