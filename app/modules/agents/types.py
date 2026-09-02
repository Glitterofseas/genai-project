"""Shared types for the Main Agent and its advisors.

One advisor per iteration, each returning a yes/no verdict, as in the workflow
diagram. SQL and the vector store are only touched on a yes.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

from ..db.store import Slot
from ..evaluation.dataset import Turn


class Action(str, Enum):
    CONTINUE = "continue"
    SCHEDULE = "schedule"
    END = "end"


class AdvisorName(str, Enum):
    EXIT = "exit"
    SCHED = "sched"
    INFO = "info"


@dataclass
class ConversationContext:
    """What an advisor sees.

    anchor is the conversation's clock, not wall time - "next Friday" resolves
    against it.
    """

    history: tuple[Turn, ...]
    anchor: dt.datetime
    position: str = "Python Dev"
    candidate_phone: str = ""

    @property
    def last_candidate_message(self) -> str:
        for turn in reversed(self.history):
            if turn.speaker == "candidate":
                return turn.text
        return ""

    def transcript(self, limit: int | None = None) -> str:
        turns = self.history[-limit:] if limit else self.history
        return "\n".join(f"{t.speaker}: {t.text}" for t in turns)


@dataclass
class AdvisorVerdict:
    """An advisor's yes/no answer, plus anything it retrieved."""

    advisor: AdvisorName
    decision: bool
    rationale: str = ""
    slots: list[Slot] = field(default_factory=list)
    retrieved: list[str] = field(default_factory=list)
    proposed: list[tuple[dt.date, dt.time | None]] = field(default_factory=list)

    def summary(self) -> str:
        verb = {
            AdvisorName.EXIT: ("end the conversation", "do not end"),
            AdvisorName.SCHED: ("schedule now", "do not schedule"),
            AdvisorName.INFO: ("information is needed", "no information needed"),
        }[self.advisor]
        head = f"{self.advisor.value} advisor: {verb[0] if self.decision else verb[1]}"
        if self.rationale:
            head += f" - {self.rationale}"
        if self.slots:
            head += "\n  available: " + "; ".join(s.label() for s in self.slots)
        if self.retrieved:
            head += "\n  retrieved: " + " ".join(self.retrieved)[:600]
        return head


@dataclass
class AgentDecision:
    action: Action
    message: str
    consulted: list[AdvisorVerdict] = field(default_factory=list)
    booked_slot: Slot | None = None
    booking: object | None = None   # BookingResult; typed loosely to avoid a cycle

    @property
    def advisors_used(self) -> list[str]:
        return [v.advisor.value for v in self.consulted]


@runtime_checkable
class Advisor(Protocol):
    name: AdvisorName

    def evaluate(self, context: ConversationContext) -> AdvisorVerdict:
        ...


def action_from_verdicts(verdicts: list[AdvisorVerdict]) -> Action:
    """Collapse the verdicts into the action this turn is graded on.

    Exit wins over Sched. In the labelled data 11 of 15 `end` turns follow an
    accepted slot rather than a rejection, so ending covers both.
    """
    for verdict in verdicts:
        if verdict.advisor is AdvisorName.EXIT and verdict.decision:
            return Action.END
    for verdict in verdicts:
        if verdict.advisor is AdvisorName.SCHED and verdict.decision:
            return Action.SCHEDULE
    return Action.CONTINUE
