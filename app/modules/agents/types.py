"""Shared contracts for the Main Agent and its Advisors.

The shapes here follow the provided workflow diagram literally:

  * The Main Agent routes to exactly ONE advisor per iteration, receives its
    output, then decides whether to consult again or answer the candidate.
  * Every advisor returns a BINARY verdict:
        Exit  -> End Conversation / Don't End Conv
        Sched -> Sched / Don't Sched      (SQL is queried only when Sched)
        Info  -> Info Needed / Info Not Needed  (vector store only when Needed)
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
    """Everything an advisor is allowed to see.

    `anchor` is the conversation's own clock - the timestamp of the turn being
    answered. All relative-date resolution hangs off it, never off wall time.
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
    """A single advisor's binary answer plus whatever it retrieved."""

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
    """Collapse advisor verdicts into the one action the turn is graded on.

    Exit outranks Sched: once ending is appropriate - whether the candidate has
    just accepted a slot or has just declined - the turn is an `end`. That
    ordering is what the labelled data shows, where 11 of 15 `end` turns follow
    a successful booking rather than a rejection.
    """
    for verdict in verdicts:
        if verdict.advisor is AdvisorName.EXIT and verdict.decision:
            return Action.END
    for verdict in verdicts:
        if verdict.advisor is AdvisorName.SCHED and verdict.decision:
            return Action.SCHEDULE
    return Action.CONTINUE
