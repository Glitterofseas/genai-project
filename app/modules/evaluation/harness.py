"""Replay the labelled conversations and score the agent.

Teacher-forced: each turn is predicted from the recorded history before it, not
from the agent's own earlier replies, so turns stay independent and one early
mistake can't cascade.

The store is opened read-only, or replaying would book slots and the second run
would score differently from the first.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..agents.types import Action, ConversationContext
from .dataset import ACTIONS, Conversation, Turn, iter_labelled


@dataclass
class Prediction:
    conversation_id: int
    turn_id: int
    expected: str
    predicted: str
    advisors: list[str] = field(default_factory=list)
    message: str = ""

    @property
    def correct(self) -> bool:
        return self.expected == self.predicted


@dataclass
class Report:
    name: str
    predictions: list[Prediction]

    @property
    def accuracy(self) -> float:
        if not self.predictions:
            return 0.0
        return sum(p.correct for p in self.predictions) / len(self.predictions)

    def confusion(self) -> dict[str, dict[str, int]]:
        matrix = {a: {b: 0 for b in ACTIONS} for a in ACTIONS}
        for p in self.predictions:
            matrix[p.expected][p.predicted] += 1
        return matrix

    def per_class(self) -> dict[str, dict[str, float]]:
        matrix = self.confusion()
        out = {}
        for action in ACTIONS:
            tp = matrix[action][action]
            fn = sum(matrix[action][b] for b in ACTIONS) - tp
            fp = sum(matrix[a][action] for a in ACTIONS) - tp
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            out[action] = {
                "precision": precision, "recall": recall, "f1": f1,
                "support": tp + fn,
            }
        return out

    def render(self) -> str:
        lines = [
            f"{self.name}",
            f"  accuracy: {self.accuracy:.1%}  ({sum(p.correct for p in self.predictions)}"
            f"/{len(self.predictions)} turns)",
            "",
            "  confusion matrix (rows = expected, cols = predicted)",
            "                " + "".join(f"{a:>10}" for a in ACTIONS),
        ]
        matrix = self.confusion()
        for expected in ACTIONS:
            row = "".join(f"{matrix[expected][p]:>10}" for p in ACTIONS)
            lines.append(f"  {expected:>12}  {row}")
        lines.append("")
        lines.append(f"  {'class':>12}{'prec':>9}{'recall':>9}{'f1':>9}{'n':>6}")
        for action, stats in self.per_class().items():
            lines.append(
                f"  {action:>12}{stats['precision']:>9.2f}{stats['recall']:>9.2f}"
                f"{stats['f1']:>9.2f}{stats['support']:>6}"
            )
        return "\n".join(lines)

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "name": self.name,
                    "accuracy": self.accuracy,
                    "confusion": self.confusion(),
                    "per_class": self.per_class(),
                    "predictions": [asdict(p) for p in self.predictions],
                },
                indent=2,
            ),
            encoding="utf-8",
        )


def context_for(conversation: Conversation, turn: Turn) -> ConversationContext:
    """The teacher-forced context for one labelled turn."""
    return ConversationContext(
        history=conversation.history_before(turn.turn_id),
        anchor=turn.timestamp,
        candidate_phone=conversation.candidate_phone,
    )


def evaluate(agent, conversations: list[Conversation], name: str = "agent") -> Report:
    predictions = []
    for conversation, turn in iter_labelled(conversations):
        decision = agent.decide(context_for(conversation, turn))
        predictions.append(
            Prediction(
                conversation_id=conversation.conversation_id,
                turn_id=turn.turn_id,
                expected=turn.label,
                predicted=decision.action.value,
                advisors=decision.advisors_used,
                message=decision.message,
            )
        )
    return Report(name=name, predictions=predictions)


# Trivial baselines. Every conversation's last turn is `end`, so a one-line
# rule already scores 100% on that class - without these for comparison, a good
# confusion matrix reads as more than it is.


class _ConstantAgent:
    def __init__(self, action: Action):
        self.action = action

    def decide(self, context):
        from ..agents.types import AgentDecision

        return AgentDecision(action=self.action, message="")


class _LastTurnIsEndAgent:
    """Predicts `end` when no candidate reply follows - i.e. the thread is over."""

    def __init__(self, conversations: list[Conversation]):
        self.last_turn = {
            c.conversation_id: max(t.turn_id for t in c.labelled_turns)
            for c in conversations
        }
        self.by_history = {}
        for conversation, turn in iter_labelled(conversations):
            self.by_history[(conversation.conversation_id, turn.turn_id)] = (
                turn.turn_id == self.last_turn[conversation.conversation_id]
            )

    def decide_for(self, conversation_id: int, turn_id: int) -> Action:
        return (
            Action.END
            if self.by_history.get((conversation_id, turn_id))
            else Action.CONTINUE
        )


def baseline_reports(conversations: list[Conversation]) -> list[Report]:
    """Floor scores every real system must clear."""
    reports = []

    for action in (Action.CONTINUE, Action.SCHEDULE, Action.END):
        agent = _ConstantAgent(action)
        reports.append(evaluate(agent, conversations, name=f"baseline: always '{action.value}'"))

    positional = _LastTurnIsEndAgent(conversations)
    predictions = []
    for conversation, turn in iter_labelled(conversations):
        predicted = positional.decide_for(conversation.conversation_id, turn.turn_id)
        predictions.append(
            Prediction(
                conversation_id=conversation.conversation_id,
                turn_id=turn.turn_id,
                expected=turn.label,
                predicted=predicted.value,
            )
        )
    reports.append(Report(name="baseline: last turn = end, else continue", predictions=predictions))
    return reports
