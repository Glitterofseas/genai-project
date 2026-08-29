"""Loading and slicing sms_conversations.json.

Shape of the data (verified):
    15 conversations, 103 turns - 59 recruiter, 44 candidate.
    Only recruiter turns carry a label: continue 25 / schedule 19 / end 15.
    Every conversation's final recruiter turn is labelled 'end'.

That last fact matters for evaluation: a rule as dumb as "the last turn is
always end" scores 15/15 on the end class, so `end` accuracy alone proves
little. See the trivial baseline in the evaluation report.
"""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from typing import Iterator, Literal

from ..config.settings import (
    CONVERSATIONS_PATH,
    TEST_CONVERSATION_IDS,
    TRAIN_CONVERSATION_IDS,
)

Action = Literal["continue", "schedule", "end"]
ACTIONS: tuple[Action, ...] = ("continue", "schedule", "end")


@dataclass(frozen=True)
class Turn:
    turn_id: int
    speaker: str
    timestamp: dt.datetime
    text: str
    label: Action | None

    @property
    def is_recruiter(self) -> bool:
        return self.speaker == "recruiter"


@dataclass(frozen=True)
class Conversation:
    conversation_id: int
    candidate_phone: str
    recruiter_phone: str
    start_time: dt.datetime
    turns: tuple[Turn, ...]

    @property
    def labelled_turns(self) -> tuple[Turn, ...]:
        return tuple(t for t in self.turns if t.label is not None)

    def history_before(self, turn_id: int) -> tuple[Turn, ...]:
        """Turns preceding `turn_id`.

        The evaluation is teacher-forced: each prediction sees the *recorded*
        history, not the agent's own earlier output. That keeps every turn an
        independent classification and stops one early mistake from cascading.
        """
        return tuple(t for t in self.turns if t.turn_id < turn_id)


def _parse_ts(raw: str) -> dt.datetime:
    return dt.datetime.fromisoformat(raw.replace("Z", "+00:00")).replace(tzinfo=None)


def load_conversations(path=CONVERSATIONS_PATH) -> list[Conversation]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    conversations = []
    for item in raw:
        turns = tuple(
            Turn(
                turn_id=t["turn_id"],
                speaker=t["speaker"],
                timestamp=_parse_ts(t["timestamp_utc"]),
                text=t["text"],
                label=t.get("label"),
            )
            for t in item["turns"]
        )
        conversations.append(
            Conversation(
                conversation_id=item["conversation_id"],
                candidate_phone=item["candidate_phone"],
                recruiter_phone=item["recruiter_phone"],
                start_time=_parse_ts(item["start_time_utc"]),
                turns=turns,
            )
        )
    return conversations


def split(conversations: list[Conversation] | None = None):
    """The fixed 10/5 split. Committed in settings so results are reproducible."""
    conversations = conversations or load_conversations()
    by_id = {c.conversation_id: c for c in conversations}
    train = [by_id[i] for i in TRAIN_CONVERSATION_IDS if i in by_id]
    test = [by_id[i] for i in TEST_CONVERSATION_IDS if i in by_id]
    return train, test


def iter_labelled(conversations: list[Conversation]) -> Iterator[tuple[Conversation, Turn]]:
    """Every (conversation, labelled recruiter turn) pair - one eval example each."""
    for conversation in conversations:
        for turn in conversation.labelled_turns:
            yield conversation, turn


def describe(conversations: list[Conversation]) -> dict:
    counts = {a: 0 for a in ACTIONS}
    for _, turn in iter_labelled(conversations):
        counts[turn.label] += 1
    return {
        "conversations": len(conversations),
        "turns": sum(len(c.turns) for c in conversations),
        "labelled_turns": sum(counts.values()),
        "labels": counts,
    }
