"""Loading and slicing sms_conversations.json.

15 conversations, 103 turns. Only the 59 recruiter turns are labelled:
continue 25, schedule 19, end 15.

Every conversation's last recruiter turn is `end`, so "last turn is always end"
scores 15/15 on that class - which is why `end` accuracy alone proves little.
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
        """Turns before `turn_id` - the recorded ones, not the agent's own."""
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
