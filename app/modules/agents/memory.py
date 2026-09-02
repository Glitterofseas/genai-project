"""Conversation memory, backed by LangChain chat message history.

The CLI and Streamlit app keep the live conversation here and build the
advisors' context from it.

Not used by the evaluation harness: that replays recorded transcripts, so
memory carrying state between them would quietly shift the scores.
"""
from __future__ import annotations

import datetime as dt

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from ..db.rules import DEFAULT_POSITION
from ..evaluation.dataset import Turn
from .types import ConversationContext

# Recruiter maps to the assistant role, candidate to the human.
RECRUITER = "recruiter"
CANDIDATE = "candidate"

TURN_GAP = dt.timedelta(minutes=2)


class ConversationMemory:
    """One candidate conversation, held as LangChain messages."""

    def __init__(
        self,
        anchor: dt.datetime,
        position: str = DEFAULT_POSITION,
        candidate_phone: str = "",
    ):
        self.history = InMemoryChatMessageHistory()
        self.anchor = anchor
        self.position = position
        self.candidate_phone = candidate_phone
        self._turn_id = 0

    # ---------- writing ----------

    def add(self, speaker: str, text: str) -> Turn:
        self._turn_id += 1
        if self._turn_id > 1:
            self.anchor += TURN_GAP
        message_cls = AIMessage if speaker == RECRUITER else HumanMessage
        self.history.add_message(
            message_cls(
                content=text,
                additional_kwargs={
                    "speaker": speaker,
                    "turn_id": self._turn_id,
                    "timestamp": self.anchor.isoformat(),
                },
            )
        )
        return self.turns[-1]

    def add_recruiter(self, text: str) -> Turn:
        return self.add(RECRUITER, text)

    def add_candidate(self, text: str) -> Turn:
        return self.add(CANDIDATE, text)

    def clear(self) -> None:
        self.history.clear()
        self._turn_id = 0

    # ---------- reading ----------

    @property
    def messages(self) -> list[BaseMessage]:
        return list(self.history.messages)

    @property
    def turns(self) -> tuple[Turn, ...]:
        """The stored messages as the Turn records the advisors expect."""
        out = []
        for message in self.history.messages:
            meta = message.additional_kwargs
            out.append(
                Turn(
                    turn_id=meta.get("turn_id", 0),
                    speaker=meta.get(
                        "speaker",
                        RECRUITER if isinstance(message, AIMessage) else CANDIDATE,
                    ),
                    timestamp=dt.datetime.fromisoformat(
                        meta.get("timestamp", self.anchor.isoformat())
                    ),
                    text=str(message.content),
                    label=None,
                )
            )
        return tuple(out)

    def context(self) -> ConversationContext:
        """What the Main Agent and its advisors are given."""
        return ConversationContext(
            history=self.turns,
            anchor=self.anchor,
            position=self.position,
            candidate_phone=self.candidate_phone,
        )

    def __len__(self) -> int:
        return len(self.history.messages)
